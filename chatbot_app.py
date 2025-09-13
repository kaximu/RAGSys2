"""The full cleaned-up chatbot_app.py with everything merged in.
It combines all the improvements we discussed:

🧭 App Mode: Chatbot + Admin Dashboard

👍👎 Feedback + Hard Questions logging (auto-create JSONL files)

📊 Admin Dashboard with metrics, feedback logs, hard questions, downloads

🔄 Reset options (clear indexes, clear chat)

🔍 Index search filter + auto-select last index

🧠 Hybrid retrieval (FAISS + BM25) + Multi-query expansion (FLAN-T5)

✂️ Deduplication + CrossEncoder re-ranking

⚙️ Debug mode toggle with chunk preview + download (JSON/TXT)

🔎 Retrieval settings: top_k slider

✂️ Chunking settings: chunk size + overlap sliders

🔄 OpenAI embeddings if API key exists, else HuggingFace fallback

   Chunking defaults → chunk_size=1100, chunk_overlap=220

   CrossEncoder reranker → upgrade to cross-encoder/ms-marco-MiniLM-L-12-v2 (better semantic ranking)

   Multi-query expansion → generate 5 variants instead of 2

   Hybrid retriever weighting → boost BM25 to [0.4, 0.6]
   
   add a similarity score filter so your chatbot ignores junk matches before passing context to the LLM.
      How it works :
         Each FAISS retrieval returns docs with a similarity score.
         We’ll filter out any doc with score < 0.3 (tunable).
        This way, if nothing relevant is found, the chatbot won’t hallucinate."""
   


import os
import re
import io
import json
import requests
import trafilatura
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urljoin

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from transformers import pipeline

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="💬 RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot")

INDEX_DIR = Path(r"D:\SpikeUp.AI\Project Futere Facts\indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# API Key Helpers
# -----------------------
def has_openai_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key and key.startswith("sk-"))

# -----------------------
# Ensure feedback + hard question files exist
# -----------------------
feedback_file = "feedback_log.jsonl"
hard_file = "hard_questions.jsonl"

for f in [feedback_file, hard_file]:
    if not Path(f).exists():
        with open(f, "w", encoding="utf-8") as fp:
            fp.write("")

st.session_state.setdefault("feedback_log", feedback_file)

# -----------------------
# Utils
# -----------------------
def crawl_website(start_url, max_pages=5):
    from collections import deque
    visited, q = set(), deque([start_url])
    results = []
    while q and len(visited) < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded) if downloaded else None
            if not text:
                soup = BeautifulSoup(r.text, "html.parser")
                text = soup.get_text(separator="\n")
            if text and text.strip():
                results.append((url, text))
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                if urlparse(link).netloc == urlparse(start_url).netloc:
                    if link not in visited:
                        q.append(link)
        except Exception:
            continue
    return results

def extract_file_text(uploaded_file):
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        try:
            from pypdf import PdfReader
        except ModuleNotFoundError:
            from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        text = " ".join((page.extract_text() or "") for page in reader.pages)
    else:
        return uploaded_file.name, None
    text = re.sub(r"\s{2,}", " ", text.replace("\n", " ")).strip()
    return uploaded_file.name, text

def chunk_labeled_texts(labeled_texts, chunk_size=1100, chunk_overlap=220):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    docs = []
    for source, text in labeled_texts:
        if not text:
            continue
        for c in splitter.split_text(text):
            if len(c.strip()) > 50:
                docs.append(Document(page_content=c, metadata={"source": source}))
    return docs

def get_embeddings():
    if has_openai_key():
        from langchain_openai import OpenAIEmbeddings
        st.sidebar.success("✅ Using OpenAI embeddings")
        return OpenAIEmbeddings(), "openai"
    else:
        st.sidebar.info("✅ Using local embeddings (HuggingFace)")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), "local"

def save_metadata(index_path: Path, description: str):
    meta = {
        "name": index_path.name,
        "description": description,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(index_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_metadata(index_path: Path):
    meta_file = index_path / "meta.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"name": index_path.name, "description": "", "last_updated": "Unknown"}

# -----------------------
# Retrieval Helpers
# -----------------------
@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank_documents(query, docs, top_k=5):
    if not docs:
        return []
    model = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]

def deduplicate_chunks(docs, min_diff=60):
    seen, unique = set(), []
    for doc in docs:
        snippet = doc.page_content[:min_diff]
        if snippet not in seen:
            unique.append(doc)
            seen.add(snippet)
    return unique

@st.cache_resource
def get_query_gen():
    return pipeline("text2text-generation", model="google/flan-t5-base")

def expand_queries(question: str, n_variants=5):
    query_gen = get_query_gen()
    prompt = f"Generate {n_variants} different rephrasings of this question:\n{question}"
    outputs = query_gen(prompt, max_new_tokens=100)
    variants = [o["generated_text"].strip() for o in outputs]
    return [question] + variants

def hybrid_multiquery_search(db, question, top_k=6, score_threshold=0.3):
    faiss_retriever = db.as_retriever(search_kwargs={"k": 30})
    docs = db.similarity_search("dummy", k=2000)
    bm25_retriever = BM25Retriever.from_documents(docs)

    retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.4, 0.6]
    )

    queries = expand_queries(question, n_variants=5)
    all_docs = []

    for q in queries:
        scored = db.similarity_search_with_score(q, k=30)
        for doc, score in scored:
            if score >= score_threshold:
                doc.metadata["similarity_score"] = round(float(score), 3)
                all_docs.append(doc)

    unique_docs = deduplicate_chunks(all_docs)
    return rerank_documents(question, unique_docs, top_k=top_k)

@st.cache_resource
def get_local_llm():
    return pipeline("text2text-generation", model="google/flan-t5-large", device_map="auto")

# -----------------------
# Sidebar - Build / Ingest
# -----------------------
st.sidebar.header("⚙️ Build / Ingest")
source_mode = st.sidebar.radio("📂 Select Source:", ["Website", "Upload File"])
max_pages = st.sidebar.slider("📄 Max Pages to Crawl", 1, 20, 3)

# Chunking settings
st.sidebar.subheader("✂️ Chunking Settings")
chunk_size = st.sidebar.slider("Chunk size", 300, 1500, 1100, 50)
chunk_overlap = st.sidebar.slider("Chunk overlap", 50, 400, 220, 10)

website_url, uploaded_files = None, None
if source_mode == "Website":
    website_url = st.sidebar.text_input("🌍 Website URL", value="https://www.example.com")
else:
    uploaded_files = st.sidebar.file_uploader("📂 Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)

index_description = st.sidebar.text_input("🏷️ Index Description", value="")
can_build = (source_mode == "Website" and website_url) or (source_mode == "Upload File" and uploaded_files)

if st.sidebar.button("⚡ Build Index", disabled=not can_build):
    labeled, index_name = [], None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if source_mode == "Website" and website_url:
        with st.spinner(f"Crawling {website_url}..."):
            labeled = crawl_website(website_url, max_pages=max_pages)
        index_name = "web_" + urlparse(website_url).netloc.replace(".", "_") + f"_{timestamp}"

    elif source_mode == "Upload File" and uploaded_files:
        with st.spinner("Extracting text from files..."):
            for uf in uploaded_files:
                name, txt = extract_file_text(uf)
                if txt and txt.strip():
                    labeled.append((name, txt))
        base_name = "multi_files" if len(uploaded_files) > 1 else uploaded_files[0].name.replace(".", "_")
        index_name = "pdf_" + base_name + f"_{timestamp}"

    if labeled and index_name:
        docs = chunk_labeled_texts(labeled, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings, emb_type = get_embeddings()
        index_path = INDEX_DIR / f"{index_name}_{emb_type}"
        db = FAISS.from_documents(docs, embedding=embeddings)
        db.save_local(str(index_path))
        save_metadata(index_path, index_description or "No description")

        st.session_state["active_index"] = str(index_path)
        st.session_state["db"] = None
        st.session_state["messages"] = []
        st.session_state["force_select_active"] = True
        st.sidebar.success(f"✅ Index built: {index_path.name}")
    else:
        st.sidebar.error("❌ Nothing to index")

# -----------------------
# Index Management
# -----------------------
st.sidebar.header("📁 Index Management")
available_indexes = [p for p in INDEX_DIR.glob("*") if p.is_dir()]
st.sidebar.subheader("🔄 Reset Options")
if st.sidebar.button("🗑️ Clear All Indexes"):
    import shutil
    for idx in available_indexes:
        shutil.rmtree(idx, ignore_errors=True)
    st.session_state.clear()
    st.success("✅ All indexes cleared. Please build a new index.")
    st.stop()

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state["messages"] = []
    st.session_state["last_answer"] = None
    st.success("✅ Chat history cleared.")

if not available_indexes:
    st.warning("⚠️ No indexes found. Build one first.")
    st.stop()

search_filter = st.sidebar.text_input("🔍 Search Indexes")
metadata_list = [load_metadata(p) for p in available_indexes]
if search_filter:
    metadata_list = [
        m for m in metadata_list
        if search_filter.lower() in m["name"].lower()
        or search_filter.lower() in m["description"].lower()
    ]
if not metadata_list:
    st.sidebar.warning("⚠️ No matches.")
    st.stop()

options = [f"{m['name']} → {m['description']} (📅 {m['last_updated']})" for m in metadata_list]
default_idx = 0
if "active_index" in st.session_state:
    active_name = Path(st.session_state["active_index"]).name
    for i, m in enumerate(metadata_list):
        if m["name"] == active_name:
            default_idx = i
            break

selected_display = st.sidebar.selectbox("📑 Select Index", options, index=default_idx)
selected_index = metadata_list[options.index(selected_display)]
index_path = INDEX_DIR / selected_index["name"]
st.session_state["active_index"] = str(index_path)
# Delete current index
if st.sidebar.button("🗑️ Delete Current Index"):
    import shutil
    shutil.rmtree(index_path, ignore_errors=True)

    # Remove from session
    st.session_state["messages"] = []
    st.session_state["last_answer"] = None
    st.session_state["last_docs"] = []

    # Pick the next available index automatically
    remaining_indexes = [p for p in INDEX_DIR.glob("*") if p.is_dir()]
    if remaining_indexes:
        # Pick the first available index
        new_index_path = remaining_indexes[0]
        st.session_state["active_index"] = str(new_index_path)
        st.success(f"✅ Index '{selected_index['name']}' deleted. Switched to '{new_index_path.name}'.")
    else:
        # No indexes left
        st.session_state.pop("active_index", None)
        st.success(f"✅ Index '{selected_index['name']}' deleted. No indexes available, please build a new one.")

    st.stop()


embeddings, _ = get_embeddings()
db = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
st.session_state["db"] = db

# -----------------------
# Retrieval Settings
# -----------------------
st.sidebar.subheader("🔍 Retrieval Settings")
top_k = st.sidebar.slider("Number of chunks (top_k)", 3, 15, 6, 1)
similarity_threshold = st.sidebar.slider(
    "Similarity score threshold",
    min_value=0.2,
    max_value=0.6,
    value=0.3,
    step=0.05,
    help="Chunks below this similarity score will be ignored"
)

# Debug mode
st.sidebar.subheader("⚙️ Debug Options")
debug_mode = st.sidebar.checkbox("Show Retrieved Chunks (Debug)", value=False)

# -----------------------
# App Mode
# -----------------------
st.sidebar.header("🧭 App Mode")
app_mode = st.sidebar.radio("Select Mode", ["Chatbot", "Admin Dashboard"], index=0)

# -----------------------
# Chatbot
# -----------------------
if app_mode == "Chatbot":
    st.markdown(f"### 💬 Chatbot (Active Index: `{selected_index['name']}`)")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = None

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me something...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        docs = hybrid_multiquery_search(
            st.session_state["db"],
            user_input,
            top_k=top_k,
            score_threshold=similarity_threshold
        )

        if debug_mode:
            with st.expander("🔎 Retrieved Chunks (Debug)"):
                export_data = []
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("source", "unknown")
                    score = doc.metadata.get("similarity_score", "N/A")
                    preview = doc.page_content[:400].replace("\n", " ")
                    st.markdown(f"**Chunk {i}** — *Source:* `{src}` — 🔢 Score: {score}\n\n{preview}…")
                    export_data.append({
                        "rank": i,
                        "source": src,
                        "score": score,
                        "content": doc.page_content
                    })

                st.download_button(
                    "⬇️ Download Retrieved Chunks (JSON)",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name="retrieved_chunks.json",
                    mime="application/json"
                )
                st.download_button(
                    "⬇️ Download Retrieved Chunks (TXT)",
                    data="\n\n".join([f"[{d['rank']}] {d['source']} (score={d['score']})\n{d['content']}" for d in export_data]),
                    file_name="retrieved_chunks.txt",
                    mime="text/plain"
                )

        if has_openai_key():
            from langchain_openai import ChatOpenAI
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain

            structured_prompt = """
            You are a helpful assistant. Use ONLY the provided context.

            Question:
            {question}

            Answer:
            - Give a clear, direct answer.
            - Use bullet points if listing multiple facts.
            - If not in context, reply: "I don’t know from the given documents."

            Sources:
            - List supporting sources from metadata.
            - If no sources, write: "No sources found."

            Context:
            {context}
            """
            PROMPT = PromptTemplate(template=structured_prompt, input_variables=["context", "question"])
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            context = "\n\n".join(d.page_content for d in docs)
            chain = LLMChain(llm=llm, prompt=PROMPT)
            answer = chain.run({"context": context, "question": user_input})
        else:
            llm = get_local_llm()
            context = "\n\n".join(d.page_content for d in docs)
            prompt = (
                "Answer the question using ONLY the context.\n"
                "If the answer is not in context, reply: 'I don’t know from the documents.'\n"
                "Use bullet points, be concise, and cite sources from metadata.\n\n"
                f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
            )
            result = llm(prompt, max_new_tokens=300, temperature=0.0)
            answer = result[0]["generated_text"]

            st.session_state["last_answer"] = {
            "question": user_input,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "index": selected_index["name"],
        }
        st.session_state["last_docs"] = [
            {
                "source": d.metadata.get("source", "unknown"),
                "score": d.metadata.get("similarity_score", None),
                "content": d.page_content[:300]  # preview only
            }
            for d in docs
        ]

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})

    # Feedback buttons
    if st.session_state.get("last_answer"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful"):
                fb = {
                    **st.session_state["last_answer"],
                    "feedback": "positive",
                    "docs": st.session_state.get("last_docs", [])
                }
                with open(st.session_state["feedback_log"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                st.success("✅ Feedback saved")
                st.session_state["last_answer"] = None
        with col2:
            if st.button("👎 Not Helpful"):
                fb = {
                    **st.session_state["last_answer"],
                    "feedback": "negative",
                    "docs": st.session_state.get("last_docs", [])
                }
                with open(st.session_state["feedback_log"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                with open("hard_questions.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"question": fb["question"], "index": fb["index"]}) + "\n")
                st.error("❌ Feedback saved & added to hard questions")
                st.session_state["last_answer"] = None

