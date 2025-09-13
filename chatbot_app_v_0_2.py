import os
import re
import json
import requests
import trafilatura
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urljoin

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="💬 RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot")

INDEX_DIR = Path(r"D:\SpikeUp.AI\Extracted\indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# API Key Helpers
# -----------------------
def load_openai_key_from_secrets():
    """Try to load OpenAI API key from Streamlit secrets or env"""
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    if key and isinstance(key, str) and key.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = key
        return True
    return bool(os.environ.get("OPENAI_API_KEY", "").startswith("sk-"))

def has_openai_key():
    return bool(os.environ.get("OPENAI_API_KEY", "").startswith("sk-"))

# Call once at startup
load_openai_key_from_secrets()

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
            # enqueue internal links
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

def chunk_labeled_texts(labeled_texts, chunk_size=800, chunk_overlap=150):
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

@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

# -----------------------
# Sidebar: Build Index
# -----------------------
st.sidebar.header("⚙️ Build / Ingest")
source_mode = st.sidebar.radio("📂 Select Source:", ["Website", "Upload File"])
max_pages = st.sidebar.slider("📄 Max Pages to Crawl", 1, 20, 3)

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
        docs = chunk_labeled_texts(labeled)
        embeddings, emb_type = get_embeddings()
        index_path = INDEX_DIR / f"{index_name}_{emb_type}"
        db = FAISS.from_documents(docs, embedding=embeddings)
        db.save_local(str(index_path))
        save_metadata(index_path, index_description or "No description")

        st.session_state["active_index"] = str(index_path)
        st.session_state["db"] = None
        st.session_state["messages"] = []
        st.session_state["force_select_active"] = True  # NEW: auto-select this index

        st.sidebar.success(f"✅ Index built: {index_path.name}")
    else:
        st.sidebar.error("❌ Nothing to index")

# -----------------------
# Index Management + Reset
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

# --- NEW logic: auto-select active index ---
default_idx = 0
if "active_index" in st.session_state:
    active_name = Path(st.session_state["active_index"]).name
    for i, m in enumerate(metadata_list):
        if m["name"] == active_name:
            default_idx = i
            break

if st.session_state.get("force_select_active"):
    selected_display = st.sidebar.selectbox("📑 Select Index", options, index=default_idx, key="index_selector")
    st.session_state["force_select_active"] = False
else:
    selected_display = st.sidebar.selectbox("📑 Select Index", options, index=default_idx, key="index_selector")

selected_index = metadata_list[options.index(selected_display)]

# -----------------------
# App Mode
# -----------------------
st.sidebar.header("🧭 App Mode")
app_mode = st.sidebar.radio("Select Mode", ["Chatbot", "Admin Dashboard"], index=0)

# -----------------------
# Chatbot
# -----------------------
if app_mode == "Chatbot":
    from langchain.prompts import PromptTemplate

    st.markdown(f"### 💬 Chatbot (Active Index: `{active_index_name}`)")

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

    chat_mode = st.sidebar.radio("🤖 Chat Mode", ["Retrieval Only", "Full Chatbot (RAG)"], index=1)
    use_openai = st.sidebar.checkbox("Use OpenAI for generation (requires API key)", value=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = None
    if "feedback_log" not in st.session_state:
        st.session_state["feedback_log"] = "feedback_log.jsonl"

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me something...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        db = st.session_state["db"]

        if chat_mode == "Retrieval Only" or (chat_mode == "Full Chatbot (RAG)" and (not has_openai_key() or not use_openai)):
            candidate_docs = db.similarity_search(user_input, k=50)
            candidate_docs = deduplicate_chunks(candidate_docs)
            docs = rerank_documents(user_input, candidate_docs, top_k=5)
            parts = ["### 🔎 Retrieved Chunks (Re-ranked, Deduped)"]
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "unknown")
                preview = doc.page_content[:400].replace("\n", " ")
                parts.append(f"**Chunk {i}** — *Source:* `{src}`\n\n{preview}…")
            answer = "\n\n".join(parts)

        else:
            from langchain_openai import ChatOpenAI
            from langchain.chains import LLMChain

            candidate_docs = db.similarity_search(user_input, k=50)
            candidate_docs = deduplicate_chunks(candidate_docs)
            docs = rerank_documents(user_input, candidate_docs, top_k=6)

            with st.expander("🔎 Retrieved Chunks (Debug)"):
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("source", "unknown")
                    st.markdown(f"**Chunk {i}** — `{src}`\n\n{doc.page_content[:400]}…")

            llm = ChatOpenAI(model="gpt-3.5-turbo")
            context = "\n\n".join(d.page_content for d in docs)
            chain = LLMChain(llm=llm, prompt=PROMPT)
            answer = chain.run({"context": context, "question": user_input})

        st.session_state["last_answer"] = {
            "question": user_input,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "index": active_index_name,
        }

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})

    # feedback buttons
    if st.session_state.get("last_answer"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful"):
                fb = {**st.session_state["last_answer"], "feedback": "positive"}
                with open(st.session_state["feedback_log"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                st.success("✅ Feedback saved")
                st.session_state["last_answer"] = None
        with col2:
            if st.button("👎 Not Helpful"):
                fb = {**st.session_state["last_answer"], "feedback": "negative"}
                with open(st.session_state["feedback_log"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                with open("hard_questions.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"question": fb["question"], "index": fb["index"]}) + "\n")
                st.error("❌ Feedback saved & added to hard questions")
                st.session_state["last_answer"] = None

# -----------------------
# Admin Dashboard
# -----------------------
else:
    st.header("📊 Admin Dashboard")

    feedback_file = st.session_state.get("feedback_log", "feedback_log.jsonl")
    hard_file = "hard_questions.jsonl"

    data = []
    if Path(feedback_file).exists():
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue

    if not data:
        st.warning("⚠️ No feedback collected yet.")
    else:
        df = pd.DataFrame(data)

        # filter by index
        if "index" in df.columns:
            unique_indexes = df["index"].unique().tolist()
            selected_idx = st.selectbox("📂 Filter by Index", ["All"] + unique_indexes)
            if selected_idx != "All":
                df = df[df["index"] == selected_idx]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("👍 Positive", (df["feedback"] == "positive").sum())
        with col2:
            st.metric("👎 Negative", (df["feedback"] == "negative").sum())

        st.subheader("📋 Feedback Records")
        st.dataframe(
            df[["timestamp", "index", "question", "feedback", "answer"]]
            .sort_values("timestamp", ascending=False),
            use_container_width=True,
            height=400,
        )

        st.subheader("🚨 Top Problematic Questions")
        if (df["feedback"] == "negative").any():
            neg_df = df[df["feedback"] == "negative"]
            top_neg = (
                neg_df.groupby(["index", "question"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(10)
            )
            st.table(top_neg)
        else:
            st.info("✅ No negative feedback yet!")

        st.download_button(
            "⬇️ Download Feedback (JSONL)",
            data="\n".join([json.dumps(r) for r in data]),
            file_name="feedback_log.jsonl",
            mime="application/json",
        )

    st.subheader("🧪 Hard Questions")
    if Path(hard_file).exists():
        hard_qs = []
        with open(hard_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    hard_qs.append(json.loads(line.strip()))
                except:
                    continue
        if hard_qs:
            hard_df = pd.DataFrame(hard_qs)
            st.dataframe(hard_df.sort_values("question"), use_container_width=True, height=300)
            st.download_button(
                "⬇️ Download Hard Questions",
                data="\n".join([json.dumps(r) for r in hard_qs]),
                file_name="hard_questions.jsonl",
                mime="application/json",
            )
        else:
            st.info("✅ Hard questions file exists but is empty.")
    else:
        st.info("⚠️ No hard questions collected yet.")
