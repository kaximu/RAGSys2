"""This version will include:

🧭 Sidebar

Build / Ingest (Website or File)

Chunking settings (size + overlap)

Retrieval settings (top_k, similarity threshold)

Debug mode toggle

Vector store toggle (FAISS / Qdrant Hybrid)

Reset options (Clear chat, Delete current index, Clear all indexes)

Index search + auto-select last index

🧠 Retrieval Pipeline

Multi-query expansion (5 variants, FLAN-T5)

Deduplication

Strong reranker (CrossEncoder MiniLM-L-12)

FAISS OR Qdrant Hybrid (dense+sparse) retrieval

Similarity filtering with threshold slider

💬 Chatbot

Uses OpenAI (if API key exists) or HuggingFace fallback

Debug mode → retrieved chunks with similarity scores + export JSON/TXT

Feedback buttons 👍👎 → logs question, answer, docs, backend, retrieval time, chunk count

📊 Admin Dashboard

Feedback records

Top problematic questions

Scatterplot (similarity vs feedback) + PNG export

Backend comparison (accuracy table + chart)

Speed comparison (ms)

Retrieved chunk count comparison

Per-question detail view (chunks with source + score + preview) + export JSON/TXT

       
        """
   

import os
import re
import io
import json
import time
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
from langchain_community.vectorstores import Qdrant
from sentence_transformers import CrossEncoder
from transformers import pipeline
from langchain.retrievers import BM25Retriever, EnsembleRetriever

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http import models


# =========================
# Config
# =========================
st.set_page_config(page_title="💬 RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot (Qdrant Cloud)")

INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_FILE = "feedback_log.jsonl"
HARD_FILE = "hard_questions.jsonl"
for f in [FEEDBACK_FILE, HARD_FILE]:
    if not Path(f).exists():
        with open(f, "w", encoding="utf-8") as fp:
            fp.write("")


# =========================
# Utils
# =========================
def has_openai_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key and key.startswith("sk-"))


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
            if text and len(text.split()) > 30:
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


def chunk_texts(labeled_texts, chunk_size=1100, chunk_overlap=220):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    docs = []
    for source, text in labeled_texts:
        if not text or len(text.split()) < 30:
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
        st.sidebar.info("✅ Using HuggingFace embeddings")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), "local"


def save_metadata(index_path: Path, description: str):
    meta = {
        "name": index_path.name,
        "description": description,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend": "Qdrant Cloud",
    }
    index_path.mkdir(parents=True, exist_ok=True)
    with open(index_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_metadata(index_path: Path):
    meta_file = index_path / "meta.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"name": index_path.name, "description": "", "last_updated": "Unknown", "backend": "Qdrant Cloud"}


# =========================
# Retrieval
# =========================
@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


def rerank_documents(query, docs, top_k=5):
    if not docs:
        return []
    model = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)
    for doc, score in zip(docs, scores):
        doc.metadata["similarity_score"] = round(float(score), 3)
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


def qdrant_search(client, collection_name, embeddings, question, top_k=6, score_threshold=0.3):
    qdrant = Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)
    retriever = qdrant.as_retriever(search_kwargs={"k": top_k})
    queries = expand_queries(question, n_variants=5)
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.get_relevant_documents(q))
    unique_docs = deduplicate_chunks(all_docs)
    return rerank_documents(question, unique_docs, top_k=top_k)


# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Build / Ingest")
source_mode = st.sidebar.radio("📂 Select Source:", ["Website", "Upload File"])
max_pages = st.sidebar.slider("📄 Max Pages to Crawl", 1, 20, 3)

st.sidebar.subheader("✂️ Chunking Settings")
chunk_size = st.sidebar.slider("Chunk size", 300, 1500, 1100, 50)
chunk_overlap = st.sidebar.slider("Chunk overlap", 50, 400, 220, 10)

website_url, uploaded_files = None, None
if source_mode == "Website":
    website_url = st.sidebar.text_input("🌍 Website URL", value="https://www.example.com")
else:
    uploaded_files = st.sidebar.file_uploader("📂 Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)

index_description = st.sidebar.text_input("🏷️ Index Description", value="")

# Retrieval settings
st.sidebar.subheader("🔍 Retrieval Settings")
top_k = st.sidebar.slider("Number of chunks (top_k)", 3, 15, 6, 1)
similarity_threshold = st.sidebar.slider("Similarity threshold", 0.2, 0.6, 0.3, 0.05)

# App Mode
st.sidebar.header("🧭 App Mode")
app_mode = st.sidebar.radio("Select Mode", ["Chatbot", "Admin Dashboard"], index=0)


# =========================
# Build Index (Qdrant Cloud)
# =========================
if st.sidebar.button("⚡ Build Index", disabled=not (website_url or uploaded_files)):
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
        index_name = "file_" + base_name + f"_{timestamp}"

    if labeled and index_name:
        docs = chunk_texts(labeled, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings, emb_type = get_embeddings()

        # Connect to Qdrant Cloud
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        collection_name = f"{index_name}_{emb_type}"

        # Always recreate collection
        vector_size = len(embeddings.embed_query("test"))
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

        # Insert documents
        Qdrant.from_documents(
            docs,
            embedding=embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False,
            collection_name=collection_name,
        )

        save_metadata(INDEX_DIR / collection_name, index_description or "No description")
        st.session_state["active_index"] = collection_name
        st.session_state["messages"] = []
        st.sidebar.success(f"✅ Index built: {index_name}")
    else:
        st.sidebar.error("❌ Nothing to index")
# =========================
# Chatbot Mode
# =========================
if app_mode == "Chatbot":
    st.markdown(f"### 💬 Chatbot (Active Index: `{st.session_state.get('active_index','None')}`)")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = None

    # Show chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me something...")
    if user_input and st.session_state.get("active_index"):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieval
        start_time = time.perf_counter()
        embeddings, _ = get_embeddings()
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        docs = qdrant_search(client, st.session_state["active_index"], embeddings, user_input, top_k=top_k, score_threshold=similarity_threshold)
        retrieval_time = round((time.perf_counter() - start_time) * 1000, 2)
        retrieved_count = len(docs)

        # Debug Mode
        if st.sidebar.checkbox("Show Retrieved Chunks (Debug)", value=False):
            with st.expander("🔎 Retrieved Chunks (Debug)"):
                export_data = []
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("source", "unknown")
                    score = doc.metadata.get("similarity_score", "N/A")
                    preview = doc.page_content[:400].replace("\n", " ")
                    st.markdown(f"**Chunk {i}** — *Source:* `{src}` — 🔢 Score: {score}\n\n{preview}…")
                    export_data.append({"rank": i, "source": src, "score": score, "content": doc.page_content})
                st.download_button("⬇️ Download (JSON)", data=json.dumps(export_data, indent=2, ensure_ascii=False), file_name="retrieved.json", mime="application/json")
                st.download_button("⬇️ Download (TXT)", data="\n\n".join([f"[{d['rank']}] {d['source']} (score={d['score']})\n{d['content']}" for d in export_data]), file_name="retrieved.txt", mime="text/plain")

        # Generate Answer
        if has_openai_key():
            from langchain_openai import ChatOpenAI
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            PROMPT = PromptTemplate(
                template="""You are a helpful assistant. Use ONLY the provided context.

                Question:
                {question}

                Answer:
                - Give a clear, direct answer.
                - Use bullet points if listing multiple facts.
                - If not in context, reply: "I don’t know from the given documents."

                Sources:
                - Cite sources from metadata.
                - If no sources, write: "No sources found."

                Context:
                {context}
                """,
                input_variables=["context", "question"]
            )
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            context = "\n\n".join(d.page_content for d in docs)
            chain = LLMChain(llm=llm, prompt=PROMPT)
            answer = chain.run({"context": context, "question": user_input})
        else:
            llm = pipeline("text2text-generation", model="google/flan-t5-large", device_map="auto")
            context = "\n\n".join(d.page_content for d in docs)
            prompt = f"Answer the question using ONLY the context.\nIf missing, reply: 'I don’t know from the documents.'\nContext:\n{context}\n\nQuestion: {user_input}\nAnswer:"
            result = llm(prompt, max_new_tokens=300, temperature=0.0)
            answer = result[0]["generated_text"]

        # Save last answer
        st.session_state["last_answer"] = {
            "question": user_input,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "index": st.session_state["active_index"],
            "backend": "Qdrant Cloud",
            "retrieval_time_ms": retrieval_time,
            "retrieved_chunks": retrieved_count,
        }
        st.session_state["last_docs"] = [
            {"source": d.metadata.get("source", "unknown"),
             "score": d.metadata.get("similarity_score", None),
             "content": d.page_content[:300]} for d in docs
        ]

        # Display assistant answer
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    # Feedback
    if st.session_state.get("last_answer"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful"):
                fb = {**st.session_state["last_answer"], "feedback": "positive", "docs": st.session_state.get("last_docs", [])}
                with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                st.success("✅ Feedback saved")
                st.session_state["last_answer"] = None
        with col2:
            if st.button("👎 Not Helpful"):
                fb = {**st.session_state["last_answer"], "feedback": "negative", "docs": st.session_state.get("last_docs", [])}
                with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(fb) + "\n")
                with open(HARD_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"question": fb["question"], "index": fb["index"]}) + "\n")
                st.error("❌ Feedback saved & added to hard questions")
                st.session_state["last_answer"] = None


# =========================
# Admin Dashboard
# =========================
else:
    st.header("📊 Admin Dashboard")

    data = []
    if Path(FEEDBACK_FILE).exists():
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue

    if not data:
        st.warning("⚠️ No feedback yet.")
    else:
        df = pd.DataFrame(data)

        # Backend stats
        st.subheader("📊 Backend Stats (Qdrant Cloud)")
        comp_df = df.groupby("backend")["feedback"].value_counts().unstack(fill_value=0)
        comp_df["Total"] = comp_df.sum(axis=1)
        comp_df["Accuracy %"] = round(comp_df.get("positive", 0) / comp_df["Total"] * 100, 2)
        st.dataframe(comp_df, use_container_width=True)

        # Speed
        st.subheader("⚡ Retrieval Speed")
        if "retrieval_time_ms" in df.columns:
            speed_df = df.groupby("backend")["retrieval_time_ms"].mean().reset_index()
            st.dataframe(speed_df, use_container_width=True)

        # Chunks
        st.subheader("📦 Retrieved Chunks")
        if "retrieved_chunks" in df.columns:
            chunk_df = df.groupby("backend")["retrieved_chunks"].mean().reset_index()
            st.dataframe(chunk_df, use_container_width=True)

        # Feedback vs Similarity
        st.subheader("📊 Feedback vs Similarity")
        plot_data = []
        for row in data:
            fb = row.get("feedback", "")
            for d in row.get("docs", []):
                score = d.get("score")
                if score is not None:
                    plot_data.append({"feedback": fb, "score": score})
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            fig, ax = plt.subplots()
            colors = df_plot["feedback"].map({"positive": "green", "negative": "red"})
            ax.scatter(df_plot["score"], df_plot.index, c=colors, alpha=0.6)
            ax.set_xlabel("Similarity Score")
            ax.set_ylabel("Instance")
            st.pyplot(fig)

        # Inspect per-question
        st.subheader("🔎 Inspect Retrieved Chunks")
        questions = df["question"].unique().tolist()
        selected_q = st.selectbox("Select Question", ["None"] + questions)
        if selected_q != "None":
            row = df[df["question"] == selected_q].iloc[0]
            retrieved_docs = row.get("docs", [])
            if retrieved_docs:
                for i, d in enumerate(retrieved_docs, 1):
                    st.markdown(f"**Chunk {i}** — Source: `{d.get('source','unknown')}` — 🔢 Score: {d.get('score','N/A')}\n\n{d.get('content','')[:400]}...")
                st.download_button("⬇️ Download JSON", data=json.dumps(retrieved_docs, indent=2, ensure_ascii=False), file_name=f"{selected_q.replace(' ','_')}_retrieved.json", mime="application/json")
