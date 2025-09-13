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
   

# chatbot_app_Qdrant.py
# Full Streamlit RAG app with Qdrant Cloud + optional FAISS + Hybrid retrieval,
# multi-query expansion, reranking, similarity thresholding, debug view,
# index management (select / delete current / delete all), feedback logging,
# admin dashboard, and reset chat.

# chatbot_app_Qdrant.py
# Full Streamlit RAG app with Qdrant Cloud + optional FAISS + Hybrid retrieval,
# multi-query expansion, reranking, similarity thresholding, debug view,
# index management (select / delete current / delete all), feedback logging,
# admin dashboard, and reset chat.

import os
import re
import io
import json
import time
import shutil
import requests
import trafilatura
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urljoin
from typing import List, Tuple, Optional

# LangChain / embeddings / retrievers
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant as LCQdrant
from sentence_transformers import CrossEncoder
from transformers import pipeline

# BM25 + Ensemble retriever imports (compat across LC versions)
try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    from langchain.retrievers import BM25Retriever  # old path

try:
    from langchain.retrievers import EnsembleRetriever
except Exception:
    EnsembleRetriever = None  # will guard below

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Optional FAISS
FAISS_AVAILABLE = True
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    FAISS_AVAILABLE = False

# =========================
# Page / Constants
# =========================
st.set_page_config(page_title="💬 RAG Chatbot (Qdrant/FAISS)", layout="wide")
st.title("💬 RAG Chatbot (Qdrant Cloud / FAISS / Hybrid)")

INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_FILE = "feedback_log.jsonl"
HARD_FILE = "hard_questions.jsonl"
for _f in [FEEDBACK_FILE, HARD_FILE]:
    if not Path(_f).exists():
        Path(_f).write_text("", encoding="utf-8")

# =========================
# Helpers / Config
# =========================
def has_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key and key.startswith("sk-"))

@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        st.error("⚠️ Qdrant credentials not set. Please add QDRANT_URL and QDRANT_API_KEY.")
        st.stop()
    return QdrantClient(url=url, api_key=api_key)

def save_jsonl(path: Path, rows: List[dict]):
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: Path) -> List[dict]:
    data = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
    return data

# =========================
# Ingestion Utils
# =========================
def crawl_website(start_url: str, max_pages: int = 5) -> List[Tuple[str, str]]:
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
                results.append((url, clean_text(text)))
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

def clean_text(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text.replace("\n", " ")).strip()

def extract_file_text(uploaded_file) -> Tuple[str, Optional[str]]:
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
    return uploaded_file.name, clean_text(text)

def chunk_texts(labeled_texts: List[Tuple[str, str]], chunk_size=1100, chunk_overlap=220) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    docs: List[Document] = []
    for source, text in labeled_texts:
        if not text or len(text.split()) < 30:
            continue
        for c in splitter.split_text(text):
            c = c.strip()
            if len(c) > 50:
                docs.append(Document(page_content=c, metadata={"source": source}))
    return docs

def corpus_sidecar_path(collection_name: str) -> Path:
    # Where we store a lightweight JSONL of chunks for BM25 per index
    return INDEX_DIR / collection_name / "corpus.jsonl"

def save_corpus(collection_name: str, docs: List[Document]):
    p = corpus_sidecar_path(collection_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"source": d.metadata.get("source", "unknown"), "content": d.page_content} for d in docs]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_corpus(collection_name: str) -> List[Document]:
    p = corpus_sidecar_path(collection_name)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line.strip())
                out.append(Document(page_content=row.get("content",""), metadata={"source": row.get("source","unknown")}))
            except Exception:
                continue
    return out

# =========================
# Embeddings / Reranker
# =========================
def get_embeddings():
    if has_openai_key():
        from langchain_openai import OpenAIEmbeddings
        st.sidebar.success("✅ Using OpenAI embeddings")
        return OpenAIEmbeddings(), "openai"
    else:
        st.sidebar.info("✅ Using HuggingFace embeddings (all-MiniLM-L6-v2)")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), "local"

@st.cache_resource
def get_reranker():
    # Small, fast cross-encoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank_documents(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    if not docs:
        return []
    model = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)
    for doc, score in zip(docs, scores):
        doc.metadata["similarity_score"] = float(score)
    # sort by score desc
    ranked = [doc for doc, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]
    return ranked[:top_k]

def apply_similarity_threshold(docs: List[Document], threshold: float) -> List[Document]:
    out = []
    for d in docs:
        s = d.metadata.get("similarity_score", None)
        if s is None:
            out.append(d)  # if missing, keep
        else:
            if s >= threshold:
                out.append(d)
    return out

def deduplicate_chunks(docs: List[Document], min_diff=80) -> List[Document]:
    seen, unique = set(), []
    for doc in docs:
        snippet = doc.page_content[:min_diff]
        if snippet not in seen:
            unique.append(doc)
            seen.add(snippet)
    return unique

@st.cache_resource
def get_query_gen():
    # CPU-friendly model
    return pipeline("text2text-generation", model="google/flan-t5-base")

def expand_queries(question: str, n_variants=5) -> List[str]:
    gen = get_query_gen()
    prompt = f"Generate {n_variants} different rephrasings of this question:\n{question}"
    outputs = gen(prompt, max_new_tokens=100)
    variants = [o["generated_text"].strip() for o in outputs]
    return [question] + variants

# =========================
# Retrievers
# =========================
def qdrant_retriever(client: QdrantClient, collection_name: str, embeddings, k: int):
    vec = LCQdrant(client=client, collection_name=collection_name, embeddings=embeddings)
    return vec.as_retriever(search_kwargs={"k": k})

def bm25_retriever_from_corpus(collection_name: str, k: int):
    docs = load_corpus(collection_name)
    if not docs:
        return None
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25

def hybrid_retrieve(query: str, client: QdrantClient, collection_name: str, embeddings, k: int) -> List[Document]:
    # Qdrant dense + local BM25 sparse (if available)
    qr = qdrant_retriever(client, collection_name, embeddings, k)
    br = bm25_retriever_from_corpus(collection_name, k)
    if EnsembleRetriever and br is not None:
        ens = EnsembleRetriever(retrievers=[qr, br], weights=[0.5, 0.5])
        return ens.get_relevant_documents(query)
    # fallback to dense-only
    return qr.get_relevant_documents(query)

def faiss_paths(collection_name: str):
    base = INDEX_DIR / collection_name
    base.mkdir(parents=True, exist_ok=True)
    return base / "faiss.index", base / "faiss.pkl"

def build_faiss_index(collection_name: str, docs: List[Document], embeddings):
    if not FAISS_AVAILABLE:
        st.sidebar.error("FAISS not available. Install faiss-cpu to enable local FAISS indexes.")
        return None
    vs = FAISS.from_documents(docs, embeddings)
    idx_path, pkl_path = faiss_paths(collection_name)
    vs.save_local(folder_path=idx_path.parent.as_posix(), index_name="faiss")  # writes faiss.index / faiss.pkl
    return vs

def load_faiss_index(collection_name: str, embeddings):
    if not FAISS_AVAILABLE:
        return None
    idx_path, pkl_path = faiss_paths(collection_name)
    if idx_path.exists() and pkl_path.exists():
        return FAISS.load_local(folder_path=idx_path.parent.as_posix(), embeddings=embeddings, index_name="faiss", allow_dangerous_deserialization=True)
    return None

# =========================
# Sidebar: Build / Settings
# =========================
st.sidebar.header("⚙️ Build / Ingest")
source_mode = st.sidebar.radio("📂 Source", ["Website", "Upload File"], index=0)
max_pages = st.sidebar.slider("📄 Max Pages to Crawl", 1, 30, 5)

st.sidebar.subheader("✂️ Chunking")
chunk_size = st.sidebar.slider("Chunk size", 300, 2000, 1100, 50)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, 220, 10)

st.sidebar.subheader("🔎 Retrieval Settings")
top_k = st.sidebar.slider("Top K", 3, 20, 6, 1)
similarity_threshold = st.sidebar.slider("Similarity threshold (reranker score)", 0.0, 1.0, 0.30, 0.01)

st.sidebar.subheader("🧠 Vector Store")
store_options = ["Qdrant (Cloud)", "FAISS (local)", "Hybrid (Qdrant + BM25)"]
if not FAISS_AVAILABLE:
    store_options = ["Qdrant (Cloud)", "Hybrid (Qdrant + BM25)"]
vector_store_choice = st.sidebar.radio("Choose backend", store_options, index=0)

debug_mode = st.sidebar.toggle("🐞 Debug mode (show retrieved chunks)", value=False)

# Reset Chat button
st.sidebar.subheader("🧹 Reset")
if st.sidebar.button("🧹 Reset Chat"):
    st.session_state["messages"] = []
    st.sidebar.success("✅ Chat cleared")
    st.experimental_rerun()

# =========================
# Index Management (Qdrant & FAISS)
# =========================
st.sidebar.header("📁 Index Management")

client = None
qdrant_collections = []
if "Qdrant" in vector_store_choice or "Hybrid" in vector_store_choice:
    client = get_qdrant_client()
    try:
        qdrant_collections = [c.name for c in client.get_collections().collections]
    except Exception as e:
        st.sidebar.error(f"Qdrant error: {e}")

# Index search/filter
filter_term = st.sidebar.text_input("🔎 Filter indexes", value="")

filtered_names = qdrant_collections
if filter_term:
    filtered_names = [n for n in qdrant_collections if filter_term.lower() in n.lower()]

# Auto-select last used or latest
if filtered_names:
    default_idx = len(filtered_names) - 1
    if "active_index" in st.session_state and st.session_state["active_index"] in filtered_names:
        default_idx = filtered_names.index(st.session_state["active_index"])
    active_index = st.sidebar.selectbox("📑 Select Qdrant Index", filtered_names, index=default_idx)
    st.session_state["active_index"] = active_index
    st.sidebar.success(f"✅ Active: {active_index}")
else:
    if "Qdrant" in vector_store_choice or "Hybrid" in vector_store_choice:
        st.sidebar.info("No Qdrant indexes yet. Build one below.")

# Delete buttons (Qdrant)
if qdrant_collections:
    st.sidebar.subheader("🗑️ Manage Qdrant")
    if st.sidebar.button("❌ Delete Current Index", disabled="active_index" not in st.session_state):
        try:
            client.delete_collection(collection_name=st.session_state["active_index"])
            st.sidebar.warning(f"Deleted: {st.session_state['active_index']}")
            st.session_state.pop("active_index", None)
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Delete failed: {e}")

    if st.sidebar.button("⚠️ Delete ALL Qdrant Indexes"):
        try:
            for cname in qdrant_collections:
                client.delete_collection(collection_name=cname)
            st.sidebar.error("🚨 All Qdrant indexes deleted")
            st.session_state.clear()
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Delete failed: {e}")

# Delete buttons (FAISS local)
if FAISS_AVAILABLE:
    st.sidebar.subheader("🗑️ Manage FAISS (local)")
    local_idx_dirs = [p for p in INDEX_DIR.glob("*") if p.is_dir()]
    if local_idx_dirs and st.sidebar.button("⚠️ Delete ALL local FAISS indexes"):
        for d in local_idx_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        st.sidebar.warning("Deleted all local FAISS folders under ./indexes")

# =========================
# Build New Index
# =========================
website_url, uploaded_files = None, None
if source_mode == "Website":
    website_url = st.sidebar.text_input("🌍 Website URL", value="https://example.com")
else:
    uploaded_files = st.sidebar.file_uploader("📂 Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)

index_description = st.sidebar.text_input("🏷️ Index Description", value="")

if st.sidebar.button("⚡ Build Index", disabled=(source_mode == "Website" and not website_url) and not uploaded_files):
    labeled, index_name = [], None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if source_mode == "Website" and website_url:
        with st.spinner(f"Crawling {website_url}..."):
            labeled = crawl_website(website_url, max_pages=max_pages)
        index_name = "web_" + urlparse(website_url).netloc.replace(".", "_") + f"_{timestamp}"

    elif source_mode == "Upload File" and uploaded_files:
        with st.spinner("Extracting text..."):
            for uf in uploaded_files:
                name, txt = extract_file_text(uf)
                if txt:
                    labeled.append((name, txt))
        base_name = "multi_files" if len(uploaded_files) > 1 else uploaded_files[0].name.replace(".", "_")
        index_name = "file_" + base_name + f"_{timestamp}"

    if not labeled:
        st.sidebar.error("❌ Nothing to index")
    else:
        docs = chunk_texts(labeled, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings, emb_type = get_embeddings()
        collection_name = f"{index_name}_{emb_type}"
        st.session_state["messages"] = []

        # Build to selected backend(s)
        built_any = False

        if "Qdrant" in vector_store_choice or "Hybrid" in vector_store_choice:
            try:
                vec_size = len(embeddings.embed_query("test"))
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vec_size, distance=models.Distance.COSINE),
                )
                LCQdrant.from_documents(
                    docs,
                    embedding=embeddings,
                    url=os.getenv("QDRANT_URL"),
                    api_key=os.getenv("QDRANT_API_KEY"),
                    prefer_grpc=False,
                    collection_name=collection_name,
                )
                # Save corpus for BM25 (for Hybrid)
                save_corpus(collection_name, docs)
                st.session_state["active_index"] = collection_name
                st.sidebar.success(f"✅ Qdrant index built: {collection_name}")
                built_any = True
            except Exception as e:
                st.sidebar.error(f"Qdrant build failed: {e}")

        if "FAISS" in vector_store_choice and FAISS_AVAILABLE:
            try:
                build_faiss_index(collection_name, docs, embeddings)
                st.sidebar.success(f"✅ FAISS local index built: {collection_name}")
                built_any = True
            except Exception as e:
                st.sidebar.error(f"FAISS build failed: {e}")

        if built_any:
            st.sidebar.success("🎉 Build complete")
        else:
            st.sidebar.error("❌ Build failed for all selected backends")

# =========================
# App Mode
# =========================
st.sidebar.header("🧭 App Mode")
app_mode = st.sidebar.radio("Mode", ["Chatbot", "Admin Dashboard"], index=0)

# =========================
# Retrieval pipeline
# =========================
def retrieve_docs(question: str, top_k: int, threshold: float) -> Tuple[List[Document], float]:
    t0 = time.perf_counter()
    embeddings, _ = get_embeddings()
    queries = expand_queries(question, n_variants=5)

    # Collect pre-rerank docs
    prelim_docs: List[Document] = []

    if vector_store_choice.startswith("Qdrant"):
        if "active_index" not in st.session_state:
            return [], 0.0
        for q in queries:
            retr = qdrant_retriever(get_qdrant_client(), st.session_state["active_index"], embeddings, k=top_k)
            prelim_docs.extend(retr.get_relevant_documents(q))

    elif vector_store_choice.startswith("Hybrid"):
        if "active_index" not in st.session_state:
            return [], 0.0
        for q in queries:
            prelim_docs.extend(hybrid_retrieve(q, get_qdrant_client(), st.session_state["active_index"], embeddings, k=top_k))

    elif vector_store_choice.startswith("FAISS"):
        # load local FAISS
        if "active_index" not in st.session_state:
            return [], 0.0
        faiss_vs = load_faiss_index(st.session_state["active_index"], embeddings)
        if faiss_vs is None:
            st.warning("FAISS index not found locally. Build FAISS or switch backend.")
            return [], 0.0
        retr = faiss_vs.as_retriever(search_kwargs={"k": top_k})
        for q in queries:
            prelim_docs.extend(retr.get_relevant_documents(q))

    # dedup, rerank, threshold, top_k
    prelim_docs = deduplicate_chunks(prelim_docs)
    ranked = rerank_documents(question, prelim_docs, top_k=max(top_k, 10))  # rank with a bit more headroom
    filtered = apply_similarity_threshold(ranked, threshold=threshold)
    final_docs = filtered[:top_k]
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return final_docs, elapsed_ms

# =========================
# Chatbot
# =========================
if app_mode == "Chatbot":
    st.markdown(f"### 💬 Chatbot (Backend: `{vector_store_choice}` — Active Index: `{st.session_state.get('active_index','None')}`)")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = None

    # show history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me something…")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        docs, retrieval_time = retrieve_docs(user_input, top_k=top_k, threshold=similarity_threshold)
        retrieved_count = len(docs)

        # Debug view
        if debug_mode:
            with st.expander("🔎 Retrieved Chunks (Debug)"):
                export_data = []
                for i, d in enumerate(docs, 1):
                    src = d.metadata.get("source", "unknown")
                    score = round(float(d.metadata.get("similarity_score", 0)), 3)
                    preview = d.page_content[:400].replace("\n"," ")
                    st.markdown(f"**Chunk {i}** — `source:` `{src}` — 🔢 score: {score}\n\n{preview}…")
                    export_data.append({"rank": i, "source": src, "score": score, "content": d.page_content})
                st.download_button("⬇️ Download JSON", data=json.dumps(export_data, indent=2, ensure_ascii=False),
                                   file_name="retrieved.json", mime="application/json")
                st.download_button("⬇️ Download TXT", data="\n\n".join(
                    [f"[{d['rank']}] {d['source']} (score={d['score']})\n{d['content']}" for d in export_data]),
                    file_name="retrieved.txt", mime="text/plain")

        # Generate answer
        if has_openai_key():
            from langchain_openai import ChatOpenAI
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            PROMPT = PromptTemplate(
                template=(
                    "You are a helpful assistant. Use ONLY the provided context.\n\n"
                    "Question:\n{question}\n\n"
                    "Answer:\n"
                    "- Give a clear, direct answer.\n"
                    "- Use bullet points if listing multiple facts.\n"
                    "- If not in context, reply: \"I don’t know from the given documents.\"\n\n"
                    "Sources:\n"
                    "- Cite sources from metadata.\n"
                    "- If no sources, write: \"No sources found.\"\n\n"
                    "Context:\n{context}\n"
                ),
                input_variables=["context", "question"]
            )
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            context = "\n\n".join(d.page_content for d in docs)
            chain = LLMChain(llm=llm, prompt=PROMPT)
            answer = chain.run({"context": context, "question": user_input})
        else:
            llm = pipeline("text2text-generation", model="google/flan-t5-base")
            context = "\n\n".join(d.page_content for d in docs)
            prompt = (
                "Answer the question using ONLY the context.\n"
                "If missing, reply: 'I don’t know from the documents.'\n"
                f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
            )
            result = llm(prompt, max_new_tokens=300, temperature=0.0)
            answer = result[0]["generated_text"]

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

        # Save last answer metadata for feedback
        st.session_state["last_answer"] = {
            "question": user_input,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "index": st.session_state.get("active_index"),
            "backend": vector_store_choice,
            "retrieval_time_ms": retrieval_time,
            "retrieved_chunks": retrieved_count,
        }
        st.session_state["last_docs"] = [
            {"source": d.metadata.get("source","unknown"),
             "score": float(d.metadata.get("similarity_score", 0)),
             "content": d.page_content[:400]} for d in docs
        ]

    # Feedback buttons
    if st.session_state.get("last_answer"):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("👍 Helpful"):
                fb = {**st.session_state["last_answer"], "feedback": "positive", "docs": st.session_state.get("last_docs", [])}
                save_jsonl(Path(FEEDBACK_FILE), [fb])
                st.success("✅ Feedback saved")
                st.session_state["last_answer"] = None
        with c2:
            if st.button("👎 Not Helpful"):
                fb = {**st.session_state["last_answer"], "feedback": "negative", "docs": st.session_state.get("last_docs", [])}
                save_jsonl(Path(FEEDBACK_FILE), [fb])
                save_jsonl(Path(HARD_FILE), [{"question": fb["question"], "index": fb["index"]}])
                st.error("❌ Feedback saved & added to hard questions")
                st.session_state["last_answer"] = None

# =========================
# Admin Dashboard
# =========================
else:
    st.header("📊 Admin Dashboard")

    data = load_jsonl(Path(FEEDBACK_FILE))
    if not data:
        st.warning("⚠️ No feedback yet.")
    else:
        df = pd.DataFrame(data)

        st.subheader("📋 Feedback Records (latest 200)")
        st.dataframe(df.tail(200), use_container_width=True)

        # Top problematic questions
        st.subheader("🚩 Top Problematic Questions (most 'negative')")
        if "feedback" in df.columns and "question" in df.columns:
            prob = (df[df["feedback"]=="negative"]
                    .groupby("question")["feedback"]
                    .count().sort_values(ascending=False)
                    .head(10))
            st.table(prob)

        # Backend comparison (accuracy)
        st.subheader("✅ Accuracy by Backend")
        if "backend" in df.columns and "feedback" in df.columns:
            acc = df.groupby("backend")["feedback"].value_counts().unstack(fill_value=0)
            acc["Total"] = acc.sum(axis=1)
            acc["Accuracy %"] = (acc.get("positive", 0) / acc["Total"] * 100).round(2)
            st.dataframe(acc, use_container_width=True)

        # Speed comparison
        if "retrieval_time_ms" in df.columns and "backend" in df.columns:
            st.subheader("⚡ Retrieval Speed (avg ms) by Backend")
            speed = df.groupby("backend")["retrieval_time_ms"].mean().round(2).reset_index()
            st.dataframe(speed, use_container_width=True)

        # Retrieved chunks comparison
        if "retrieved_chunks" in df.columns and "backend" in df.columns:
            st.subheader("📦 Retrieved Chunks (avg) by Backend")
            chunks = df.groupby("backend")["retrieved_chunks"].mean().round(2).reset_index()
            st.dataframe(chunks, use_container_width=True)

        # Feedback vs Similarity scatter
        st.subheader("📈 Feedback vs Similarity Score")
        plot_points = []
        for row in data:
            fb = row.get("feedback","")
            for d in row.get("docs", []):
                sc = d.get("score", None)
                if sc is not None:
                    plot_points.append({"feedback": fb, "score": sc})
        if plot_points:
            dfp = pd.DataFrame(plot_points)
            fig, ax = plt.subplots()
            colors = dfp["feedback"].map({"positive":"green","negative":"red"}).fillna("gray")
            ax.scatter(dfp["score"], range(len(dfp)), c=colors, alpha=0.6)
            ax.set_xlabel("Similarity Score (Cross-Encoder)")
            ax.set_ylabel("Instance")
            st.pyplot(fig)

        # Per-question detail view
        st.subheader("🔎 Per-Question Detail")
        qlist = df["question"].dropna().unique().tolist()
        selq = st.selectbox("Select Question", ["(None)"] + qlist, index=0)
        if selq != "(None)":
            row = df[df["question"] == selq].tail(1).iloc[0]
            retrieved_docs = row.get("docs", [])
            if retrieved_docs:
                for i, d in enumerate(retrieved_docs, 1):
                    st.markdown(f"**Chunk {i}** — Source: `{d.get('source','unknown')}` — 🔢 Score: {round(float(d.get('score',0)),3)}\n\n{d.get('content','')[:500]}…")
                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(retrieved_docs, indent=2, ensure_ascii=False),
                    file_name=f"{selq.replace(' ','_')}_retrieved.json",
                    mime="application/json"
                )
