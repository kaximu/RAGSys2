import os
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


# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="💬 RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot with Feedback & Admin Dashboard")

INDEX_DIR = Path(r"D:\SpikeUp.AI\Extracted\indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# Utils
# -----------------------
def crawl_website(start_url, max_pages=5):
    """Crawl domain pages and extract text"""
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
        except Exception as e:
            print(f"⚠️ Skipping {url}: {e}")
    return results


def extract_file_text(uploaded_file):
    """Extract text from TXT or PDF"""
    if uploaded_file.type == "text/plain":
        return uploaded_file.name, uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return uploaded_file.name, text
    return uploaded_file.name, None


def chunk_labeled_texts(labeled_texts, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for source, text in labeled_texts:
        if not text:
            continue
        chunks = splitter.split_text(text)
        docs.extend([Document(page_content=c, metadata={"source": source}) for c in chunks])
    return docs


def get_embeddings():
    """Pick embeddings: OpenAI if key exists, else local HuggingFace"""
    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"].startswith("sk-"):
        from langchain_openai import OpenAIEmbeddings
        st.sidebar.success("✅ Using OpenAI embeddings")
        return OpenAIEmbeddings(), "openai"
    else:
        st.sidebar.info("✅ Using local embeddings (free)")
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
# Sidebar: Build Index
# -----------------------
st.sidebar.header("⚙️ Settings")
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
    if source_mode == "Website" and website_url:
        with st.spinner(f"Crawling {website_url}..."):
            labeled = crawl_website(website_url, max_pages=max_pages)
        index_name = urlparse(website_url).netloc.replace(".", "_")
    elif source_mode == "Upload File" and uploaded_files:
        with st.spinner("Extracting text from files..."):
            for uf in uploaded_files:
                name, txt = extract_file_text(uf)
                if txt and txt.strip():
                    labeled.append((name, txt))
        index_name = "multi_files" if len(uploaded_files) > 1 else uploaded_files[0].name.replace(".", "_")

    if labeled and index_name:
        docs = chunk_labeled_texts(labeled)
        embeddings, emb_type = get_embeddings()
        index_path = INDEX_DIR / f"{index_name}_{emb_type}"
        db = FAISS.from_documents(docs, embedding=embeddings)
        db.save_local(str(index_path))
        save_metadata(index_path, index_description or "No description provided")
        st.sidebar.success(f"✅ Index built: {index_path.name}")
    else:
        st.sidebar.error("❌ Nothing to index")


# -----------------------
# Index Management
# -----------------------
available_indexes = [p for p in INDEX_DIR.glob("*") if p.is_dir()]
if not available_indexes:
    st.warning("⚠️ No indexes found.")
    st.stop()

search_filter = st.sidebar.text_input("🔍 Search Indexes")
metadata_list = [load_metadata(p) for p in available_indexes]
if search_filter:
    metadata_list = [m for m in metadata_list if search_filter.lower() in m["name"].lower() or search_filter.lower() in m["description"].lower()]
if not metadata_list:
    st.sidebar.warning("⚠️ No matches.")
    st.stop()

options = [f"{m['name']} → {m['description']} (📅 {m['last_updated']})" for m in metadata_list]
selected_display = st.sidebar.selectbox("📑 Select Index", options)
selected_index = metadata_list[options.index(selected_display)]

# Load index
embeddings, _ = get_embeddings()
db = FAISS.load_local(str(INDEX_DIR / selected_index["name"]), embeddings, allow_dangerous_deserialization=True)

st.info(f"**📑 Index:** `{selected_index['name']}`\n\n"
        f"🏷️ {selected_index['description'] or 'No description'}\n\n"
        f"🕒 Last Updated: {selected_index['last_updated']}")


# -----------------------
# App Mode: Chatbot | Admin Dashboard
# -----------------------
app_mode = st.sidebar.radio("📌 Select App Mode", ["Chatbot", "Admin Dashboard"])


# -----------------------
# Chatbot Mode
# -----------------------
if app_mode == "Chatbot":
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    prompt_template = """
    You are a helpful assistant. Use ONLY the information in the context below.
    If you don’t know the answer from the context, say "I don’t know."

    Context:
    {context}

    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    mode = st.sidebar.radio("🤖 Chat Mode:", ["Retrieval Only", "Full Chatbot (RAG)"])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = None
    if "feedback_log" not in st.session_state:
        st.session_state["feedback_log"] = "feedback_log.jsonl"

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask me something..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if mode == "Retrieval Only":
            docs = db.max_marginal_relevance_search(user_input, k=4, fetch_k=12)
            parts = ["### 🔎 Retrieved Chunks"]
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "unknown")
                parts.append(f"**Chunk {i}** — *Source:* `{src}`\n\n{doc.page_content[:500]}…")
            answer = "\n\n".join(parts)
        else:
            retriever = db.as_retriever(search_kwargs={"k": 8, "fetch_k": 20})
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True,
            )
            result = qa(user_input)
            answer = f"{result['result']}\n\n### 📑 Sources:\n"
            for doc in result["source_documents"]:
                src = doc.metadata.get("source", "unknown")
                preview = doc.page_content[:200].replace("\n", " ")
                answer += f"- `{src}` → {preview}…\n"

        st.session_state["last_answer"] = {
            "question": user_input,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        }

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})

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
                    f.write(json.dumps({"question": fb["question"]}) + "\n")
                st.error("❌ Feedback saved & added to hard questions")
                st.session_state["last_answer"] = None


# -----------------------
# Admin Dashboard
# -----------------------
else:
    st.header("📊 Admin Dashboard: Feedback Analytics")

    feedback_file = st.session_state.get("feedback_log", "feedback_log.jsonl")
    hard_file = "hard_questions.jsonl"

    # Load feedback logs
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

        # Summary
        st.subheader("📊 Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👍 Positive", (df["feedback"] == "positive").sum())
        with col2:
            st.metric("👎 Negative", (df["feedback"] == "negative").sum())

        # Filter
        filter_choice = st.radio("Filter by feedback:", ["All", "Positive", "Negative"])
        if filter_choice != "All":
            df = df[df["feedback"] == filter_choice.lower()]

        # Show table
        st.subheader("📋 Feedback Records")
        st.dataframe(
            df[["timestamp", "question", "feedback", "answer"]]
            .sort_values("timestamp", ascending=False),
            use_container_width=True,
            height=400,
        )

        # Problematic questions
        st.subheader("🚨 Top Problematic Questions (👎 Negative)")
        if (df["feedback"] == "negative").any():
            neg_df = df[df["feedback"] == "negative"]
            top_neg = (
                neg_df.groupby("question")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(10)
            )
            st.table(top_neg)
        else:
            st.info("✅ No negative feedback yet!")

        # Export feedback
        st.download_button(
            "⬇️ Download Feedback (JSONL)",
            data="\n".join([json.dumps(r) for r in data]),
            file_name="feedback_log.jsonl",
            mime="application/json",
        )

    # Hard questions section
    st.subheader("🧪 Hard Questions (for retraining)")
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
            st.dataframe(
                hard_df.sort_values("question"),
                use_container_width=True,
                height=300,
            )
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
