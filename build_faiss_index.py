import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# -----------------------
# Load chunks from text file
# -----------------------
chunks_file = r"D:\SpikeUp.AI\Project Futere Facts\website_chunks.txt"
chunks = []
with open(chunks_file, "r", encoding="utf-8") as f:
    current_chunk = []
    for line in f:
        if line.startswith("--- Chunk"):
            if current_chunk:
                chunks.append("".join(current_chunk).strip())
                current_chunk = []
        else:
            current_chunk.append(line)
    if current_chunk:
        chunks.append("".join(current_chunk).strip())

print(f"✅ Loaded {len(chunks)} chunks")

# Wrap into LangChain Documents
docs = [Document(page_content=txt) for txt in chunks]

# -----------------------
# Local Embeddings (HuggingFace)
# -----------------------
print("🔄 Building FAISS index with HuggingFace local embeddings...")
local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db_local = FAISS.from_documents(docs, embedding=local_embeddings)
local_index_path = r"D:\SpikeUp.AI\Project Futere Facts\faiss_index_local"
db_local.save_local(local_index_path)
print(f"📂 Saved local FAISS index to: {local_index_path}")

# -----------------------
# OpenAI Embeddings (optional)
# -----------------------
if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"].startswith("sk-"):
    try:
        from langchain_openai import OpenAIEmbeddings
        print("🔄 Building FAISS index with OpenAI embeddings...")
        openai_embeddings = OpenAIEmbeddings()
        db_openai = FAISS.from_documents(docs, embedding=openai_embeddings)
        openai_index_path = r"D:\SpikeUp.AI\Project Futere Facts\faiss_index_openai"
        db_openai.save_local(openai_index_path)
        print(f"📂 Saved OpenAI FAISS index to: {openai_index_path}")
    except Exception as e:
        print(f"⚠️ Failed to build OpenAI index: {e}")
else:
    print("⚠️ No OPENAI_API_KEY found → skipped OpenAI FAISS index")
