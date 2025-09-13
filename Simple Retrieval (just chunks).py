from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Decide which index to load
if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"].startswith("sk-"):
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    index_path = r"D:\SpikeUp.AI\Project Futere Facts\faiss_index_openai"
    print("✅ Using OpenAI embeddings")
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = r"D:\SpikeUp.AI\Project Futere Facts\faiss_index_local"
    print("✅ Using local embeddings")

# Load FAISS
db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Ask a question
query = input("❓ Ask your question: ")
docs = db.similarity_search(query, k=3)

print("\n--- Retrieved Chunks ---")
for i, doc in enumerate(docs, start=1):
    print(f"\n[{i}] {doc.page_content[:400]}...")
