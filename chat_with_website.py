import requests
from bs4 import BeautifulSoup
import trafilatura
import streamlit as st
from urllib.parse import urljoin, urlparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# -----------------------
# Step 1: Crawl website
# -----------------------
def crawl_website(base_url, max_pages=10):
    visited, to_visit, texts = set(), [base_url], []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue

        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded)
            if text:
                texts.append({"url": url, "text": text})

            visited.add(url)

            # extract links
            html = requests.get(url).text
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=True):
                href = urljoin(url, link["href"])
                if urlparse(href).netloc == urlparse(base_url).netloc:
                    if href not in visited:
                        to_visit.append(href)

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return texts


# -----------------------
# Step 2: Chunk text
# -----------------------
def chunk_documents(documents, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in documents:
        for chunk in splitter.split_text(doc["text"]):
            chunks.append({"text": chunk, "url": doc["url"]})
    return chunks


# -----------------------
# Step 3: Build Vector DB
# -----------------------
def build_vector_db(chunks):
    embeddings = OpenAIEmbeddings()
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["url"]} for c in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)


# -----------------------
# Step 4: QA Chain
# -----------------------
def build_qa_chain(vector_db):
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


# -----------------------
# Step 5: Streamlit App
# -----------------------
def main():
    st.title("🔎 Chat with a Website (RAG)")

    website_url = st.text_input("Enter a website URL:", "https://example.com")
    if st.button("Crawl & Build Chatbot"):
        with st.spinner("Crawling website..."):
            documents = crawl_website(website_url, max_pages=10)
        with st.spinner("Indexing content..."):
            chunks = chunk_documents(documents)
            vector_db = build_vector_db(chunks)
            st.session_state.qa_chain = build_qa_chain(vector_db)
        st.success("✅ Chatbot is ready! Ask away.")

    if "qa_chain" in st.session_state:
        query = st.text_input("Ask a question:")
        if query:
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain(query)
                st.write("### Answer")
                st.write(result["result"])

                if "source_documents" in result:
                    st.write("### Sources")
                    for doc in result["source_documents"]:
                        st.write(f"- {doc.metadata['source']}")


if __name__ == "__main__":
    main()
