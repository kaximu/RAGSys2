import os
import requests
from bs4 import BeautifulSoup
import trafilatura
import streamlit as st
from urllib.parse import urljoin, urlparse


# -----------------------
# Extract structured content with BeautifulSoup
# -----------------------
def extract_with_bs(url):
    try:
        html = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")

        text_parts = []

        # Headings, paragraphs, list items
        for tag in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li"]):
            text_parts.append(tag.get_text(separator=" ", strip=True))

        # Tables (tab-separated)
        for table in soup.find_all("table"):
            rows = []
            for row in table.find_all("tr"):
                cols = [col.get_text(separator=" ", strip=True) for col in row.find_all(["td","th"])]
                if cols:
                    rows.append("\t".join(cols))
            if rows:
                text_parts.append("\n".join(rows))

        return "\n".join(text_parts)

    except Exception as e:
        print(f"Error extracting with BeautifulSoup from {url}: {e}")
        return ""


# -----------------------
# Crawl website
# -----------------------
def crawl_website(base_url, max_pages=20, include_clean=True, include_structured=True):
    visited, to_visit, texts = set(), [base_url], []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue

        try:
            clean_text, structured_text = "", ""

            if include_clean:
                downloaded = trafilatura.fetch_url(url)
                clean_text = trafilatura.extract(downloaded) if downloaded else ""

            if include_structured:
                structured_text = extract_with_bs(url)

            # Merge based on checkboxes
            combined_text = f"\n--- Page: {url} ---\n"
            if include_clean and clean_text:
                combined_text += f"\n[CLEAN TEXT]\n{clean_text}\n"
            if include_structured and structured_text:
                combined_text += f"\n[STRUCTURED DATA]\n{structured_text}\n"

            if clean_text or structured_text:
                texts.append(combined_text)

            visited.add(url)

            # Crawl links (stay in same domain)
            html = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
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
# Streamlit App
# -----------------------
def main():
    st.title("🌐 Website Extractor (Auto Save to File)")

    website_url = st.text_input("Enter a website URL:", "https://example.com")
    max_pages = st.slider("Max pages to crawl", 1, 100, 10)

    include_clean = st.checkbox("Include clean text", value=True)
    include_structured = st.checkbox("Include structured data (tables, lists, headings)", value=True)

    # Path where the file will always be saved
    save_path = r"D:\SpikeUp.AI\Project Futere Facts\website_content.txt"  # 👈 change this path

    if st.button("Extract Data"):
        if not include_clean and not include_structured:
            st.warning("⚠️ Please select at least one extraction option.")
            return

        with st.spinner("Crawling website..."):
            texts = crawl_website(
                website_url, 
                max_pages=max_pages, 
                include_clean=include_clean, 
                include_structured=include_structured
            )
            if not texts:
                st.error("❌ Could not extract any content.")
                return

            full_text = "\n".join(texts)

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save file automatically
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(full_text)

        st.success(f"✅ Extracted {len(texts)} pages")
        st.info(f"📂 File saved automatically to: {save_path}")

        # Show preview
        st.subheader("📄 Preview of Extracted Content")
        preview_length = min(len(full_text), 2000)
        st.text(full_text[:preview_length] + ("..." if len(full_text) > preview_length else ""))


if __name__ == "__main__":
    main()
