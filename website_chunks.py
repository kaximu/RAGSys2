from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path to your extracted website file
input_file = r"D:\SpikeUp.AI\Project Futere Facts\website_content.txt"
output_file = r"D:\SpikeUp.AI\Project Futere Facts\website_chunks.txt"

# Load the text
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Initialize the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # number of characters per chunk
    chunk_overlap=50,    # overlap between chunks
    length_function=len
)

# Split into chunks
chunks = splitter.split_text(text)

print(f"✅ Split into {len(chunks)} chunks")

# Save each chunk into a text file
with open(output_file, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks, start=1):
        f.write(f"\n--- Chunk {i} ---\n")
        f.write(chunk)
        f.write("\n")

print(f"📂 Chunks saved to: {output_file}")
