# test_ingestion.py
from ingestion.pdf_loader import PDFLoader
from ingestion.web_loader import WebLoader

# Test PDF loading
pdf_loader = PDFLoader()
pdf_chunks = pdf_loader.load_directory("data/documents/")

# Test Web loading
web_loader = WebLoader()
urls = [
    "https://arxiv.org/abs/2005.11401",
    "https://arxiv.org/abs/2510.12323",
    "https://www.techrxiv.org/users/783664/articles/940787-efficient-usage-of-rag-systems-in-the-world-of-llms",
    "https://www.preprints.org/manuscript/202512.0359",
    "https://www.mdpi.com/2673-2688/7/1/15"
]
web_chunks, failed_urls = web_loader.load_urls(urls)

# Inspect results
all_chunks = pdf_chunks + web_chunks
print(f"\n--- Ingestion Summary ---")
print(f"PDF chunks: {len(pdf_chunks)}")
print(f"Web chunks: {len(web_chunks)}")
print(f"Total Failed URLs: {len(failed_urls)}")
print(f"Failed URLs \n{failed_urls}")
print(f"Total chunks: {len(all_chunks)}")

# Preview a few chunks
for chunk in all_chunks[:5]:
    print(f"\nSource: {chunk.metadata['source']}")
    print(f"Chunk ID: {chunk.metadata['chunk_id']}")
    print(f"Length: {chunk.metadata['chunk_length']} chars")
    print(f"Preview: {chunk.content[:500]}...")