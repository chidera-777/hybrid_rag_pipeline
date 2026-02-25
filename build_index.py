import argparse
from pathlib import Path
from ingestion.pdf_loader import PDFLoader
from ingestion.web_loader import WebLoader
from vectorstore.qdrant_store import QdrantStore
import logging
from transformers.utils import logging as hf_logging
from config import *



def main():
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description="Build the RAG Pipeline index")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=str(DOCUMENTS_DIR),
        help="Directory containing the PDF files to ingest"
    )
    parser.add_argument(
        "--web-file",
        type=str,
        default=str(WEB_FILE),
        help="Directory containing the web pages to ingest"
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force the index to be rebuilt from scratch even if they exist"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the existing index and start from scratch"
    )
    args = parser.parse_args()
    
    vector_store = QdrantStore(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )
    if args.clear:
        vector_store.clear_collection()
    
    stats = vector_store.get_collection_stats()
    all_chunks = []
    
    if Path(args.pdf_dir).exists():
        pdf_loader = PDFLoader()
        pdf_chunks = pdf_loader.load_directory(args.pdf_dir)
        all_chunks.extend(pdf_chunks)
        
    if Path(args.web_file).exists():
        with open(args.web_file, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
        web_loader = WebLoader()
        web_chunks, failed_urls = web_loader.load_urls(urls)
        all_chunks.extend(web_chunks)
        if failed_urls:
            print(f"Failed to load URLs: {failed_urls}")
    
    if all_chunks:
        upload_stats =vector_store.add_documents(all_chunks, skip_existing=not args.force_reindex)
        print("\n=== Index Summary ===")
        print(f"Total chunks processed: {upload_stats['total']}")
        print(f"New chunks added: {upload_stats['added']}")
        print(f"Existing chunks skipped: {upload_stats['skipped']}")
        
        # Show final stats
        final_stats = vector_store.get_collection_stats()
        print(f"Final collection size: {final_stats['total_chunks']} chunks")
    else:
        print("No documents found to index")
        
if __name__ == "__main__":
    main()
        