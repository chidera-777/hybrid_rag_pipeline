import time
from pathlib import Path
from typing import List, Dict, Optional
from ingestion.pdf_loader import PDFLoader
from ingestion.web_loader import WebLoader
from ingestion.base_loader import Document
from ingestion.chunker import Chunker
from vectorstore.qdrant_store import QdrantStore
from retrieval.sparse_retriever import SparseRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from reranker.cross_encoder import Reranker
from generation.generator import Generator
from config import *


class RAGPipeline:
    def __init__(self, rebuild_index:bool=False, use_diversity:bool=False):
        self.use_diversity = use_diversity
        self.vector_store = QdrantStore(
            collection_name=COLLECTION_NAME,
            host=QDRANT_HOST,
            port=QDRANT_PORT,
        )
        stats = self.vector_store.get_collection_stats()
        if stats["total_chunks"] == 0:
            raise RuntimeError("No chunks found in the collection. Please run the ingestion pipeline first.")
        self.chunks = self.load_all_chunks()
        self.sparse_retriever = SparseRetriever(self.chunks)
        self.reranker = Reranker()
        self.generator = Generator(use_local=True)
        
        
        def load_all_chunks(self):
            chunks = []
            offset = None
            while True:
                records, offset = self.vector_store.client.scroll(
                    collection_name=self.vector_store.collection_name,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vector=False
                )
                
                if not records:
                    break
                
                for record in records:
                    chunk = Document(
                        content=record.payload.get("content"),
                        metadata={k:v for k,v in record.payload.items() if k not in ["content", "content_hash"]}
                    )
                    chunks.append(chunk)
                if offset is None:
                    break
            return chunks