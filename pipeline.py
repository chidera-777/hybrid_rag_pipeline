import time
import os
import logging
from transformers.utils import logging as hf_logging
from typing import List
from ingestion.base_loader import Document
from vectorstore.qdrant_store import QdrantStore
from retrieval.sparse_retriever import SparseRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from reranker.cross_encoder import Reranker
import concurrent.futures
from generation.generator import Generator
from config import *


class RAGPipeline:
    def __init__(self, use_diversity:bool=False):
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.use_diversity = use_diversity
        self.vector_store = QdrantStore(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME
        )
        stats = self.vector_store.get_collection_stats()
        if stats["total_chunks"] == 0:
            raise RuntimeError("No chunks found in the collection. Please run the ingestion pipeline first.")
        self.chunks = self.load_all_chunks()
        self.sparse_retriever = SparseRetriever(self.chunks)
        self.reranker = Reranker()
        self.generator = Generator()
        
        
    def load_all_chunks(self):
        chunks = []
        offset = None
        while True:
            records, offset = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                offset=offset,
                limit=100,
                with_payload=True,
                with_vectors=False
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
    
    
    def query(self, question:str, return_metadata:bool=False):
        start_time = time.time()
        metadata = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            dense_future = executor.submit(self.vector_store.query, question)
            sparse_future = executor.submit(self.sparse_retriever.query, question)
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()
        
        fusion_time = time.time()
        fused_ids = reciprocal_rank_fusion(dense_results, sparse_results)
        chunk_map = {c.metadata["chunk_id"]:c for c in self.chunks}
        fused_chunks = [chunk_map[chunk_id] for chunk_id in fused_ids if chunk_id in chunk_map]
        metadata["fusion_time"] = time.time() - fusion_time
        
        rerank_start_time = time.time()
        if self.use_diversity:
            top_chunks = self.reranker.rerank_with_diversity(question, fused_chunks)
        else:
            top_chunks = self.reranker.rerank(question, fused_chunks, return_scores=True)
        metadata["rerank_time"] = time.time() - rerank_start_time
        
        if not self.use_diversity:
            actual_chunks = [chunk for chunk, _ in top_chunks]
            metadata["rerank_scores"] = [score for _, score in top_chunks]
        else:
            actual_chunks = top_chunks
            
        gen_start_time = time.time()
        result = self.generator.generate(question, actual_chunks)
        metadata["generation_time"] = time.time() - gen_start_time
        metadata["total_time"] = time.time() - start_time
        
        response = {
            "answer": result["answer"],
            "sources": result["sources"],
            "model": result.get("model", "Unknown"),
        }
        if return_metadata:
            response["metadata"] = metadata
        
        return response
    
    
    def batch_query(self, questions:List[str], return_metadata:bool=False):
        results = []
        for i, question in enumerate(questions):
            result = self.query(question, return_metadata=return_metadata)
            results.append(result)
            if i % 100 == 0:
                print(f"Processed {i} questions")
        return results
