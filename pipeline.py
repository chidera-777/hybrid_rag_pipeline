import time
import os
import logging
from transformers.utils import logging as hf_logging
from typing import List, Optional
from ingestion.base_loader import Document
from vectorstore.qdrant_store import QdrantStore
from retrieval.sparse_retriever import SparseRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from reranker.cross_encoder import Reranker
import concurrent.futures
from generation.generator import Generator
from agent.react_agent import ReActAgent
from agent.tools import ToolMode, ToolRegistry
from agent.memory import MemoryManager
from config import *
import uuid


class RAGPipeline:
    def __init__(
        self,
        qdrant_url: str = "QDRANT_URL",
        qdrant_api_key: str = "QDRANT_API_KEY",
        collection_name: str = "COLLECTION_NAME",
        use_diversity: bool = True,
        embedder: Optional[object] = None,
        reranker: Optional[Reranker] = None,
        generator: Optional[Generator] = None,
        enable_agent: bool = False,
        tool_mode: str = "strict",
        enable_memory: bool = False,
        conversation_id: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.use_diversity = use_diversity
        self.vector_store = QdrantStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            embedder=embedder
        )
        stats = self.vector_store.get_collection_stats()
        if stats["total_chunks"] == 0:
            raise RuntimeError("No chunks found in the collection. Please run the ingestion pipeline first.")
        self.chunks = self.load_all_chunks()
        self.sparse_retriever = SparseRetriever(self.chunks)
        self.reranker = reranker or Reranker()
        self.generator = generator or Generator()
        self.enable_agent = enable_agent
        self.tool_mode = ToolMode.STRICT if tool_mode == "strict" else ToolMode.RELAXED
        self.enable_memory = enable_memory
        self.conversation_id = conversation_id
        self.agent = None
        self.tool_registry = None
        self.memory_manager = None
        if enable_agent:
            # Use provided tool_registry or create new one
            self.tool_registry = tool_registry or ToolRegistry(mode=self.tool_mode)
            if enable_memory:
                if not conversation_id:
                    conversation_id = str(uuid.uuid4())
                self.conversation_id = conversation_id
                self.memory_manager = MemoryManager(
                    tenant_id=collection_name,
                    conversation_id=conversation_id,
                    enable_conversation_memory=True,
                    enable_pattern_memory=True
                )
            self.agent = ReActAgent(
                rag_pipeline=self,
                llm=self.generator.llm,
                tool_mode=self.tool_mode,
                tool_registry=self.tool_registry,
                memory_manager=self.memory_manager,
                enable_memory=enable_memory
            )
        
        
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
    
    
    def agentic_query(self, question: str, max_iterations: int = 3, return_metadata: bool = False, stream: bool = False):
        """
        Execute agentic query with multi-step reasoning (ReAct).
        
        The agent will:
        1. Reason about what information to retrieve (navigation only)
        2. Retrieve information across multiple iterations
        3. Generate final answer grounded exclusively in observations
        
        Args:
            question: The question to answer
            max_iterations: Maximum reasoning iterations (1-5)
            return_metadata: Whether to return reasoning trace
            stream: If True, returns generator for streaming. If False, returns complete dict.
        
        Returns:
            If stream=False: Dict with answer, sources, iterations, conversation_id, and optional reasoning trace
            If stream=True: Generator yielding event dicts (conversation_id in final event)
        """
        if not self.enable_agent or not self.agent:
            raise RuntimeError("Agent not enabled. Set enable_agent=True when initializing RAGPipeline")
        
        self.agent.max_iterations = max_iterations
        result = self.agent.run(question, return_metadata=return_metadata, stream=stream)
        
        if stream:
            def stream_with_conversation_id():
                for event in result:
                    if event.get('type') == 'answer_complete' and self.enable_memory and self.conversation_id:
                        event['conversation_id'] = self.conversation_id
                    yield event
            return stream_with_conversation_id()
        
        # Non-streaming: add conversation_id to result
        if self.enable_memory and self.conversation_id:
            result["conversation_id"] = self.conversation_id
            
        if return_metadata:
            result["metadata"] = {
                "reasoning_trace": result.get("reasoning_trace"),
                "observations": result.get("observations"),
                "tool_mode": result.get("tool_mode"),
                "total_chunks_retrieved": result.get("total_chunks_retrieved")
            }
        
        return result
    
    
    def batch_query(self, questions:List[str], return_metadata:bool=False):
        results = []
        for i, question in enumerate(questions):
            result = self.query(question, return_metadata=return_metadata)
            results.append(result)
            if i % 100 == 0:
                logging.info(f"Processed {i} questions")
        return results
