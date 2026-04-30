from typing import Dict, Any, Optional
from agent.tools.base_tool import BaseTool, ToolResult
import concurrent.futures
from retrieval.hybrid import reciprocal_rank_fusion
import logging

logger = logging.getLogger(__name__)


class RetrieveTool(BaseTool):
    """
    Retrieve information from the tenant's knowledge base.
    
    Faithfulness: TRUE
    - All output is grounded in the tenant's indexed documents
    - Uses existing RAG pipeline (dense + sparse + rerank)
    """
    
    def __init__(self):
        super().__init__(
            name="retrieve",
            description="Search the knowledge base for relevant information. Use this for questions about documents you've indexed.",
            faithful=True,
            requires_auth=False
        )
    
    def execute(self, input_data: str, context: Optional[Dict[str, Any]] = None):
        """
        Execute retrieval from knowledge base.
        
        Args:
            input_data: Search query
            context: Must contain 'rag_pipeline' key
        
        Returns:
            ToolResult with retrieved chunks
        """
        try:
            if not context or 'rag_pipeline' not in context:
                return ToolResult(
                    success=False,
                    output="",
                    error="RAG pipeline not available in context"
                )
            
            rag_pipeline = context['rag_pipeline']
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                dense_future = executor.submit(rag_pipeline.vector_store.query, input_data)
                sparse_future = executor.submit(rag_pipeline.sparse_retriever.query, input_data)
                dense_results = dense_future.result()
                sparse_results = sparse_future.result()
            
            # Fusion
            fused_ids = reciprocal_rank_fusion(dense_results, sparse_results)
            chunk_map = {c.metadata["chunk_id"]: c for c in rag_pipeline.chunks}
            fused_chunks = [chunk_map[chunk_id] for chunk_id in fused_ids if chunk_id in chunk_map]
            
            # Rerank
            if rag_pipeline.use_diversity:
                top_chunks = rag_pipeline.reranker.rerank_with_diversity(input_data, fused_chunks, top_k=5)
            else:
                top_chunks = rag_pipeline.reranker.rerank(input_data, fused_chunks, top_k=5)
                if top_chunks and isinstance(top_chunks[0], tuple):
                    top_chunks = [chunk for chunk, _ in top_chunks]
            
            if not top_chunks:
                return ToolResult(
                    success=True,
                    output="No relevant information found in the knowledge base.",
                    metadata={"chunks_count": 0}
                )
            
            # Format output
            output = f"Retrieved {len(top_chunks)} relevant chunks:\n\n"
            for i, chunk in enumerate(top_chunks, 1):
                output += f"[Chunk {i}]\n{chunk.content}\n"
                output += f"[Metadata] {chunk.metadata}\n\n"
            
            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "chunks_count": len(top_chunks),
                    "chunks": [
                        {
                            "content": chunk.content,
                            "metadata": chunk.metadata
                        }
                        for chunk in top_chunks
                    ]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in retrieve tool: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )
