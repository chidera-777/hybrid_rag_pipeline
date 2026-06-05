from pipeline import RAGPipeline
from typing import Optional
from api.endpoints.tool_endpoints import get_tool_registry

tenant_pipelines = {}

def get_tenant_pipeline(
    tenant: dict,
    embedder,
    reranker,
    generator,
    enable_agent: bool = False,
    tool_mode: str = "strict",
    enable_memory: bool = False,
    conversation_id: Optional[str] = None
):
    """
    Get or create a pipeline for a tenant with optional conversation isolation.
    
    Parameters:
    - tenant: Tenant dict with config
    - embedder: SentenceTransformer instance
    - reranker: Reranker instance
    - generator: Generator instance
    - enable_agent: Enable agentic mode
    - tool_mode: "strict" or "relaxed"
    - enable_memory: Enable memory management
    - conversation_id: Optional conversation ID for memory isolation
    
    Returns:
    - RAGPipeline instance
    """
    tenant_id = tenant["tenant_id"]
    
    if enable_agent and enable_memory and conversation_id:
        cache_key = f"{tenant_id}_agent_{tool_mode}_mem_True_conv_{conversation_id}"
    elif enable_agent:
        cache_key = f"{tenant_id}_agent_{tool_mode}_mem_{enable_memory}"
    else:
        cache_key = tenant_id
    
    if cache_key not in tenant_pipelines:
        config = tenant["config"]
        
        # Get shared tool registry for this tenant
        tool_registry = get_tool_registry(tenant_id, tool_mode) if enable_agent else None
        
        pipeline = RAGPipeline(
            qdrant_url=config["QDRANT_URL"],
            qdrant_api_key=config["QDRANT_API_KEY"],
            collection_name=config["COLLECTION_NAME"],
            embedder=embedder,
            reranker=reranker,
            generator=generator,
            enable_agent=enable_agent,
            tool_mode=tool_mode,
            enable_memory=enable_memory,
            conversation_id=conversation_id,
            tool_registry=tool_registry,
            tenant_id=tenant_id
        )
        tenant_pipelines[cache_key] = pipeline
    
    return tenant_pipelines[cache_key]
