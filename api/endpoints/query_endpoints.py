from fastapi import APIRouter, HTTPException, Depends, Form
from api.auth import require_tenant_ready
from api.pipeline_helper import get_tenant_pipeline
from api.app_state import get_app_state
from mlops.pipeline import MLOpsPipeline
from schemas.tenant import QueryRequest, QueryResponse, AgenticQueryRequest, AgenticQueryResponse
import json
import time

query_router = APIRouter(prefix="/api", tags=["Query"])

mlops = MLOpsPipeline()


@query_router.post("/query", response_model=QueryResponse)
async def query(request: str = Form(...), tenant: dict = Depends(require_tenant_ready)):
    """
    Query the index for a tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - request (str): JSON string containing the query request. Return metadata is optional (default: false). See example below.
    
    Returns:
    - QueryResponse: The response containing the answer, sources, model, latency, and tenant ID.
    
    Example:
    - request = {
        "question": "What is the policy of the company?",
        "return_metadata": false
    }
    """
    try:
        app_state = get_app_state()
        if not app_state:
            raise HTTPException(status_code=500, detail="App state not initialized")
        
        request_dict = json.loads(request) if isinstance(request, str) else request
        request = QueryRequest(**request_dict)
        start_time = time.time()
        pipeline = get_tenant_pipeline(
            tenant=tenant,
            embedder=app_state['embedder'],
            reranker=app_state['reranker'],
            generator=app_state['generator']
        )
        response = pipeline.query(
            question=request.question,
            return_metadata=request.return_metadata or False
        )
        latency = time.time() - start_time
        mlops.log_query_metrics(
            tenant_id=tenant["tenant_id"],
            latency=latency,
            tokens_used=len(response["answer"].split()),
        )
        
        return QueryResponse(
            answer=response["answer"],
            sources=response["sources"],
            model=response.get("model", "Unknown"),
            latency=round(latency, 2),
            tenant_id=tenant["tenant_id"],
            metadata=response.get("metadata", {})
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query: {str(e)}"
        )


@query_router.post("/query/agentic", response_model=AgenticQueryResponse)
async def agentic_query(request: str = Form(...), tenant: dict = Depends(require_tenant_ready)):
    """
    Query using multi-step agentic reasoning (ReAct) with tools.
    The agent will:
    1. Reason about what information to retrieve (navigation only)
    2. Retrieve information across multiple iterations  
    3. Generate final answer grounded exclusively in observations
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - request (str): JSON string containing the agentic query request. See example below.
    - Setting tool_mode to "relaxed" will allow the agent to use external tools (web_search, calculator, etc.), default is "strict".
    
    Returns:
    - AgenticQueryResponse: The response containing answer, sources, iterations, and optional reasoning trace.
    
    Example:
    - request = {
        "question": "What are the pricing tiers and what features does each include?",
        "max_iterations": 3,
        "return_metadata": true
        "tool_mode": "strict"
    }
    """
    try:
        app_state = get_app_state()
        if not app_state:
            raise HTTPException(status_code=500, detail="App state not initialized")
        
        request_dict = json.loads(request) if isinstance(request, str) else request
        request = AgenticQueryRequest(**request_dict)
        start_time = time.time()
        
        tool_mode = getattr(request, 'tool_mode', 'strict')
        conversation_id = getattr(request, 'conversation_id', None)
        
        pipeline = get_tenant_pipeline(
            tenant=tenant,
            embedder=app_state['embedder'],
            reranker=app_state['reranker'],
            generator=app_state['generator'],
            enable_agent=True,
            tool_mode=tool_mode,
            enable_memory=True,
            conversation_id=conversation_id
        )
        
        response = pipeline.agentic_query(
            question=request.question,
            max_iterations=request.max_iterations or 3,
            return_metadata=request.return_metadata or False
        )
        
        latency = time.time() - start_time
        mlops.log_query_metrics(
            tenant_id=tenant["tenant_id"],
            latency=latency,
            tokens_used=len(response["answer"].split()),
        )
        
        return AgenticQueryResponse(
            answer=response["answer"],
            sources=response["sources"],
            model=response.get("model", "Unknown"),
            latency=round(latency, 2),
            tenant_id=tenant["tenant_id"],
            iterations=response["iterations"],
            conversation_id=response.get("conversation_id"),
            reasoning_trace=response.get("reasoning_trace", None),
            metadata=response.get("metadata", [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute agentic query: {str(e)}"
        )
