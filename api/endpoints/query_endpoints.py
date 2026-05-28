from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.responses import StreamingResponse
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
async def query(request: QueryRequest = Body(...), tenant: dict = Depends(require_tenant_ready)):
    """
    Query the index for a tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - request (QueryRequest): The query request containing the question and optional metadata flag. See example below.
    
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


@query_router.post("/query/agentic")
async def agentic_query(request: AgenticQueryRequest = Body(...), tenant: dict = Depends(require_tenant_ready), stream: bool = False):
    """
    Query using multi-step agentic reasoning (ReAct) with tools.
    The agent will:
    1. Reason about what information to retrieve (navigation only)
    2. Retrieve information across multiple iterations  
    3. Generate final answer grounded exclusively in observations
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - request (AgenticQueryRequest): The agentic query request containing question, max_iterations, tool_mode, etc. See example below.
    - stream (bool[query_param]): If true, returns Server-Sent Events stream. Default: false.
    
    ### Note: 
    - Setting tool_mode to "relaxed" will allow the agent to use external tools (web_search, calculator, etc.), default is "strict".
    - Setting return_metadata to true will return the reasoning trace, default is false.
    - max_iteration is the maximum number of iterations the agent will run, default is 3.
    - Setting stream=true will stream reasoning steps and answer generation in real-time.
    - The response will include a conversation_id if the agent is enabled and memory is enabled for new conversations, making the field an optional field, therefore, subsequent queries with the same conversation_id will continue the conversation. How to use the conversation_id is described in the example below.
    
    Returns:
    - AgenticQueryResponse (if stream=false): JSON response with answer, sources, iterations
    - StreamingResponse (if stream=true): Server-Sent Events with real-time updates
    
    Example (non-streaming):
    - request = {
        "question": "What are the pricing tiers and what features does each include?",
        "max_iterations": 3,
        "return_metadata": true,
        "tool_mode": "strict",
        "conversation_id": "1234567890"
    }
    
    Example (streaming):
    - Same request with ?stream=true query parameter
    - Returns SSE events: reasoning, answer_chunk, answer_complete, complete
    """
    try:
        app_state = get_app_state()
        if not app_state:
            raise HTTPException(status_code=500, detail="App state not initialized")
        
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
        
        # Streaming mode
        if stream:
            def sync_event_generator():
                start_time = time.time()
                conversation_id_captured = None
                
                try:
                    # Stream events as they come
                    for event in pipeline.agentic_query(
                        question=request.question,
                        max_iterations=request.max_iterations or 3,
                        stream=True
                    ):
                        if event.get('type') == 'answer_complete' and 'conversation_id' in event:
                            conversation_id_captured = event['conversation_id']
                        
                        yield f"data: {json.dumps(event)}\n\n"
                    
                    latency = time.time() - start_time
                    mlops.log_query_metrics(
                        tenant_id=tenant["tenant_id"],
                        latency=latency,
                        tokens_used=0,
                    )
                    
                    completion_event = {
                        "type": "complete",
                        "latency": round(latency, 2),
                        "tenant_id": tenant["tenant_id"]
                    }
                    
                    if conversation_id_captured:
                        completion_event["conversation_id"] = conversation_id_captured
                    
                    yield f"data: {json.dumps(completion_event)}\n\n"
                    
                except Exception as e:
                    error_event = {"type": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_event)}\n\n"
            
            return StreamingResponse(
                sync_event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # Non-streaming mode
        start_time = time.time()
        
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
            metadata=response.get("metadata", [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute agentic query: {str(e)}"
        )
