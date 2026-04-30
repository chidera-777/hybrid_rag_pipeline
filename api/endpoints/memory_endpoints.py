from fastapi import APIRouter, HTTPException, Depends
from api.auth import authenticate_tenant
from schemas.memory import MemoryStatsResponse, MemoryClearRequest, MemoryClearResponse
from agent.memory.dynamodb_store import DynamoDBMemoryStore
from api.pipeline_helper import get_tenant_pipeline
from api.app_state import get_app_state
import json
import logging

logger = logging.getLogger(__name__)

memory_router = APIRouter(prefix="/api/tenant/memory", tags=["Memory Management"])


@memory_router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(tenant: dict = Depends(authenticate_tenant)):
    """
    Get memory statistics for your tenant.
    
    Returns:
    - Conversation memory stats (turns, last question)
    - Pattern memory stats (categories, query count)
    """
    try:
        app_state = get_app_state()
        if not app_state:
            raise HTTPException(status_code=500, detail="App state not initialized")
        
        pipeline = get_tenant_pipeline(
            tenant=tenant,
            embedder=app_state['embedder'],
            reranker=app_state['reranker'],
            generator=app_state['generator'],
            enable_agent=True,
            enable_memory=True
        )
        
        if not pipeline.memory_manager:
            return MemoryStatsResponse(
                tenant_id=tenant["tenant_id"],
                conversation_memory_enabled=False,
                pattern_memory_enabled=False
            )
        
        stats = pipeline.memory_manager.get_statistics()
        return MemoryStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get memory stats: {str(e)}"
        )


@memory_router.post("/clear", response_model=MemoryClearResponse)
async def clear_memory(request: MemoryClearRequest, tenant: dict = Depends(authenticate_tenant)):
    """
    Clear memory for your tenant.
    
    Parameters:
    - memory_type: "conversation", "patterns", or "all"
    
    Returns:
    - Confirmation message
    """
    try:
        app_state = get_app_state()
        if not app_state:
            raise HTTPException(status_code=500, detail="App state not initialized")
        
        pipeline = get_tenant_pipeline(
            tenant=tenant,
            embedder=app_state['embedder'],
            reranker=app_state['reranker'],
            generator=app_state['generator'],
            enable_agent=True,
            enable_memory=True
        )
        
        if not pipeline.memory_manager:
            raise HTTPException(
                status_code=400,
                detail="Memory not enabled for this tenant"
            )
        
        if request.memory_type == "conversation":
            pipeline.memory_manager.clear_conversation()
            message = "Conversation memory cleared"
        elif request.memory_type == "patterns":
            pipeline.memory_manager.clear_patterns()
            message = "Pattern memory cleared"
        elif request.memory_type == "all":
            pipeline.memory_manager.clear_all()
            message = "All memory cleared"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid memory type: {request.memory_type}"
            )
        
        return MemoryClearResponse(
            message=message,
            memory_type=request.memory_type,
            tenant_id=tenant["tenant_id"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear memory: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear memory: {str(e)}"
        )


@memory_router.get("/export")
async def export_memory(tenant: dict = Depends(authenticate_tenant)):
    """
    Export learned patterns as JSON.
    
    Returns:
    - JSON containing all learned patterns
    - Query categories and statistics
    """
    try:
        app_state = get_app_state()
        if not app_state:
            raise HTTPException(status_code=500, detail="App state not initialized")
        
        pipeline = get_tenant_pipeline(
            tenant=tenant,
            embedder=app_state['embedder'],
            reranker=app_state['reranker'],
            generator=app_state['generator'],
            enable_agent=True,
            enable_memory=True
        )
        
        if not pipeline.memory_manager:
            raise HTTPException(
                status_code=400,
                detail="Memory not enabled for this tenant"
            )
        
        patterns_json = pipeline.memory_manager.export_patterns()
        
        if not patterns_json:
            return {"message": "No patterns to export", "patterns": []}
        
        return json.loads(patterns_json)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export memory: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export memory: {str(e)}"
        )


@memory_router.get("/conversations")
async def list_conversations(tenant: dict = Depends(authenticate_tenant)):
    """
    List all conversations for the tenant.
    
    Returns conversation IDs with metadata:
    - conversation_id
    - turn_count
    - first_question
    - last_updated
    """
    try:
        db_store = DynamoDBMemoryStore()
        conversations = db_store.list_conversations(tenant["tenant_id"])
        
        return {
            "tenant_id": tenant["tenant_id"],
            "total_conversations": len(conversations),
            "conversations": conversations
        }
    except Exception as e:
        logger.error(f"Failed to list conversations for {tenant['tenant_id']}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list conversations: {str(e)}"
        )


@memory_router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    tenant: dict = Depends(authenticate_tenant)
):
    """
    Delete a specific conversation.
    
    Parameters:
    - conversation_id: The conversation ID to delete
    
    Returns:
    - Confirmation message
    """
    try:
        db_store = DynamoDBMemoryStore()
        db_store.clear_conversation_history(tenant["tenant_id"], conversation_id)
        
        return {
            "message": f"Conversation {conversation_id} deleted successfully",
            "tenant_id": tenant["tenant_id"],
            "conversation_id": conversation_id
        }
    except Exception as e:
        logger.error(f"Failed to delete conversation {conversation_id} for {tenant['tenant_id']}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete conversation: {str(e)}"
        )
