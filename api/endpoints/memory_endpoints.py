from fastapi import APIRouter, HTTPException, Depends
from api.auth import authenticate_tenant
from schemas.memory import MemoryStatsResponse, MemoryClearRequest, MemoryClearResponse
from agent.memory.dynamodb_store import DynamoDBMemoryStore
from typing import Optional
import logging

logger = logging.getLogger(__name__)

memory_router = APIRouter(prefix="/api/tenant/memory", tags=["Memory Management"])


def convert_decimals(obj):
    """Recursively convert DynamoDB Decimal types to int/float."""
    from decimal import Decimal
    if isinstance(obj, list):
        return [convert_decimals(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    else:
        return obj


@memory_router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(tenant: dict = Depends(authenticate_tenant), conversation_id: Optional[str] = None):
    """
    Get memory statistics for your tenant.
    
    Parameters:
    - conversation_id (optional): Specific conversation ID to get stats for
    
    Returns:
    - Conversation memory stats (turns, last question)
    - Pattern memory stats (categories, query count)
    """
    try:
        db_store = DynamoDBMemoryStore()
        tenant_id = tenant["tenant_id"]
        
        if conversation_id:
            conv_history = db_store.get_conversation_history(tenant_id, conversation_id)
            conversation_turns = len(conv_history)
            conversation_summary = f"Conversation {conversation_id} with {conversation_turns} turns" if conv_history else "No conversation history"
        else:
            all_convs = db_store.list_conversations(tenant_id)
            conversation_turns = sum(int(c['turn_count']) for c in all_convs)
            conversation_summary = f"{len(all_convs)} conversations with {conversation_turns} total turns"
        
        pattern_stats = convert_decimals(db_store.get_pattern_statistics(tenant_id))
        
        return MemoryStatsResponse(
            tenant_id=tenant_id,
            conversation_memory_enabled=True,
            pattern_memory_enabled=True,
            conversation_turns=conversation_turns,
            conversation_summary=conversation_summary,
            pattern_statistics=pattern_stats
        )
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get memory stats: {str(e)}"
        )


@memory_router.post("/clear", response_model=MemoryClearResponse)
async def clear_memory(request: MemoryClearRequest, tenant: dict = Depends(authenticate_tenant), conversation_id: Optional[str] = None):
    """
    Clear memory for your tenant.
    
    Parameters:
    - memory_type: "conversation", "patterns", or "all"
    - conversation_id (optional): Specific conversation ID to clear (only for conversation type)
    
    Returns:
    - Confirmation message
    """
    try:
        db_store = DynamoDBMemoryStore()
        tenant_id = tenant["tenant_id"]
        
        if request.memory_type == "conversation":
            db_store.clear_conversation_history(tenant_id, conversation_id)
            message = f"Conversation memory cleared" + (f" for conversation {conversation_id}" if conversation_id else " for all conversations")
        elif request.memory_type == "patterns":
            db_store.clear_patterns(tenant_id)
            message = "Pattern memory cleared"
        elif request.memory_type == "all":
            db_store.clear_conversation_history(tenant_id)
            db_store.clear_patterns(tenant_id)
            message = "All memory cleared"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid memory type: {request.memory_type}"
            )
        
        return MemoryClearResponse(
            message=message,
            memory_type=request.memory_type,
            tenant_id=tenant_id
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
        db_store = DynamoDBMemoryStore()
        patterns = db_store.get_patterns(tenant["tenant_id"])
        
        if not patterns:
            return {"message": "No patterns to export", "patterns": []}
        
        export_data = {
            "tenant_id": tenant["tenant_id"],
            "total_patterns": len(patterns),
            "patterns": convert_decimals(patterns)
        }
        
        return export_data
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
        conversations = convert_decimals(db_store.list_conversations(tenant["tenant_id"]))
        
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

