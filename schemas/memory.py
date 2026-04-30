from pydantic import BaseModel, Field
from typing import Optional, Literal


class MemoryStatsResponse(BaseModel):
    """Response for memory statistics."""
    tenant_id: str
    conversation_memory_enabled: bool
    pattern_memory_enabled: bool
    conversation_turns: Optional[int] = None
    conversation_summary: Optional[str] = None
    pattern_statistics: Optional[dict] = None


class MemoryClearRequest(BaseModel):
    """Request to clear memory."""
    memory_type: Literal["conversation", "patterns", "all"] = Field(
        ...,
        description="Type of memory to clear: conversation (current chat), patterns (learned patterns), or all"
    )


class MemoryClearResponse(BaseModel):
    """Response after clearing memory."""
    message: str
    memory_type: str
    tenant_id: str


class RecordInteractionRequest(BaseModel):
    """Request to manually record an interaction (for pattern learning)."""
    question: str = Field(..., min_length=1, max_length=1000)
    answer_summary: str = Field(..., min_length=1, max_length=500, description="Brief summary of answer")
    category: Optional[str] = Field(None, description="Query category (pricing, technical, support, etc.)")
    successful_docs: Optional[list] = Field(None, description="Document sources that provided good answers")


class RecordInteractionResponse(BaseModel):
    """Response after recording interaction."""
    message: str
    recorded_in: list
