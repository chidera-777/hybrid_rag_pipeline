from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel


class MemoryEntry(BaseModel):
    """Single memory entry."""
    timestamp: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a new memory entry with current timestamp."""
        return cls(
            timestamp=datetime.now().isoformat(),
            content=content,
            metadata=metadata or {}
        )


class BaseMemory(ABC):
    """
    Base class for agent memory.
    
    Memory helps the agent:
    - Understand conversation context
    - Learn query patterns
    - Navigate to relevant documents more efficiently
    
    Memory does NOT:
    - Provide facts for answers
    - Replace document retrieval
    - Generate claims
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
    
    @abstractmethod
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a memory entry."""
        pass
    
    @abstractmethod
    def get_context(self, max_entries: int = 5):
        """Get memory context for agent reasoning (navigation only)."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all memory."""
        pass
    
    @abstractmethod
    def get_all(self):
        """Get all memory entries."""
        pass
