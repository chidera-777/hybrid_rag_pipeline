from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result from tool execution."""
    success: bool
    output: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseTool(ABC):
    """
    Base class for all tools in the agentic RAG system.
    
    Key principle: Tools are classified by faithfulness.
    - faithful=True: Output is grounded in tenant's knowledge base
    - faithful=False: Output comes from external sources (web, computation, etc.)
    """
    
    def __init__(self, name: str, description: str, faithful: bool, requires_auth: bool = False):
        self.name = name
        self.description = description
        self.faithful = faithful
        self.requires_auth = requires_auth
        
    @abstractmethod
    def execute(self, input_data: str, context: Optional[Dict[str, Any]] = None):
        """
        Execute the tool with given input.
        
        Args:
            input_data: The input string for the tool
            context: Optional context (e.g., tenant_id, rag_pipeline, etc.)
        
        Returns:
            ToolResult with success status, output, and metadata
        """
        pass
    
    def to_dict(self):
        """Convert tool to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "faithful": self.faithful,
            "requires_auth": self.requires_auth
        }
    
    def get_prompt_description(self):
        """
        Get tool description for inclusion in agent prompt.
        Includes faithfulness indicator.
        """
        faithful_indicator = "[FAITHFUL - KB]" if self.faithful else "[EXTERNAL]"
        return f"{self.name}{faithful_indicator}: {self.description}"
