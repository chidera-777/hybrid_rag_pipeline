from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Literal


class ToolRegistrationRequest(BaseModel):
    """Request to register a custom tool."""
    name: str = Field(..., min_length=1, max_length=50, pattern=r'^[a-z_][a-z0-9_]*$', 
                      description="Tool name (lowercase, underscores only)")
    description: str = Field(..., min_length=10, max_length=500,
                            description="Clear description of what the tool does")
    faithful: bool = Field(..., description="True if tool output is grounded in your data, False if external")
    endpoint_url: HttpUrl = Field(..., description="URL to call for tool execution")
    method: Literal["POST", "GET"] = Field("POST", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="Optional HTTP headers")
    auth_token: Optional[str] = Field(None, description="Optional bearer token for authentication")


class ToolRegistrationResponse(BaseModel):
    """Response after tool registration."""
    tool_name: str
    faithful: bool
    mode_availability: Dict[str, bool]
    message: str


class ToolListResponse(BaseModel):
    """Response listing all tools."""
    mode: str
    tools: list
    faithful_count: int
    unfaithful_count: int


class ToolModeUpdateRequest(BaseModel):
    """Request to update tool mode."""
    mode: Literal["strict", "relaxed"] = Field(..., description="strict: only faithful tools | relaxed: all tools")


class ToolModeUpdateResponse(BaseModel):
    """Response after mode update."""
    mode: str
    available_tools_count: int
    message: str
