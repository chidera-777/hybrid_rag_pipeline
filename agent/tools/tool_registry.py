from typing import Dict, List, Optional, Any
from enum import Enum
from agent.tools.base_tool import BaseTool, ToolResult
from agent.tools.retrieve_tool import RetrieveTool
from agent.tools.web_search_tool import WebSearchTool
from agent.tools.calculator_tool import CalculatorTool
from agent.tools.custom_tool import CustomTool
import logging

logger = logging.getLogger(__name__)


class ToolMode(str, Enum):
    """
    Tool execution modes.
    
    STRICT: Only faithful tools (grounded in KB)
    RELAXED: All tools, with clear source attribution
    """
    STRICT = "strict"
    RELAXED = "relaxed"


class ToolRegistry:
    """
    Central registry for managing tools.
    
    Responsibilities:
    1. Register built-in and custom tools
    2. Filter tools by mode (strict/relaxed)
    3. Execute tools with proper context
    4. Track tool usage and faithfulness
    """
    
    def __init__(self, mode: ToolMode = ToolMode.STRICT):
        self.mode = mode
        self.tools: Dict[str, BaseTool] = {}
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register built-in tools."""
        self.register_tool(RetrieveTool())
        self.register_tool(WebSearchTool())
        self.register_tool(CalculatorTool())
    
    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry."""
        if tool.name in self.tools:
            logger.warning(f"Tool {tool.name} already registered. Overwriting.")
        
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (faithful={tool.faithful})")
    
    def register_custom_tool(self, name: str, description: str, faithful: bool, endpoint_url: str, method: str = "POST", headers: Optional[Dict[str, str]] = None, auth_token: Optional[str] = None):
        """
        Register a custom tenant tool.
        
        Args:
            name: Tool name (must be unique)
            description: Tool description for agent
            faithful: Whether tool output is grounded in tenant's data
            endpoint_url: URL to call for tool execution
            method: HTTP method (POST or GET)
            headers: Optional HTTP headers
            auth_token: Optional bearer token
        """
        tool = CustomTool(
            name=name,
            description=description,
            faithful=faithful,
            endpoint_url=endpoint_url,
            method=method,
            headers=headers,
            auth_token=auth_token
        )
        self.register_tool(tool)
    
    def get_available_tools(self):
        """
        Get tools available in current mode.
        
        STRICT mode: Only faithful=True tools
        RELAXED mode: All tools
        """
        if self.mode == ToolMode.STRICT:
            return [tool for tool in self.tools.values() if tool.faithful]
        else:
            return list(self.tools.values())
    
    def get_tool(self, name: str):
        """Get a specific tool by name."""
        return self.tools.get(name)
    
    def execute_tool(self, tool_name: str, input_data: str, context: Optional[Dict[str, Any]] = None):
        """
        Execute a tool with given input.
        
        Args:
            tool_name: Name of the tool to execute
            input_data: Input for the tool
            context: Optional context (rag_pipeline, tenant_id, etc.)
        
        Returns:
            ToolResult with execution outcome
        """
        tool = self.get_tool(tool_name)
        
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' not found"
            )
        if self.mode == ToolMode.STRICT and not tool.faithful:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' is not available in STRICT mode (unfaithful tool)"
            )
        
        try:
            result = tool.execute(input_data, context)
            if result.metadata is None:
                result.metadata = {}
            result.metadata["tool_name"] = tool_name
            result.metadata["faithful"] = tool.faithful
            result.metadata["mode"] = self.mode.value
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution error: {str(e)}"
            )
    
    def get_tools_description(self):
        """
        Get formatted description of available tools for agent prompt.
        """
        available_tools = self.get_available_tools()
        
        if not available_tools:
            return "No tools available."
        
        description = f"Available tools ({self.mode.value} mode):\n\n"
        for tool in available_tools:
            description += f"- {tool.get_prompt_description()}\n"
        
        return description
    
    def set_mode(self, mode: ToolMode):
        """Change the tool execution mode."""
        self.mode = mode
        logger.info(f"Tool mode changed to: {mode.value}")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their properties."""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False
