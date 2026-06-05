from fastapi import APIRouter, HTTPException, Depends, Body
from api.auth import authenticate_tenant
from schemas.tool import ToolRegistrationRequest, ToolRegistrationResponse, ToolListResponse
from agent.tools import ToolRegistry, ToolMode
from agent.tools.tool_store import ToolStore

tool_router = APIRouter(prefix="/api/tenant/tools", tags=["Tool Management"])

_tenant_tool_registries = {}
_tool_store = ToolStore()


def get_tool_registry(tenant_id: str, mode: str = "strict"):
    """Get or create tool registry for tenant, loading custom tools from DynamoDB."""
    if tenant_id not in _tenant_tool_registries:
        tool_mode = ToolMode.STRICT if mode == "strict" else ToolMode.RELAXED
        registry = ToolRegistry(mode=tool_mode)
        
        custom_tools = _tool_store.list_tools(tenant_id)
        for tool in custom_tools:
            registry.register_custom_tool(
                name=tool['tool_name'],
                description=tool['description'],
                faithful=tool['faithful'],
                endpoint_url=tool['endpoint_url'],
                method=tool['method'],
                headers=tool.get('headers'),
                auth_token=tool.get('auth_token')
            )
        
        _tenant_tool_registries[tenant_id] = registry
    
    # Update mode if different
    registry = _tenant_tool_registries[tenant_id]
    tool_mode = ToolMode.STRICT if mode == "strict" else ToolMode.RELAXED
    registry.set_mode(tool_mode)
    
    return registry


@tool_router.post("/register", response_model=ToolRegistrationResponse)
async def register_custom_tool(tool: ToolRegistrationRequest = Body(...), tenant: dict = Depends(authenticate_tenant)):
    """
    Register a custom tool for your tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - tool (ToolRegistrationRequest): The tool registration request containing the tool name, description, and faithfulness to the knowledge base.
    
    Returns:
    - ToolRegistrationResponse: The response containing the tool name, faithfulness, and status.
    
    Example:
    - tool = {
        "name": "enterprise_search",
        "description": "Search the enterprise search for current information not in the knowledge base. Use this for recent events, external facts, or when KB search returns no results.",
        "faithful": True,
        "endpoint_url": "https://enterprise-search.mycompany.com/api/search",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json"
        },
        "auth_token": "my-secret-token"
    }
    """
    try:
        tenant_id = tenant["tenant_id"]
        
        # Check if tool already exists
        existing_tool = _tool_store.get_tool(tenant_id, tool.name)
        if existing_tool:
            raise HTTPException(
                status_code=400,
                detail=f"Tool '{tool.name}' is already registered. Please unregister it first or use a different name."
            )
        
        # Check if it's a built-in tool name
        if tool.name in ["retrieve", "web_search", "calculator"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot register tool with reserved name '{tool.name}'. This is a built-in tool."
            )
        
        _tool_store.save_tool(
            tenant_id=tenant_id,
            tool_name=tool.name,
            description=tool.description,
            faithful=tool.faithful,
            endpoint_url=str(tool.endpoint_url),
            method=tool.method,
            headers=tool.headers,
            auth_token=tool.auth_token
        )
        registry = get_tool_registry(tenant_id)
        registry.register_custom_tool(
            name=tool.name,
            description=tool.description,
            faithful=tool.faithful,
            endpoint_url=str(tool.endpoint_url),
            method=tool.method,
            headers=tool.headers,
            auth_token=tool.auth_token
        )
        
        return ToolRegistrationResponse(
            tool_name=tool.name,
            faithful=tool.faithful,
            mode_availability={
                "strict": tool.faithful,
                "relaxed": True
            },
            message=f"Tool '{tool.name}' registered successfully. Available in {'strict and relaxed' if tool.faithful else 'relaxed'} mode."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register tool: {str(e)}"
        )


@tool_router.get("", response_model=ToolListResponse)
async def list_tools(tenant: dict = Depends(authenticate_tenant), mode: str = "strict"):
    """
    List all available tools for your tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - mode (str[query_param]): The mode to filter tools by (strict or relaxed, default: strict).

    Returns:
    - ToolListResponse: The response containing the tools, faithful count, and unfaithful count.
    """
    try:
        registry = get_tool_registry(tenant["tenant_id"], mode)
        
        all_tools = registry.list_tools()
        available_tools = [t for t in all_tools if mode == "relaxed" or t["faithful"]]
        
        faithful_count = sum(1 for t in all_tools if t["faithful"])
        unfaithful_count = len(all_tools) - faithful_count
        
        return ToolListResponse(
            mode=mode,
            tools=available_tools,
            faithful_count=faithful_count,
            unfaithful_count=unfaithful_count
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tools: {str(e)}"
        )


@tool_router.delete("/{tool_name}")
async def unregister_tool(tool_name: str, tenant: dict = Depends(authenticate_tenant)):
    """
    Unregister a custom tool.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - tool_name (str[path_param]): The name of the tool to unregister.
    
    Returns:
    - dict: A dictionary containing the message and tool name.
    """
    try:
        if tool_name in ["retrieve", "web_search", "calculator"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot unregister built-in tool '{tool_name}'"
            )
        
        tenant_id = tenant["tenant_id"]
        
        _tool_store.delete_tool(tenant_id, tool_name)
        
        registry = get_tool_registry(tenant_id)
        success = registry.unregister_tool(tool_name)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{tool_name}' not found"
            )
        
        return {
            "message": f"Tool '{tool_name}' unregistered successfully",
            "tool_name": tool_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unregister tool: {str(e)}"
        )
