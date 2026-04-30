from fastapi import APIRouter, HTTPException, Depends
from api.auth import authenticate_tenant
from api.pipeline_helper import get_tenant_pipeline
from api.app_state import get_app_state
from schemas.tool import ToolRegistrationRequest, ToolRegistrationResponse, ToolListResponse

tool_router = APIRouter(prefix="/api/tenant/tools", tags=["Tool Management"])


@tool_router.post("/register", response_model=ToolRegistrationResponse)
async def register_custom_tool(tool: ToolRegistrationRequest, tenant: dict = Depends(authenticate_tenant)):
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
        app_state = get_app_state()
        if not app_state:
            raise HTTPException(status_code=500, detail="App state not initialized")
        
        pipeline = get_tenant_pipeline(
            tenant=tenant,
            embedder=app_state['embedder'],
            reranker=app_state['reranker'],
            generator=app_state['generator'],
            enable_agent=True
        )
        
        if not pipeline.tool_registry:
            raise HTTPException(
                status_code=400,
                detail="Tool registry not available. Enable agent mode first."
            )
        
        pipeline.tool_registry.register_custom_tool(
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
        app_state = get_app_state()
        if not app_state:
            raise HTTPException(status_code=500, detail="App state not initialized")
        
        pipeline = get_tenant_pipeline(
            tenant=tenant,
            embedder=app_state['embedder'],
            reranker=app_state['reranker'],
            generator=app_state['generator'],
            enable_agent=True,
            tool_mode=mode
        )
        
        if not pipeline.tool_registry:
            return ToolListResponse(
                mode=mode,
                tools=[],
                faithful_count=0,
                unfaithful_count=0
            )
        
        all_tools = pipeline.tool_registry.list_tools()
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
        app_state = get_app_state()
        if not app_state:
            raise HTTPException(status_code=500, detail="App state not initialized")
        
        if tool_name in ["retrieve", "web_search", "calculator"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot unregister built-in tool '{tool_name}'"
            )
        
        pipeline = get_tenant_pipeline(
            tenant=tenant,
            embedder=app_state['embedder'],
            reranker=app_state['reranker'],
            generator=app_state['generator'],
            enable_agent=True
        )
        
        if not pipeline.tool_registry:
            raise HTTPException(
                status_code=400,
                detail="Tool registry not available"
            )
        
        success = pipeline.tool_registry.unregister_tool(tool_name)
        
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
