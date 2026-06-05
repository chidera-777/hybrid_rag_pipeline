from typing import Dict, Any, Optional
from agent.tools.base_tool import BaseTool, ToolResult
import requests
import logging

logger = logging.getLogger(__name__)


class CustomTool(BaseTool):
    """
    Custom tool registered by tenant.
    
    Faithfulness: Defined by tenant
    - Can be faithful (if it queries tenant's own systems with grounded data)
    - Can be unfaithful (if it queries external APIs)
    - Tenant specifies faithfulness at registration
    """
    
    def __init__(self, name: str, description: str, faithful: bool, endpoint_url: str, method: str = "POST", headers: Optional[Dict[str, str]] = None, auth_token: Optional[str] = None):
        super().__init__(
            name=name,
            description=description,
            faithful=faithful,
            requires_auth=bool(auth_token)
        )
        self.endpoint_url = endpoint_url
        self.method = method.upper()
        self.headers = headers or {}
        self.auth_token = auth_token
        
        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"
    
    def execute(self, input_data: str, context: Optional[Dict[str, Any]] = None):
        """
        Execute custom tool by calling the registered endpoint.
        
        Args:
            input_data: Input to send to the endpoint
            context: Optional context (tenant_id, etc.)
        
        Returns:
            ToolResult with endpoint response
        """
        try:
            payload = {
                "input": input_data,
                "context": context or {}
            }
            
            if self.method == "POST":
                response = requests.post(
                    self.endpoint_url,
                    json=payload,
                    headers=self.headers,
                    timeout=30
                )
            elif self.method == "GET":
                response = requests.get(
                    self.endpoint_url,
                    params={"input": input_data},
                    headers=self.headers,
                    timeout=30
                )
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unsupported HTTP method: {self.method}"
                )
            
            response.raise_for_status()
            try:
                result_data = response.json()
                output = result_data.get("output", str(result_data))
            except:
                output = response.text
            
            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "source": "custom_tool",
                    "tool_name": self.name,
                    "faithful": self.faithful,
                    "endpoint": self.endpoint_url,
                    "status_code": response.status_code
                }
            )
            
        except requests.exceptions.Timeout:
            return ToolResult(
                success=False,
                output="",
                error="Tool endpoint timeout (30s)"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling custom tool {self.name}: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Tool endpoint error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error in custom tool {self.name}: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )
    
    def to_dict(self):
        """Convert custom tool to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "type": "custom",
            "endpoint_url": self.endpoint_url,
            "method": self.method
        })
        return base_dict
