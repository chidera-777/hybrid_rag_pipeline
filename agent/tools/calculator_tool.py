from typing import Dict, Any, Optional
from agent.tools.base_tool import BaseTool, ToolResult
import re
import logging

logger = logging.getLogger(__name__)


class CalculatorTool(BaseTool):
    """
    Perform mathematical calculations.
    
    Faithfulness: FALSE
    - Output is computed, not from documents
    - Useful for numerical reasoning
    - Only available in RELAXED mode
    """
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Input should be a valid mathematical expression (e.g., '2 + 2', '10 * 5 + 3', '100 / 4').",
            faithful=False,
            requires_auth=False
        )
    
    def execute(self, input_data: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """
        Execute calculation.
        
        Args:
            input_data: Mathematical expression
            context: Optional context
        
        Returns:
            ToolResult with calculation result
        """
        try:
            sanitized = re.sub(r'[^0-9+\-*/().\s]', '', input_data)
            
            if not sanitized.strip():
                return ToolResult(
                    success=False,
                    output="",
                    error="Invalid mathematical expression"
                )
            try:
                result = eval(sanitized, {"__builtins__": {}}, {})
                
                return ToolResult(
                    success=True,
                    output=f"Calculation: {sanitized} = {result}",
                    metadata={
                        "expression": sanitized,
                        "result": result,
                        "source": "computation",
                        "faithful": False
                    }
                )
            except Exception as calc_error:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Calculation error: {str(calc_error)}"
                )
            
        except Exception as e:
            logger.error(f"Error in calculator tool: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )
