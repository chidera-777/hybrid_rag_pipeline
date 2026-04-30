from typing import Dict, Any, Optional
from agent.tools.base_tool import BaseTool, ToolResult
from tavily import TavilyClient
import logging
import dotenv
import os

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

class WebSearchTool(BaseTool):
    """
    Search the web for information not in the knowledge base.
    
    Faithfulness: FALSE
    - Output comes from external web sources
    - Should be clearly marked in final answer
    - Only available in RELAXED mode
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="web_search",
            description="Search the web for current information not in the knowledge base. Use this for recent events, external facts, or when KB search returns no results.",
            faithful=False,
            requires_auth=True
        )
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not found. Web search will not work.")
    
    def execute(self, input_data: str, context: Optional[Dict[str, Any]] = None):
        """
        Execute web search using Tavily.
        
        Args:
            input_data: Search query
            context: Optional context
        
        Returns:
            ToolResult with search results
        """
        try:
            if not self.api_key:
                return ToolResult(
                    success=False,
                    output="",
                    error="TAVILY_API_KEY not configured"
                )
            
            logger.info(f"Web search requested for: {input_data}")
            
            tavily_client = TavilyClient(api_key=self.api_key)
            response = tavily_client.search(
                query=input_data,
                max_results=5,
                search_depth="basic",
                include_answer=True
            )
            formatted_output = self._format_results(response)
            
            return ToolResult(
                success=True,
                output=formatted_output,
                metadata={
                    "source": "web",
                    "query": input_data,
                    "faithful": False,
                    "result_count": len(response.get('results', [])),
                    "answer": response.get('answer')
                }
            )
            
        except Exception as e:
            logger.error(f"Error in web_search tool: {e}")
            return ToolResult(
                success=False,
                output="",
                error=f"Web search failed: {str(e)}"
            )
    
    def _format_results(self, response: dict):
        """Format Tavily search results."""
        results = response.get('results', [])
        answer = response.get('answer')
        
        if not results and not answer:
            return "No web results found."
        
        output = ""
        
        if answer:
            output += f"[Tavily Answer]\n{answer}\n\n"
        
        if results:
            output += f"Found {len(results)} web sources:\n\n"
            for i, result in enumerate(results, 1):
                output += f"[Web Source {i}]\n"
                output += f"Title: {result.get('title', 'N/A')}\n"
                output += f"Content: {result.get('content', 'N/A')}\n"
                output += f"URL: {result.get('url', 'N/A')}\n"
                output += f"Score: {result.get('score', 'N/A')}\n\n"
        
        return output
