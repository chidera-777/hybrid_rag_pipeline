"""
Test script for Tool Registry endpoints.

This script demonstrates how to register custom tools for your RAG agent.
"""

import requests
import json

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "your_api_key"

headers = {
    "X-API-Key": API_KEY
}


# Example 1: Faithful tool - Internal company database search
def register_company_database_tool():
    """
    Register a tool that searches your internal company database.
    This is FAITHFUL because it searches your own data sources.
    """
    tool = {
        "name": "company_database",
        "description": "Search the internal company database for employee records, project information, and organizational data. Use when KB doesn't have internal company info.",
        "faithful": True,
        "endpoint_url": "https://api.mycompany.com/database/search",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json"
        },
        "auth_token": "your-database-api-token"
    }
    
    response = requests.post(
        f"{API_URL}/api/tenant/tools/register",
        headers=headers,
        json=tool
    )
    
    print("=== Company Database Tool ===")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


# Example 2: Faithful tool - CRM integration
def register_crm_tool():
    """
    Register a tool that queries your CRM system.
    This is FAITHFUL because it accesses your customer data.
    """
    tool = {
        "name": "crm_lookup",
        "description": "Look up customer information, support tickets, and account details from the CRM system. Use for customer-specific queries.",
        "faithful": True,
        "endpoint_url": "https://crm.mycompany.com/api/v1/search",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "X-CRM-Version": "v1"
        },
        "auth_token": "your-crm-api-token"
    }
    
    response = requests.post(
        f"{API_URL}/api/tenant/tools/register",
        headers=headers,
        json=tool
    )
    
    print("=== CRM Lookup Tool ===")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


# Example 3: Unfaithful tool - Stock price checker
def register_stock_price_tool():
    """
    Register a tool that checks current stock prices.
    This is UNFAITHFUL because it uses external data not in your KB.
    """
    tool = {
        "name": "stock_price",
        "description": "Get current stock prices and market data from financial APIs. Use for real-time market information.",
        "faithful": False,
        "endpoint_url": "https://api.example-finance.com/v1/quote",
        "method": "GET",
        "headers": {
            "Content-Type": "application/json"
        },
        "auth_token": "your-finance-api-token"
    }
    
    response = requests.post(
        f"{API_URL}/api/tenant/tools/register",
        headers=headers,
        json=tool
    )
    
    print("=== Stock Price Tool ===")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


# Example 4: Unfaithful tool - Weather API
def register_weather_tool():
    """
    Register a tool that checks weather information.
    This is UNFAITHFUL because it uses external weather data.
    """
    tool = {
        "name": "weather_check",
        "description": "Get current weather conditions and forecasts for any location. Use when users ask about weather.",
        "faithful": False,
        "endpoint_url": "https://api.weather.com/v1/current",
        "method": "GET",
        "headers": {
            "Content-Type": "application/json"
        },
        "auth_token": "your-weather-api-token"
    }
    
    response = requests.post(
        f"{API_URL}/api/tenant/tools/register",
        headers=headers,
        json=tool
    )
    
    print("=== Weather Check Tool ===")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


# Example 5: Faithful tool - Document management system
def register_document_management_tool():
    """
    Register a tool that searches your document management system.
    This is FAITHFUL because it searches your internal documents.
    """
    tool = {
        "name": "doc_management",
        "description": "Search the document management system for contracts, policies, and official documents. Use for formal document retrieval.",
        "faithful": True,
        "endpoint_url": "https://docs.mycompany.com/api/search",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "X-API-Version": "2.0"
        },
        "auth_token": "your-dms-api-token"
    }
    
    response = requests.post(
        f"{API_URL}/api/tenant/tools/register",
        headers=headers,
        json=tool
    )
    
    print("=== Document Management Tool ===")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


def list_tools(mode="strict"):
    """List all registered tools."""
    response = requests.get(
        f"{API_URL}/api/tenant/tools?mode={mode}",
        headers=headers
    )
    
    print(f"=== Tools List (mode={mode}) ===")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


def unregister_tool(tool_name):
    """Unregister a tool."""
    response = requests.delete(
        f"{API_URL}/api/tenant/tools/{tool_name}",
        headers=headers
    )
    
    print(f"=== Unregister Tool: {tool_name} ===")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


if __name__ == "__main__":
    print("🔧 Tool Registry Test Script\n")
    print("=" * 60)
    print()
    
    # First, list built-in tools
    print("📋 Built-in tools (before registration):")
    list_tools(mode="strict")
    list_tools(mode="relaxed")
    
    # Register faithful tools
    print("✅ Registering FAITHFUL tools (available in strict mode):")
    register_company_database_tool()
    register_crm_tool()
    register_document_management_tool()
    
    # Register unfaithful tools
    print("⚠️ Registering UNFAITHFUL tools (only in relaxed mode):")
    register_stock_price_tool()
    register_weather_tool()
    
    # List all tools in strict mode
    print("📋 All tools in STRICT mode (faithful only):")
    list_tools(mode="strict")
    
    # List all tools in relaxed mode
    print("📋 All tools in RELAXED mode (all tools):")
    list_tools(mode="relaxed")
    
    # Unregister a custom tool
    print("🗑️ Unregistering a custom tool:")
    unregister_tool("weather_check")
    
    # List again to confirm deletion
    print("📋 Tools after deletion:")
    list_tools(mode="relaxed")
    
    print("=" * 60)
    print("✅ Test complete!")
