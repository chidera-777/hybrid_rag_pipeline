import boto3
import json
from typing import Dict, List, Optional
from decimal import Decimal


class ToolStore:
    """
    DynamoDB persistence layer for custom tool registrations.
    
    Table: RAG-CustomTools
    Primary Key: tenant_id (HASH), tool_name (RANGE)
    """
    
    def __init__(self, region: str = "eu-west-1"):
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table('RAG-CustomTools')
    
    def save_tool(self, tenant_id: str, tool_name: str, description: str, faithful: bool, 
                  endpoint_url: str, method: str = "POST", headers: Optional[Dict] = None, 
                  auth_token: Optional[str] = None):
        """Save a custom tool registration."""
        item = {
            'tenant_id': tenant_id,
            'tool_name': tool_name,
            'description': description,
            'faithful': faithful,
            'endpoint_url': endpoint_url,
            'method': method,
            'headers': json.dumps(headers or {}),
            'auth_token': auth_token or ''
        }
        
        self.table.put_item(Item=item)
    
    def get_tool(self, tenant_id: str, tool_name: str):
        """Get a specific tool registration."""
        try:
            response = self.table.get_item(
                Key={
                    'tenant_id': tenant_id,
                    'tool_name': tool_name
                }
            )
            item = response.get('Item')
            if item:
                item['headers'] = json.loads(item.get('headers', '{}'))
            return item
        except Exception:
            return None
    
    def list_tools(self, tenant_id: str):
        """List all custom tools for a tenant."""
        response = self.table.query(
            KeyConditionExpression='tenant_id = :tid',
            ExpressionAttributeValues={':tid': tenant_id}
        )
        
        items = response.get('Items', [])
        for item in items:
            item['headers'] = json.loads(item.get('headers', '{}'))
        
        return items
    
    def delete_tool(self, tenant_id: str, tool_name: str):
        """Delete a custom tool registration."""
        try:
            self.table.delete_item(
                Key={
                    'tenant_id': tenant_id,
                    'tool_name': tool_name
                }
            )
            return True
        except Exception:
            return False
    
    def delete_all_tools(self, tenant_id: str):
        """Delete all custom tools for a tenant."""
        tools = self.list_tools(tenant_id)
        
        with self.table.batch_writer() as batch:
            for tool in tools:
                batch.delete_item(
                    Key={
                        'tenant_id': tenant_id,
                        'tool_name': tool['tool_name']
                    }
                )
    
    @staticmethod
    def create_table(region: str = "eu-west-1"):
        """Create DynamoDB table for custom tools."""
        dynamodb = boto3.client('dynamodb', region_name=region)
        
        try:
            dynamodb.create_table(
                TableName='RAG-CustomTools',
                KeySchema=[
                    {'AttributeName': 'tenant_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'tool_name', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'tenant_id', 'AttributeType': 'S'},
                    {'AttributeName': 'tool_name', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            print("Created RAG-CustomTools table")
        except dynamodb.exceptions.ResourceInUseException:
            print("RAG-CustomTools table already exists")
