import boto3
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal


class DynamoDBMemoryStore:
    """
    DynamoDB persistence layer for agent memory.
    
    Tables:
    - ConversationMemory: Short-term conversation history (with TTL)
    - PatternMemory: Long-term query patterns
    """
    
    def __init__(self, region: str = "eu-west-1"):
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.conversation_table = self.dynamodb.Table('RAG-ConversationMemory')
        self.pattern_table = self.dynamodb.Table('RAG-PatternMemory')
    
    def save_conversation_turn(self, tenant_id: str, conversation_id: str, question: str, answer_summary: str, metadata: Optional[Dict] = None):
        """Save a conversation turn with 24-hour TTL."""
        timestamp = int(time.time())
        ttl = timestamp + (24 * 60 * 60)
        
        item = {
            'tenant_id': tenant_id,
            'sort_key': f"conversation#{conversation_id}#{timestamp}",
            'conversation_id': conversation_id,
            'question': question,
            'answer_summary': answer_summary,
            'timestamp': timestamp,
            'ttl': ttl,
            'metadata': json.dumps(metadata or {})
        }
        
        self.conversation_table.put_item(Item=item)
    
    def get_conversation_history(self, tenant_id: str, conversation_id: str, max_turns: int = 10):
        """Get recent conversation turns for a specific conversation."""
        response = self.conversation_table.query(
            KeyConditionExpression='tenant_id = :tid AND begins_with(sort_key, :prefix)',
            ExpressionAttributeValues={
                ':tid': tenant_id,
                ':prefix': f'conversation#{conversation_id}#'
            },
            ScanIndexForward=False,
            Limit=max_turns
        )
        
        items = response.get('Items', [])
        return list(reversed(items))
    
    def clear_conversation_history(self, tenant_id: str, conversation_id: Optional[str] = None):
        """Clear conversation history. If conversation_id provided, clear only that conversation."""
        if conversation_id:
            prefix = f'conversation#{conversation_id}#'
        else:
            prefix = 'conversation#'
        
        response = self.conversation_table.query(
            KeyConditionExpression='tenant_id = :tid AND begins_with(sort_key, :prefix)',
            ExpressionAttributeValues={
                ':tid': tenant_id,
                ':prefix': prefix
            }
        )
        with self.conversation_table.batch_writer() as batch:
            for item in response.get('Items', []):
                batch.delete_item(
                    Key={
                        'tenant_id': item['tenant_id'],
                        'sort_key': item['sort_key']
                    }
                )
    
    def list_conversations(self, tenant_id: str):
        """List all conversation IDs for a tenant with metadata."""
        response = self.conversation_table.query(
            KeyConditionExpression='tenant_id = :tid AND begins_with(sort_key, :prefix)',
            ExpressionAttributeValues={
                ':tid': tenant_id,
                ':prefix': 'conversation#'
            }
        )
        
        conversations = {}
        for item in response.get('Items', []):
            conv_id = item.get('conversation_id')
            if conv_id not in conversations:
                conversations[conv_id] = {
                    'conversation_id': conv_id,
                    'turn_count': 0,
                    'first_question': item.get('question'),
                    'last_updated': item.get('timestamp')
                }
            conversations[conv_id]['turn_count'] += 1
            if item.get('timestamp') > conversations[conv_id]['last_updated']:
                conversations[conv_id]['last_updated'] = item.get('timestamp')
        
        return list(conversations.values())
    
    
    def save_pattern(self, tenant_id: str, category: str, question: str, successful_docs: List[str], metadata: Optional[Dict] = None):
        """Save or update a query pattern."""
        timestamp = int(time.time())
        
        try:
            response = self.pattern_table.get_item(
                Key={
                    'tenant_id': tenant_id,
                    'category': category
                }
            )
            existing = response.get('Item', {})
            query_count = existing.get('query_count', 0) + 1
            existing_docs = existing.get('successful_docs', [])
            
            all_docs = existing_docs + successful_docs
            unique_docs = list(dict.fromkeys(all_docs))[-20:]
            
        except Exception:
            query_count = 1
            unique_docs = successful_docs[:20]
        
        item = {
            'tenant_id': tenant_id,
            'category': category,
            'query_count': query_count,
            'successful_docs': unique_docs,
            'last_question': question[:100],
            'last_updated': timestamp,
            'metadata': json.dumps(metadata or {})
        }
        
        self.pattern_table.put_item(Item=item)
    
    def get_patterns(self, tenant_id: str):
        """Get all patterns for a tenant."""
        response = self.pattern_table.query(
            KeyConditionExpression='tenant_id = :tid',
            ExpressionAttributeValues={':tid': tenant_id}
        )
        
        return response.get('Items', [])
    
    def get_pattern_by_category(self, tenant_id: str, category: str):
        """Get a specific pattern by category."""
        try:
            response = self.pattern_table.get_item(
                Key={
                    'tenant_id': tenant_id,
                    'category': category
                }
            )
            return response.get('Item')
        except Exception:
            return None
    
    def clear_patterns(self, tenant_id: str):
        """Clear all patterns for a tenant."""
        response = self.pattern_table.query(
            KeyConditionExpression='tenant_id = :tid',
            ExpressionAttributeValues={':tid': tenant_id}
        )
        
        with self.pattern_table.batch_writer() as batch:
            for item in response.get('Items', []):
                batch.delete_item(
                    Key={
                        'tenant_id': item['tenant_id'],
                        'category': item['category']
                    }
                )
    
    def get_pattern_statistics(self, tenant_id: str):
        """Get pattern statistics for a tenant."""
        patterns = self.get_patterns(tenant_id)
        
        if not patterns:
            return {
                'total_patterns': 0,
                'total_queries': 0,
                'categories': [],
                'most_common_category': None
            }
        
        total_queries = sum(p.get('query_count', 0) for p in patterns)
        categories = [p['category'] for p in patterns]
        most_common = max(patterns, key=lambda x: x.get('query_count', 0))
        
        return {
            'total_patterns': len(patterns),
            'total_queries': total_queries,
            'categories': categories,
            'most_common_category': most_common['category']
        }
    
    
    @staticmethod
    def create_tables(region: str = "eu-west-1"):
        """Create DynamoDB tables for memory storage."""
        dynamodb = boto3.client('dynamodb', region_name=region)
        
        # Conversation Memory Table
        try:
            dynamodb.create_table(
                TableName='RAG-ConversationMemory',
                KeySchema=[
                    {'AttributeName': 'tenant_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'sort_key', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'tenant_id', 'AttributeType': 'S'},
                    {'AttributeName': 'sort_key', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            print("Created RAG-ConversationMemory table")
            time.sleep(5)
            dynamodb.update_time_to_live(
                TableName='RAG-ConversationMemory',
                TimeToLiveSpecification={
                    'Enabled': True,
                    'AttributeName': 'ttl'
                }
            )
            print("Enabled TTL on RAG-ConversationMemory table")
            
        except dynamodb.exceptions.ResourceInUseException:
            print("RAG-ConversationMemory table already exists")
            try:
                dynamodb.update_time_to_live(
                    TableName='RAG-ConversationMemory',
                    TimeToLiveSpecification={
                        'Enabled': True,
                        'AttributeName': 'ttl'
                    }
                )
                print("Enabled TTL on existing RAG-ConversationMemory table")
            except Exception as e:
                print(f"TTL already enabled or error: {e}")
        
        # Pattern Memory Table
        try:
            dynamodb.create_table(
                TableName='RAG-PatternMemory',
                KeySchema=[
                    {'AttributeName': 'tenant_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'category', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'tenant_id', 'AttributeType': 'S'},
                    {'AttributeName': 'category', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            print("Created RAG-PatternMemory table")
        except dynamodb.exceptions.ResourceInUseException:
            print("RAG-PatternMemory table already exists")
