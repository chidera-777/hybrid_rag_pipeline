import boto3
import sys

def clear_conversation_memory(tenant_id=None, region='eu-west-1'):
    """
    Clear conversation memory from DynamoDB.
    
    Args:
        tenant_id: If provided, clear only this tenant's data. Otherwise, clear all.
        region: AWS region
    """
    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table('RAG-ConversationMemory')
    
    if tenant_id:
        print(f"Clearing conversation memory for tenant: {tenant_id}")
        response = table.query(
            KeyConditionExpression='tenant_id = :tid',
            ExpressionAttributeValues={':tid': tenant_id}
        )
    else:
        print("Clearing ALL conversation memory...")
        response = table.scan()
    
    items = response.get('Items', [])
    
    if not items:
        print("No items found to delete.")
        return
    
    print(f"Found {len(items)} items to delete.")
    
    with table.batch_writer() as batch:
        for item in items:
            batch.delete_item(
                Key={
                    'tenant_id': item['tenant_id'],
                    'sort_key': item['sort_key']
                }
            )
            print(f"Deleted: {item['tenant_id']} - {item['sort_key']}")
    
    print(f"✅ Successfully deleted {len(items)} conversation memory items.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tenant_id = sys.argv[1]
        clear_conversation_memory(tenant_id=tenant_id)
    else:
        confirm = input("This will delete ALL conversation memory. Are you sure? (yes/no): ")
        if confirm.lower() == 'yes':
            clear_conversation_memory()
        else:
            print("Cancelled.")
