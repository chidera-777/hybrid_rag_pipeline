import sys
import os
sys.path.insert(0, '/opt/python')
sys.path.insert(0, os.path.dirname(__file__))

from mlops.build_index import IndexBuilder

def handler(event, context):
    tenant_id = event['tenant_id']
    
    try:
        builder = IndexBuilder()
        builder.build_index(tenant_id)
        
        return {
            'statusCode': 200,
            'tenant_id': tenant_id,
            'status': 'success',
            'message': 'Index built successfully'
        }
    
    except Exception as e:
        raise Exception(f"Build failed: {str(e)}")