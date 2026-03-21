import sys
import os
import json
import boto3

sys.path.insert(0, '/opt/python')
sys.path.insert(0, os.path.dirname(__file__))

from mlops.build_index import IndexBuilder
from mlops.pipeline import MLOpsPipeline

def get_secrets():
    """Fetch secrets from AWS Secrets Manager"""
    client = boto3.client('secretsmanager', region_name='eu-west-1')
    secret = client.get_secret_value(SecretId='rag-pipeline/env')
    return json.loads(secret['SecretString'])

def handler(event, context):
    tenant_id = event['tenant_id']
    
    # Load secrets into environment
    secrets = get_secrets()
    for key, value in secrets.items():
        os.environ[key] = value
    
    try:
        mlops = MLOpsPipeline()
        builder = IndexBuilder(mlops=mlops)
        builder.build_index(tenant_id)
        
        return {
            'statusCode': 200,
            'tenant_id': tenant_id,
            'status': 'success',
            'message': 'Index built successfully'
        }
    
    except Exception as e:
        raise Exception(f"Build failed: {str(e)}")