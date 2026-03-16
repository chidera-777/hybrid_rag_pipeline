import boto3
import json
from datetime import datetime
from qdrant_client import QdrantClient
from mlops.storage_manager import StorageManager
from mlops.build_index import IndexBuilder
from threading import Thread
import logging
import os
import dotenv
from decimal import Decimal

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MLOpsPipeline:
    def __init__(self):
        self.dynamodb = boto3.resource("dynamodb")
        self._ensure_dynamodb_table()
        self.tenants_table = self.dynamodb.Table("rag-tenants")
        self.lambda_client = boto3.client("lambda")
        self.storage_manager = StorageManager()
        self.cloudwatch = boto3.client("cloudwatch")
        self.use_lambda = os.getenv("USE_LAMBDA", "false").lower() == "true"
        
    def _ensure_dynamodb_table(self):
        try:
            dynamodb_client = boto3.client("dynamodb")
            dynamodb_client.describe_table(TableName="rag-tenants")
        except Exception:
            dynamodb_client.create_table(
                TableName="rag-tenants",
                KeySchema=[
                    {"AttributeName": "tenant_id", "KeyType": "HASH"}
                ],
                AttributeDefinitions=[
                    {"AttributeName": "tenant_id", "AttributeType": "S"}
                ],
                BillingMode="PAY_PER_REQUEST",
                Tags=[
                    {'Key': 'Service', 'Value': 'RAG-as-a-Service'}
                ]
            )
            waiter = dynamodb_client.get_waiter('table_exists')
            waiter.wait(TableName="rag-tenants")
        except Exception as e:
            raise Exception(f"Failed to create DynamoDB table: {e}")
        
        
    def _validate_tenant_config(self, config: dict):
        try:
            client = QdrantClient(url=config["QDRANT_URL"], api_key=config["QDRANT_API_KEY"])
            client.get_collections()
            return True
        except Exception as e:
            logging.error(f"Error validating tenant config: {e}")
            return False
        
    def validate_tenant_config(self, config: dict):
        return self._validate_tenant_config(config)
        
    
    def store_tenant(self, tenant_id: str, api_key: str, company_name: str, contact_email: str, storage_info: dict, config: dict):
        storage_info_temp = storage_info.copy()
        if "access_credentials" in storage_info_temp:
            storage_info_temp["access_credentials"] = {
                "access_key_id": storage_info_temp["access_credentials"]["access_key_id"],
                "note": "Use GET /tenant/credentials to retrieve the actual credentials",
            }

        self.tenants_table.put_item(
            Item={
                "tenant_id": tenant_id,
                "api_key": api_key,
                "company_name": company_name,
                "contact_email": contact_email,
                "storage_info": json.dumps(storage_info_temp),
                "config": json.dumps(config),
                "status": "awaiting_data",
                "created_at": datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'files_uploaded': 0,
                'chunks_indexed': 0
            }
        )
        logging.info(f"Stored tenant {tenant_id} with config {config}")
        
        
    def get_tenant_by_api_key(self, api_key: str):
        response = self.tenants_table.scan(
            FilterExpression="api_key = :key",
            ExpressionAttributeValues={":key": api_key},
        )
        if not response['Items']:
            return None
        tenant = response['Items'][0]
        tenant['storage_info'] = json.loads(tenant['storage_info'])
        tenant['config'] = json.loads(tenant['config'])
        return tenant
    
    
    def get_tenant_by_tenant_id(self, tenant_id: str):
        response = self.tenants_table.get_item(
            Key={"tenant_id": tenant_id}
        )
        if not response['Item']:
            return None
        tenant = response['Item']
        tenant['storage_info'] = json.loads(tenant['storage_info'])
        tenant['config'] = json.loads(tenant['config'])
        return tenant
    
    
    def update_tenant_status(self, tenant_id: str, status: str, **kwargs):
        update_parts = ['#status = :status', '#updated_at = :updated']
        expression_values = {":status": status, ":updated": datetime.now().isoformat()}
        expression_names = {'#status': 'status', '#updated_at': 'updated_at'}
        for key, value in kwargs.items():
            name_token = f"#{key}"
            value_token = f":{key}"
            update_parts.append(f"{name_token} = {value_token}")
            expression_names[name_token] = key
            expression_values[value_token] = value
        update_expressions = 'SET ' + ', '.join(update_parts)
        
        def to_dynamodb(v):
            if isinstance(v, float):
                return Decimal(str(v))
            if isinstance(v, int):
                return Decimal(v)
            if isinstance(v, dict):
                return {k: to_dynamodb(val) for k, val in v.items()}
            if isinstance(v, list):
                return [to_dynamodb(i) for i in v]
            return v
        
        expression_values = {k: to_dynamodb(v) for k, v in expression_values.items()}
        
        self.tenants_table.update_item(
            Key={"tenant_id": tenant_id},
            UpdateExpression=update_expressions,
            ExpressionAttributeValues=expression_values,
            ExpressionAttributeNames=expression_names
        )
        logging.info(f"Updated tenant {tenant_id} status to {status}")
        
    
    def trigger_index_build(self, tenant_id: str):
        if self.use_lambda:
            return self._trigger_index_build_lambda(tenant_id)
        else:
            return self._trigger_index_build_threading(tenant_id)
        
    
    def _trigger_index_build_lambda(self, tenant_id: str):
        try:
            response = self.lambda_client.invoke(
                FunctionName="RAG-IndexBuild",
                InvocationType="Event",
                Payload=json.dumps({"tenant_id": tenant_id})
            )
            self.update_tenant_status(tenant_id, "building")
            logging.info(f"Triggered index build for tenant {tenant_id}")
            return response["ResponseMetadata"]["RequestId"]
        except Exception as e:
            logging.error(f"Error triggering index build for tenant {tenant_id}: {e}")
            self.update_tenant_status(tenant_id, "failed")
            raise
        
    
    def _trigger_index_build_threading(self, tenant_id: str):
        def build_async():
            try:
                builder = IndexBuilder(mlops=self, storage_manager=self.storage_manager)
                builder.build_index(tenant_id)
            except Exception as e:
                logging.error(f"Error building index for tenant {tenant_id}: {e}")
                self.update_tenant_status(tenant_id, "failed")
                raise
            
        thread = Thread(target=build_async, daemon=True)
        thread.start()
        
    
    def log_query_metrics(self, tenant_id: str, latency: float, tokens_used: int):
        try:
            metric_data = [
                {
                    'MetricName': 'QueryLatency',
                    'Value': latency,
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'TenantId', 'Value': tenant_id}
                    ]
                },
                {
                    'MetricName': 'TokensUsed',
                    'Value': tokens_used,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'TenantId', 'Value': tenant_id}
                    ]
                },
                {
                    'MetricName': 'QueryCount',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'TenantId', 'Value': tenant_id}
                    ]
                }
            ]
            self.cloudwatch.put_metric_data(
                Namespace='RAG-as-a-Service',
                MetricData=metric_data
            )
        except Exception as e:
            logging.error(f"Failed to log query metric: {e}")
            
    
    def log_index_build_metrics(self, tenant_id: str, duration: float, chunks_indexed: int, files_processed: int, success: bool):
        try:
            metric_data = [
                {
                    'MetricName': 'IndexBuildDuration',
                    'Value': duration,
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'TenantId', 'Value': tenant_id}
                    ]
                },
                {
                    'MetricName': 'ChunksIndexed',
                    'Value': chunks_indexed,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'TenantId', 'Value': tenant_id}
                    ]
                },
                {
                    'MetricName': 'FilesProcessed',
                    'Value': files_processed,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'TenantId', 'Value': tenant_id}
                    ]
                },
                {
                    'MetricName': 'IndexBuildSuccess',
                    'Value': 1 if success else 0,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'TenantId', 'Value': tenant_id}
                    ]
                }
            ]
            self.cloudwatch.put_metric_data(
                Namespace='RAG-as-a-Service',
                MetricData=metric_data
            )
        except Exception as e:
            logging.error(f"Failed to log index build metric: {e}")
