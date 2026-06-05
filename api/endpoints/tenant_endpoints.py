from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from api.auth import authenticate_tenant
from api.pipeline_helper import tenant_pipelines
from mlops.pipeline import MLOpsPipeline
from mlops.storage_manager import StorageManager
from schemas.tenant import TenantRegisteration, TenantConfig, TenantResponse
from datetime import datetime, timedelta
import secrets
import uuid
import yaml
import json

tenant_router = APIRouter(prefix="/api/tenant", tags=["Tenant Management"])

mlops = MLOpsPipeline()
storage_manager = StorageManager()


@tenant_router.post("/register", response_model=TenantResponse)
async def register_tenant(tenant: str = Form(...), config_file: UploadFile = File(...)):
    """
    Register a new tenant with the RAG service.
    
    Parameters:
    - tenant (str): JSON string containing the tenant configuration (see example below).
    - config_file (UploadFile): YAML or JSON file containing the tenant configuration.
    
    Returns:
    - TenantResponse: The response containing the tenant ID, API key, and other details.
    
    Note:
    - S3 buckets and region are automatically created if the storage type is 'managed', so this field is not required.
    - For 'own_s3' storage type, the bucket must already exist and be accessible to the RAG service (S3 bucket and region required).
    
    Example:
    - tenant = {
        "company_name": "MyCompany",
        "contact_email": "mycompany@example.com",
        "storage_type": "managed",
        "s3_bucket": "my-company-docs",
        "s3_region": "eu-west-1"
    }
    """
    try:
        tenant_dict = json.loads(tenant) if isinstance(tenant, str) else tenant
        tenant = TenantRegisteration(**tenant_dict)
        contents = await config_file.read()
        if config_file.filename.endswith((".yaml", ".yml")):
            config_dict = yaml.safe_load(contents)
        elif config_file.filename.endswith(".json"):
            config_dict = json.loads(contents)
        else:
            raise HTTPException(status_code=400, detail="Invalid config file format (must be .yaml or .json)")
        
        tenant_config = TenantConfig(**config_dict)
        mlops.validate_tenant_config(config_dict)
        
        tenant_id = str(uuid.uuid4())
        api_key = f"rsk_{secrets.token_urlsafe(32)}"
        if tenant.storage_type == "managed":
            storage_info = storage_manager.create_managed_storage(tenant_id)
            next = (
                "🎉 Registration successful!\n\n"
                "IMPORTANT: Store these AWS credentials securely.\n\n"
                "Next steps:\n"
                "1. Configure AWS CLI with the provided credentials:\n"
                f"   aws configure set aws_access_key_id {storage_info['access_credentials']['access_key_id']}\n"
                "   aws configure set aws_secret_access_key <secret_access_key>\n"
                f"   aws configure set region {storage_info['region']}\n\n"
                "2. Upload your documents anytime:\n"
                f"   aws s3 cp documents/ s3://{storage_info['bucket']}/{storage_info['prefix']} --recursive\n\n"
                "3. After uploading, build your index:\n"
                f"   curl -X POST 'http://your-api.com/api/tenant/{tenant_id}/build' \\\n"
                f"     -H 'X-API-Key: {api_key}'\n\n"
                "4. Check status until 'ready':\n"
                "   curl 'http://your-api.com/api/tenant/status' \\\n"
                f"     -H 'X-API-Key: {api_key}'\n\n"
                "5. Query your documents:\n"
                "   curl -X POST 'http://your-api.com/api/query' \\\n"
                f"     -H 'X-API-Key: {api_key}' \\\n"
                "     -H 'Content-Type: application/json' \\\n"
                "     -d request='{\"question\": \"Your question here\"}'\n\n"
                "Lost credentials? Call GET /tenant/credentials with your API key.\n"
                "Need to change credentials? Call POST /tenant/credentials/rotate"
            )
        else:
            if not tenant.s3_bucket:
                raise HTTPException(status_code=400, detail="S3 bucket is required")
            if not storage_manager.validate_tenant_bucket(tenant.s3_bucket, tenant.s3_region):
                raise HTTPException(status_code=400, detail=f"Cannot access S3 bucket {tenant.s3_bucket}")
            storage_info = {
                "type": "own_s3",
                "bucket": tenant.s3_bucket,
                "region": tenant.s3_region,
                "prefix": ""
            }
            next = (
                "🎉 Registration successful!\n\n"
                "Next steps:\n"
                "1. Upload your documents to your S3 bucket:\n"
                f"   aws s3 cp documents/ s3://{tenant.s3_bucket}/ --recursive\n\n"
                "2. Build your index:\n"
                f"   curl -X POST 'http://your-api.com/api/tenant/{tenant_id}/build' \\\n"
                f"     -H 'X-API-Key: {api_key}'\n\n"
                "3. Check status until 'ready':\n"
                "   curl 'http://your-api.com/api/tenant/status' \\\n"
                f"     -H 'X-API-Key: {api_key}'\n\n"
                "4. Query your documents:\n"
                "   curl -X POST 'http://your-api.com/api/query' \\\n"
                f"     -H 'X-API-Key: {api_key}' \\\n"
                "     -H 'Content-Type: application/json' \\\n"
                "     -d request='{\"question\": \"Your question here\"}'\n\n"
            )
        mlops.store_tenant(tenant_id, api_key, tenant.company_name, tenant.contact_email, storage_info, config_dict)
        return TenantResponse(
            tenant_id=tenant_id,
            api_key=api_key,
            company_name=tenant.company_name,
            storage_type=tenant.storage_type,
            storage_info=storage_info,
            status="awaiting_data",
            created_at=mlops.get_tenant_by_tenant_id(tenant_id)["created_at"],
            next_steps=next
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register tenant: {str(e)}"
        )


@tenant_router.get("/status")
async def get_tenant_status(tenant: dict = Depends(authenticate_tenant)):
    """
    Get the status of a tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    
    Returns:
    - TenantResponse: The response containing the tenant ID and other details.
    """
    return {
        "tenant_id": tenant["tenant_id"],
        "company_name": tenant["company_name"],
        "contact_email": tenant["contact_email"],
        "status": tenant["status"],
        "storage_type": tenant["storage_info"]["type"],
        "chunks_indexed": tenant.get("chunks_indexed", 0),
        "files_processed": tenant.get("files_processed", 0),
        "last_build_duration": tenant.get("last_build_duration", 0),
        "last_build_at": tenant.get("last_build_at", "N/A"),
        "error": tenant.get("error") if tenant["status"] == "failed" else None
    }


@tenant_router.get("/credentials")
async def get_tenant_credentials(tenant: dict = Depends(authenticate_tenant)):
    """
    Get the credentials of a tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    
    Returns:
    - dict: A dictionary containing the tenant's S3 bucket, prefix, region, IAM user, credentials, and other details.    
    """
    storage_info = tenant["storage_info"]
    if storage_info["type"] != "managed":
        raise HTTPException(status_code=400, detail="Credentials are only available for managed storage.")
    return {
        "bucket": storage_info["bucket"],
        "prefix": storage_info["prefix"],
        "region": storage_info["region"],
        "iam_user": storage_info["iam_user"],
        "credentials": storage_info.get("access_credentials", {"note": "Credentials not available. Contact support at ferdinandchidera49@gmail.com"}),
        "instructions": (
            "1. Configure AWS CLI with the provided credentials:\n"
            f"   aws configure set aws_access_key_id {storage_info['access_credentials']['access_key_id']}\n"
            "   aws configure set aws_secret_access_key <secret_access_key>\n"
            f"   aws configure set region {storage_info['region']}\n\n"
        )
    }


@tenant_router.post("/credentials/rotate")
async def rotate_tenant_credentials(tenant: dict = Depends(authenticate_tenant)):
    """
    Change the credentials of a tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    
    Returns:
    - dict: A dictionary containing the new credentials and instructions for updating the configuration.
    """
    if tenant["storage_info"]["type"] != "managed":
        raise HTTPException(status_code=400, detail="Credentials are only available for managed storage")
    try:
        storage_info = tenant["storage_info"]
        new_credentials = storage_manager.rotate_tenant_credentials(tenant["tenant_id"])
        storage_info["access_credentials"] = {
            "access_key_id": new_credentials["access_key_id"],
            "secret_access_key": new_credentials["secret_access_key"],
            "note": "Store these credentials securely. They do not expire."
        }
        
        mlops.tenants_table.update_item(
            Key={'tenant_id': tenant['tenant_id']},
            UpdateExpression='SET storage_info = :info',
            ExpressionAttributeValues={
                ':info': json.dumps(storage_info)
            }
        )
        return {
            "message": "Credentials updated successfully.",
            "new_credentials": new_credentials,
            "instructions": (
                "Old Credentials have been revoked. Update your configuration:\n"
                f"   aws configure set aws_access_key_id {new_credentials['access_key_id']}\n"
                "   aws configure set aws_secret_access_key <secret_access_key>\n"
            )
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update credentials: {str(e)}"
        )


@tenant_router.put("/config", response_model=dict)
async def update_tenant_config(config_file: UploadFile = File(...), tenant: dict = Depends(authenticate_tenant)):
    """
    Update tenant configuration (Qdrant URL, API key, collection name).
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - config_file (UploadFile): YAML or JSON file containing the updated tenant configuration.
    
    Returns:
    - dict: A dictionary containing the success message and updated configuration.
    """
    try:
        contents = await config_file.read()
        if config_file.filename.endswith((".yaml", ".yml")):
            config_dict = yaml.safe_load(contents)
        elif config_file.filename.endswith(".json"):
            config_dict = json.loads(contents)
        else:
            raise HTTPException(status_code=400, detail="Invalid config file format (must be .yaml or .json)")
        
        tenant_config = TenantConfig(**config_dict)
        mlops.validate_tenant_config(config_dict)
        
        mlops.tenants_table.update_item(
            Key={'tenant_id': tenant['tenant_id']},
            UpdateExpression='SET config = :config',
            ExpressionAttributeValues={
                ':config': json.dumps(config_dict)
            }
        )
        
        keys_to_delete = [k for k in tenant_pipelines.keys() if k.startswith(tenant['tenant_id'])]
        for key in keys_to_delete:
            del tenant_pipelines[key]
        
        return {
            "message": "Configuration updated successfully",
            "tenant_id": tenant['tenant_id'],
            "updated_config": config_dict,
            "note": "Pipeline cache cleared. New configuration will be used on next query."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration: {str(e)}"
        )


@tenant_router.delete("")
async def delete_tenant(tenant: dict = Depends(authenticate_tenant)):
    """
    Delete a tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    
    Returns:
    - dict: A dictionary containing the message, tenant ID, and other details.
    """
    tenant_id = tenant["tenant_id"]
    try:
        if tenant["storage_info"]["type"] == "managed":
            storage_manager.delete_tenant_data(tenant_id, tenant['storage_info'])
        mlops.tenants_table.delete_item(Key={'tenant_id': tenant_id})
        
        keys_to_delete = [k for k in tenant_pipelines.keys() if k.startswith(tenant_id)]
        for key in keys_to_delete:
            del tenant_pipelines[key]
        return {
            "message": f"Tenant {tenant_id} deleted successfully",
            "tenant_id": tenant_id,
            "note": "Data deletion may take a few minutes to complete\n Remember to manually delete your Qdrant collection if you want to remove all data."
        }  
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete tenant {tenant_id}: {str(e)}"
        )


@tenant_router.get("/metrics")
async def get_tenant_metrics(tenant: dict = Depends(authenticate_tenant), hours: int = 24, period: int = 300):
    """
    Get metrics for a tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - hours (int[query_param]): The number of hours to look back (default: 24).
    - period (int[query_param]): The period in seconds for the metrics (default: 300).
    
    Returns:
    - dict: A dictionary containing the metrics for the tenant.
    """
    end = datetime.now()
    start = end - timedelta(hours=hours)
    ns = "RAG-as-a-Service"
    dims = [{'Name': 'TenantId', 'Value': tenant['tenant_id']}]
    queries = [
        {'Id': 'querylatency', 'MetricStat': {'Metric': {'Namespace': ns, 'MetricName': 'QueryLatency', 'Dimensions': dims}, 'Period': period, 'Stat': 'Average'}, 'ReturnData': True},
        {'Id': 'tokensused', 'MetricStat': {'Metric': {'Namespace': ns, 'MetricName': 'TokensUsed', 'Dimensions': dims}, 'Period': period, 'Stat': 'Sum'}, 'ReturnData': True},
        {'Id': 'querycount', 'MetricStat': {'Metric': {'Namespace': ns, 'MetricName': 'QueryCount', 'Dimensions': dims}, 'Period': period, 'Stat': 'Sum'}, 'ReturnData': True},
        {'Id': 'buildduration', 'MetricStat': {'Metric': {'Namespace': ns, 'MetricName': 'IndexBuildDuration', 'Dimensions': dims}, 'Period': period, 'Stat': 'Average'}, 'ReturnData': True},
        {'Id': 'chunksindexed', 'MetricStat': {'Metric': {'Namespace': ns, 'MetricName': 'ChunksIndexed', 'Dimensions': dims}, 'Period': period, 'Stat': 'Sum'}, 'ReturnData': True},
        {'Id': 'filesprocessed', 'MetricStat': {'Metric': {'Namespace': ns, 'MetricName': 'FilesProcessed', 'Dimensions': dims}, 'Period': period, 'Stat': 'Sum'}, 'ReturnData': True},
        {'Id': 'buildsuccess', 'MetricStat': {'Metric': {'Namespace': ns, 'MetricName': 'IndexBuildSuccess', 'Dimensions': dims}, 'Period': period, 'Stat': 'Sum'}, 'ReturnData': True},
    ]
    resp = mlops.cloudwatch.get_metric_data(StartTime=start, EndTime=end, MetricDataQueries=queries, ScanBy='TimestampAscending')
    series = {}
    latest = {}
    for r in resp.get('MetricDataResults', []):
        pts = [{'t': ts.isoformat(), 'v': v} for ts, v in sorted(zip(r.get('Timestamps', []), r.get('Values', [])))]
        series[r['Id']] = pts
        latest[r['Id']] = pts[-1]['v'] if pts else None
    return {
        'tenant_id': tenant['tenant_id'],
        'range': {'start': start.isoformat(), 'end': end.isoformat(), 'hours': hours, 'period': period},
        'latest': latest,
        'series': series
    }
