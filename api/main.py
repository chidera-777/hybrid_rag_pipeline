from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Header, APIRouter, Form
from fastapi.middleware.cors import CORSMiddleware
from mlops.pipeline import MLOpsPipeline
from mlops.storage_manager import StorageManager
from contextlib import asynccontextmanager
from pipeline import RAGPipeline
from typing import List
from api.auth import *
from sentence_transformers import SentenceTransformer
from reranker.cross_encoder import Reranker
from generation.generator import Generator
from schemas.tenant import *
from datetime import datetime, timedelta
import secrets
import uuid
import yaml
import json
import boto3
import time

@asynccontextmanager
async def lifespan(app):
    print("\n" + "="*60)
    print("RAG-as-a-Service API Starting...")
    print("="*60 + "\n")
    app.state.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    app.state.reranker = Reranker()
    app.state.generator = Generator()
    yield

app = FastAPI(
    title="RAG-as-a-Service",
    description="A Multi-Tenant RAG API Service with managed and self-hosted storage",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlops = MLOpsPipeline()
storage_manager = StorageManager()
tenant_pipelines = {}
router = APIRouter(prefix="/api")

@router.get("/health")
async def health():
    """
    Check the health of the RAG service.
    
    Returns:
    - dict: A dictionary containing the service status, version, and component status.
    """
    try:
        mlops.tenants_table.table_status
        storage_manager.s3.list_buckets()
        
        return {
            "status": "healthy",
            "service": "RAG-as-a-Service",
            "version": "1.0.0",
            "components": {
                "dynamodb": "connected",
                "s3": "connected",
                "iam": "connected"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

def get_tenant_pipeline(tenant: dict):
    tenant_id = tenant["tenant_id"]
    if tenant_id not in tenant_pipelines:
        config = tenant["config"]
        
        pipeline = RAGPipeline(
            qdrant_url=config["QDRANT_URL"],
            qdrant_api_key=config["QDRANT_API_KEY"],
            collection_name=config["COLLECTION_NAME"],
            embedder=app.state.embedder,
            reranker=app.state.reranker,
            generator=app.state.generator
        )
        tenant_pipelines[tenant_id] = pipeline
    return tenant_pipelines[tenant_id] 

@router.get("")
async def root():
    return {
        "service": "RAG-as-a-Service",
        "version": "1.0.0",
        "description": "A Multi-Tenant RAG API Service with managed and self-hosted storage",
        "workflow": {
            "1": "POST /tenant/register - Register your Organization",
            "2": "Upload documents to S3 (credentials provided)",
            "3": "POST /tenant/{tenant_id}/build - Build your index",
            "4": "GET /tenant/status - Check if index is ready",
            "5": "POST /query - Query your documents"
        },
        "storage_options": {
            "managed": "We provide S3 storage (easiest)",
            "own_s3": "Use your own S3 bucket (full control)"
        }
    }


@router.post("/tenant/register", response_model=TenantResponse)
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


@router.get("/tenant/status")
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
    
    
@router.get("/tenant/credentials")
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
    

@router.post("/tenant/credentials/rotate")
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
    
 
@router.post("/tenant/{tenant_id}/upload")
async def upload_tenant_data(tenant_id: str, tenant: dict = Depends(authenticate_tenant), files: List[UploadFile] = File(...)):
    """
    Upload data to a tenant's storage bucket.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - tenant_id (str[query_param]): The ID of the tenant (passed in the path).
    - files (List[UploadFile]): The files to upload.
    
    Note: Swagger UI does not support multiple file uploads. Use aws cli to upload multiple files (recommended) or curl instead.
    Example:
    - aws s3 cp documents/ s3://<bucket_name>/<prefix> --recursive
    
    - curl -X POST "http://127.0.0.1:8000/api/tenant/<tenant_id>/upload" \
    -H "X-API-Key: rsk_...your_api_key..." \
    -F "files=@/path/to/doc1.pdf" \
    -F "files=@/path/to/doc2.pdf" \
    -F "files=@/path/to/urls.txt;type=text/plain"
    
    Returns:
    - dict: A dictionary containing the upload status, uploaded files, failed files, and errors.
    
    Example:
    
    """
    if tenant["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    if tenant["storage_info"]["type"] != "own_s3":
        raise HTTPException(status_code=400, detail="Direct upload is only available for managed storage\n Use AWS CLI to upload files")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Too many files. Max of 50 per upload")
    if files is None:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    storage_info = tenant["storage_info"]
    uploaded = []
    errors = []
    for file in files:
        try:
            if file.size and file.size > 1024 * 1024 * 100:
                errors.append({"file": file.filename, "error": "File exceeds 100MB limit"})
                continue
            if not file.filename.endswith((".pdf", ".txt")):
                errors.append({"file": file.filename, "error": "Only PDF and text files are supported currently."})
                continue
            
            s3_key = f"{storage_info['prefix']}{file.filename}"
            s3 = boto3.client("s3")
            s3.upload_fileobj(file, storage_info["bucket"], s3_key)
            uploaded.append({
                "filename": file.filename,
                "size": file.size,
                "s3_key": s3_key
            })
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    return {
        "uploaded": len(uploaded),
        "failed": len(errors),
        "files": uploaded,
        "errors": errors if errors else None,
        "message": (
            f"Successfully uploaded {len(uploaded)} files. "
            f"Call POST /tenant/{tenant_id}/build to start indexing."
        )
    }
            

@router.post("/tenant/{tenant_id}/build")
async def build_index(tenant_id: str, tenant: dict = Depends(authenticate_tenant)):
    """
    Build the index for a tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - tenant_id (str[query_param]): The ID of the tenant (passed in the path).
    
    Returns:
    - dict: A dictionary containing the index build status, the number of chunks indexed, and the list of files processed.
    """
    if tenant["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    if tenant["status"] == "building":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot build index in current status {tenant['status']}\n Build already in progress"
        )
    
    files = storage_manager.list_tenant_files(tenant_id=tenant_id, storage_info=tenant["storage_info"])
    if not files:
        raise HTTPException(
            status_code=400,
            detail=f"No files found for tenant {tenant_id} to index. Please upload documents first."
        )
    is_rebuild = tenant["status"] == "ready"
    try:
        mlops.trigger_index_build(tenant_id)
        mlops.update_tenant_status(tenant_id, "building")
        return {
            "message": f"Index {'rebuild' if is_rebuild else 'build'} started",
            "tenant_id": tenant_id,
            "status": "building",
            "files_processed": len(files),
            "is_rebuild": is_rebuild
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger index build: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query(request: str = Form(...), tenant: dict = Depends(require_tenant_ready)):
    """
    Query the index for a tenant.
    
    Parameters:
    - x-api-key (str[header]): The API key of the tenant (passed in the header).
    - request (str): JSON string containing the query request. Return metadata is optional (default: false). See example below.
    
    Returns:
    - QueryResponse: The response containing the answer, sources, model, latency, and tenant ID.
    
    Example:
    - request = {
        "question": "What is the policy of the company?",
        "return_metadata": false
    }
    """
    try:
        request_dict = json.loads(request) if isinstance(request, str) else request
        request = QueryRequest(**request_dict)
        start_time = time.time()
        pipeline = get_tenant_pipeline(tenant)
        response = pipeline.query(
            question=request.question,
            return_metadata=request.return_metadata or False
        )
        latency = time.time() - start_time
        mlops.log_query_metrics(
            tenant_id=tenant["tenant_id"],
            latency=latency,
            tokens_used=len(response["answer"].split()),
        )
        
        return QueryResponse(
            answer=response["answer"],
            sources=response["sources"],
            model=response.get("model", "Unknown"),
            latency=round(latency, 2),
            tenant_id=tenant["tenant_id"],
            metadata=response.get("metadata", {})
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query: {str(e)}"
        )


@router.delete("/tenant")
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
        if tenant_id in tenant_pipelines:
            del tenant_pipelines[tenant_id]
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


@router.get("/tenant/metrics")
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

app.include_router(router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=8000,
#         log_level="info",
#         reload=True,
#     )
