from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from api.auth import authenticate_tenant
from mlops.pipeline import MLOpsPipeline
from mlops.storage_manager import StorageManager
from typing import List
import boto3

document_router = APIRouter(prefix="/api/tenant", tags=["Document Management"])

mlops = MLOpsPipeline()
storage_manager = StorageManager()


@document_router.post("/{tenant_id}/upload")
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
    """
    if tenant["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    if tenant["storage_info"]["type"] != "managed":
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


@document_router.post("/{tenant_id}/build")
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
