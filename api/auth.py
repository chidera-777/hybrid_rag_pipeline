from fastapi import Header, HTTPException
from mlops.pipeline import MLOpsPipeline

mlops = MLOpsPipeline()

async def authenticate_tenant(x_api_key: str = Header(...)):
    if not x_api_key.startswith("rsk_"):
        raise HTTPException(status_code=401, detail="Invalid API key format")
    tenant = mlops.get_tenant_by_api_key(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant


async def require_tenant_ready(x_api_key: str = Header(...)):
    tenant = await authenticate_tenant(x_api_key)
    if not tenant["status"] == "ready":
        raise HTTPException(status_code=503, detail=f"Service not ready!! Current status: {tenant['status']}")
    return tenant