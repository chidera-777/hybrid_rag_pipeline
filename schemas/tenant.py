from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

class TenantConfig(BaseModel):
    COLLECTION_NAME: str = Field(..., min_length=3, max_length=50, description="The name of the collection in Qdrant")
    QDRANT_URL: str = Field(..., description="The URL of the Qdrant instance")
    QDRANT_API_KEY: str = Field(..., description="The API key for the Qdrant instance")
    
    @field_validator("QDRANT_URL")
    def validate_qdrant_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("QDRANT URL must start with http:// or https://")
        return v
    

class TenantRegisteration(BaseModel):
    company_name: str = Field(..., min_length=3, max_length=50)
    contact_email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    storage_type: Literal["managed", "own_s3"] = Field(
        "managed",
        description="managed: We provide S3 | own_s3: You provide your S3 bucket"
    )
    s3_bucket: Optional[str] = Field(None, description="Required if storage_type='own_s3'")
    s3_region: Optional[str] = Field("eu-west-1", description="S3 bucket region")
    
    
class TenantResponse(BaseModel):
    tenant_id: str
    api_key: str
    company_name: str
    storage_type: str
    storage_info: dict
    status: str
    created_at: str
    next_steps: str
    
    
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    return_metadata: Optional[bool] = Field(False, description="Whether to return metadata")
    

class QueryResponse(BaseModel):
    answer: str
    sources: list
    model: str
    latency: float
    tenant_id: str
    metadata: Optional[dict] = None
