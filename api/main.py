from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from reranker.cross_encoder import Reranker
from generation.generator import Generator
from api.app_state import set_app_state
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@asynccontextmanager
async def lifespan(app):
    logging.info("="*60)
    logging.info("RAG-as-a-Service API Starting...")
    logging.info("="*60)
    app.state.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    app.state.reranker = Reranker()
    app.state.generator = Generator()
    
    # Initialize shared app state for all endpoint modules
    set_app_state(app.state.embedder, app.state.reranker, app.state.generator)
    
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

# Root endpoints
root_router = APIRouter(prefix="/api", tags=["Service Info"])

@root_router.get("/health")
async def health():
    """
    Check the health of the RAG service.
    
    Returns:
    - dict: A dictionary containing the service status, version, and component status.
    """
    from mlops.pipeline import MLOpsPipeline
    from mlops.storage_manager import StorageManager
    
    mlops = MLOpsPipeline()
    storage_manager = StorageManager()
    
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
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

@root_router.get("")
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

# Include all routers
app.include_router(root_router)

from api.endpoints.tenant_endpoints import tenant_router
from api.endpoints.document_endpoints import document_router
from api.endpoints.query_endpoints import query_router
from api.endpoints.tool_endpoints import tool_router
from api.endpoints.memory_endpoints import memory_router

app.include_router(tenant_router)
app.include_router(document_router)
app.include_router(query_router)
app.include_router(tool_router)
app.include_router(memory_router)
