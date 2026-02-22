# config.py
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EVAL_DIR = DATA_DIR / "eval"

# Vector Store Settings
VECTOR_STORE_TYPE = "qdrant"
COLLECTION_NAME = "rag_docs"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Retrieval Settings
DENSE_RETRIEVAL_TOP_K = 10
SPARSE_RETRIEVAL_TOP_K = 10
RRF_K = 60

# Generation Settings
# USE_LOCAL_LLM = True  # Set to False to use Anthropic API
# LOCAL_LLM_MODEL = "llama3.1:8b"
# ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# MAX_GENERATION_TOKENS = 1000

# Logging
LOG_RETRIEVAL_SCORES = True
LOG_RERANKER_SCORES = True
LOG_GENERATION_TIME = True