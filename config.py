# config.py
import os
import dotenv
from pathlib import Path

dotenv.load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
WEB_FILE = DATA_DIR / "web.txt"
EVAL_DIR = DATA_DIR / "eval"

# Vector Store Settings
VECTOR_STORE_TYPE = "qdrant"
COLLECTION_NAME = "rag_docs"
QDRANT_URL = os.getenv("URL")
QDRANT_API_KEY = os.getenv("API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Generation Settings
# USE_LOCAL_LLM = True  # Set to False to use Anthropic API
# LOCAL_LLM_MODEL = "llama3.1:8b"
# ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# MAX_GENERATION_TOKENS = 1000

# Evaluation Settings
EVAL_BATCH_SIZE = 5
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCE_THRESHOLD = 0.7

# Logging
LOG_RETRIEVAL_SCORES = True
LOG_RERANKER_SCORES = True
LOG_GENERATION_TIME = True