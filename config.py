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

# Evaluation Settings
EVAL_BATCH_SIZE = 5
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCE_THRESHOLD = 0.7

# Logging
LOG_RETRIEVAL_SCORES = True
LOG_RERANKER_SCORES = True
LOG_GENERATION_TIME = True