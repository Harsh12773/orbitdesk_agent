import os
from pathlib import Path

# Paths to assignment files
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
RESOLVED_CASES_PATH = DATA_DIR / "resolved_cases.json"
OUTPUT_SCHEMA_PATH = DATA_DIR / "output_schema.json"

# Local AI Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_RETRIES = 1