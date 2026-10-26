from pathlib import Path

MODEL_ID = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# Use the same verified revision as your successful V2 bootstrap.
MODEL_REVISION = "b207367332321f8e44f96e224ef15bc607f4dbf0"

COLLECTION_NAME = "technical_docs_v2"
QDRANT_URL = "http://localhost:6333"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_PATH = (
    PROJECT_ROOT / "data" / "state" / "registry_v2.db"
)

SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"