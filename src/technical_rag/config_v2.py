
"""Compatibility configuration for existing V2 scripts."""

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PROJECT_ROOT,
    QDRANT_URL,
    REGISTRY_PATH,
    SOURCES_PATH,
    V2_PROFILE,
)


MODEL_ID = V2_PROFILE.embedding_model_id
MODEL_REVISION = V2_PROFILE.embedding_revision
COLLECTION_NAME = V2_PROFILE.collection_name
