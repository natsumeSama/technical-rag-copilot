
"""Shared configuration for the Technical RAG Copilot."""

import os
from dataclasses import dataclass
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"
REGISTRY_PATH = PROJECT_ROOT / "data" / "state" / "registry_v2.db"


# Infrastructure
QDRANT_URL = "http://localhost:6333"


# Document processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# Models
DENSE_MODEL_ID = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
V2_MODEL_REVISION = "b207367332321f8e44f96e224ef15bc607f4dbf0"

SPARSE_MODEL_ID = "Qdrant/bm25"
NLI_MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
LLM_MODEL_ID = "qwen3:1.7b"


@dataclass(frozen=True)
class CollectionProfile:
    collection_name: str
    embedding_model_id: str
    embedding_revision: str | None


# Our three collection configurations
V1_PROFILE = CollectionProfile(
    collection_name="technical_docs_v1",
    embedding_model_id=DENSE_MODEL_ID,
    embedding_revision=None,
)

V2_PROFILE = CollectionProfile(
    collection_name="technical_docs_v2",
    embedding_model_id=DENSE_MODEL_ID,
    embedding_revision=V2_MODEL_REVISION,
)

HYBRID_PROFILE = CollectionProfile(
    collection_name="technical_docs_hybrid_exp",
    embedding_model_id=DENSE_MODEL_ID,
    embedding_revision=V2_MODEL_REVISION,
)


def get_pipeline_profile() -> CollectionProfile:
    """Select the collection supported by the current RAG pipeline."""

    collection_name = os.getenv(
        "RAG_COLLECTION",
        V1_PROFILE.collection_name,
    )

    if collection_name == V1_PROFILE.collection_name:
        return V1_PROFILE

    if collection_name == V2_PROFILE.collection_name:
        return V2_PROFILE

    raise ValueError(
        f"Unsupported RAG_COLLECTION: {collection_name!r}. "
        "The current RAG pipeline supports V1 and V2. "
        "The hybrid collection is still experimental."
    )
