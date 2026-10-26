
"""Populate the experimental Qdrant hybrid collection."""

import json
from pathlib import Path

from qdrant_client import QdrantClient

from technical_rag.chunking import chunk_document
from technical_rag.config_v2 import (
    MODEL_ID,
    MODEL_REVISION,
    COLLECTION_NAME,
    QDRANT_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PROJECT_ROOT,
    SOURCES_PATH,
)
from technical_rag.ingestion.corpus import load_chunks
from technical_rag.embeddings import EmbeddingModel
from technical_rag.ingestion.embedding_reuse import EmbeddingSpec
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.retrieval.hybrid_index import (
    build_hybrid_points,
    index_hybrid_points,
)
from technical_rag.retrieval.sparse_embeddings import BM25SparseEncoder
from technical_rag.vector_store import QdrantVectorStore


HYBRID_COLLECTION = "technical_docs_hybrid_exp"



def main():
    # 1. Connect to both collections
    dense_store = QdrantVectorStore(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    client = QdrantClient(url=QDRANT_URL)

    if not dense_store.client.collection_exists(COLLECTION_NAME):
        raise RuntimeError("V2 dense collection is missing")

    if not client.collection_exists(HYBRID_COLLECTION):
        raise RuntimeError(
            "Run bootstrap_hybrid.py first"
        )

    # 2. Load documents and models
    chunks = load_chunks(
        sources_path=SOURCES_PATH,
        project_root=PROJECT_ROOT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    dense_model = EmbeddingModel(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    spec = EmbeddingSpec(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        dimension=dense_model.dimension,
    )

    sparse_encoder = BM25SparseEncoder()

    # 3. Generate or reuse both vector types
    result = build_hybrid_points(
        chunks=chunks,
        dense_store=dense_store,
        dense_model=dense_model,
        sparse_encoder=sparse_encoder,
        spec=spec,
    )

    print(f"Chunks: {len(chunks)}")
    print(f"Dense reused: {result.reused}")
    print(f"Dense calculated: {result.computed}")

    # 4. Store and verify the hybrid vectors
    count = index_hybrid_points(
        client=client,
        collection_name=HYBRID_COLLECTION,
        points=result.points,
    )

    print(f"Verified {count} hybrid points")


if __name__ == "__main__":
    main()
