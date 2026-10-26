
"""Initialize the V2 Qdrant collection exactly once."""

import json
from pathlib import Path

from technical_rag.chunking import chunk_document
from technical_rag.embeddings import EmbeddingModel
from technical_rag.ingestion.embedding_reuse import EmbeddingSpec
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.vector_store import QdrantVectorStore
from technical_rag.config_v2 import (
    PROJECT_ROOT,
    SOURCES_PATH,
    MODEL_ID,
    MODEL_REVISION,
    COLLECTION_NAME,
    QDRANT_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from technical_rag.ingestion.corpus import load_chunks

def main():
    # 1. Protect any existing collection
    store = QdrantVectorStore(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    if store.client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"{COLLECTION_NAME} already exists. "
            "Bootstrap will not overwrite it."
        )

    # 2. Load the source documents
    chunks = load_chunks(
        sources_path=SOURCES_PATH,
        project_root=PROJECT_ROOT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"Loaded {len(chunks)} chunks")

    # 3. Load a pinned embedding model
    embedder = EmbeddingModel(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    spec = EmbeddingSpec(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        dimension=embedder.dimension,
        normalize=True,
        input_format="section-title-v1",
        encode_method="encode_document",
    )

    # 4. Generate the initial embeddings
    vectors = embedder.embed_documents(chunks)

    # 5. Create the protected V2 collection
    store.create_collection(
        vector_size=embedder.dimension,
        fail_if_exists=True,
    )

    # 6. Store chunks, vectors, and fingerprints
    store.upsert_chunks(
        chunks=chunks,
        embeddings=vectors,
        embedding_fingerprint=spec.fingerprint(),
    )

    # 7. Verify that the points were stored
    count = store.client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    if count != len(chunks):
        raise RuntimeError(
            f"Expected {len(chunks)} points, found {count}"
        )

    points, _ = store.client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if (
        not points
        or points[0].payload.get("_embedding_fingerprint")
        != spec.fingerprint()
        or "chunk_index" not in points[0].payload
    ):
        raise RuntimeError("Stored payload validation failed")

    print(f"Verified {count} points in {COLLECTION_NAME}")
    print(f"Embedding fingerprint: {spec.fingerprint()}")


if __name__ == "__main__":
    main()
