
"""Initialize the V2 Qdrant collection exactly once."""

import json
from pathlib import Path

from technical_rag.chunking import chunk_document
from technical_rag.embeddings import EmbeddingModel
from technical_rag.ingestion.embedding_reuse import EmbeddingSpec
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.vector_store import QdrantVectorStore


ROOT = Path(__file__).resolve().parents[1]

SOURCES_PATH = ROOT / "data/sources.json"

MODEL_ID = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MODEL_REVISION = "b207367332321f8e44f96e224ef15bc607f4dbf0"

COLLECTION_NAME = "technical_docs_v2"
QDRANT_URL = "http://localhost:6333"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_corpus():
    sources = json.loads(
        SOURCES_PATH.read_text(encoding="utf-8")
    )

    all_chunks = []

    for source in sources:
        document = load_markdown(
            file_path=str(ROOT / source["file_path"]),
            source_name=source["source_name"],
            source_url=source["source_url"],
            document_type=source["document_type"],
            metadata=source["metadata"],
        )

        chunks = chunk_document(
            document=document,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("Cannot index an empty corpus")

    if len({c.id for c in all_chunks}) != len(all_chunks):
        raise ValueError("Duplicate chunk IDs detected")

    return all_chunks


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
    chunks = load_corpus()
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
