
"""Incrementally synchronize the V2 document corpus."""

import json
from hashlib import sha256
from pathlib import Path

from technical_rag.config_v2 import (
    MODEL_ID,
    MODEL_REVISION,
    COLLECTION_NAME,
    QDRANT_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from technical_rag.embeddings import EmbeddingModel
from technical_rag.ingestion.embedding_reuse import EmbeddingSpec
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.ingestion.registry import DocumentRegistry
from technical_rag.ingestion.registry_sync import (
    sync_source_document,
    sync_deleted_document,
)
from technical_rag.vector_store import QdrantVectorStore


ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "data/sources.json"
REGISTRY_PATH = ROOT / "data/state/registry_v2.db"


def load_documents():
    sources = json.loads(
        SOURCES_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(sources, list) or not sources:
        raise ValueError("Source list must not be empty")

    documents = []

    for source in sources:
        document = load_markdown(
            file_path=str(ROOT / source["file_path"]),
            source_name=source["source_name"],
            source_url=source["source_url"],
            document_type=source["document_type"],
            metadata=source["metadata"],
        )

        documents.append(document)

    ids = [document.id for document in documents]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs")

    return documents


def compute_index_signature(document, spec):
    configuration = {
        "document": document.model_dump(
            mode="json",
            exclude={"content"},
        ),
        "embedding": spec.fingerprint(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "schema_version": 1,
    }

    serialized = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(serialized.encode("utf-8")).hexdigest()


def run_sync(
    documents,
    registry,
    store,
    embedder,
    spec,
):
    if not documents:
        raise ValueError("Cannot synchronize an empty corpus")

    if store.collection_name != COLLECTION_NAME:
        raise ValueError("Unexpected Qdrant collection")

    current_ids = {document.id for document in documents}

    if len(current_ids) != len(documents):
        raise ValueError("Duplicate document IDs")

    stats = {
        "updated": 0,
        "skipped": 0,
        "deleted": 0,
    }

    # 1. Synchronize new or changed documents
    for document in documents:
        signature = compute_index_signature(document, spec)

        if registry.get_index_signature(document.id) == signature:
            stats["skipped"] += 1
            continue

        report = sync_source_document(
            document=document,
            registry=registry,
            store=store,
            embedder=embedder,
            spec=spec,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            index_signature=signature,
        )

        stats["updated"] += 1

        print(
            f"Synced {document.title}: "
            f"{report.upserted} upserts, "
            f"{report.deleted} deletions"
        )

    # 2. Delete documents removed from the source list
    deleted_ids = registry.list_indexed_ids() - current_ids

    for document_id in sorted(deleted_ids):
        sync_deleted_document(
            document_id=document_id,
            registry=registry,
            store=store,
        )
        stats["deleted"] += 1

    return stats


def main():
    # Load and validate all sources before writing anything
    documents = load_documents()

    store = QdrantVectorStore(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    if not store.client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            "V2 collection not found. "
            "Initialize it with bootstrap_v2.py first."
        )

    registry = DocumentRegistry(REGISTRY_PATH)

    embedder = EmbeddingModel(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    spec = EmbeddingSpec(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        dimension=embedder.dimension,
    )

    stats = run_sync(
        documents,
        registry,
        store,
        embedder,
        spec,
    )

    print(f"Synchronization complete: {stats}")
    print("Restart the RAG API to rebuild BM25.")


if __name__ == "__main__":
    main()
