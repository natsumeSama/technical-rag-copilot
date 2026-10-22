
"""Coordinate Qdrant synchronization and SQLite state."""

from ..chunking import chunk_document
from ..models import Chunk, Document
from ..vector_store import QdrantVectorStore

from .chunk_delta import plan_chunk_delta
from .embedding_reuse import (
    EmbeddingSpec,
    prepare_embeddings,
)
from .qdrant_sync import SyncReport, sync_document
from .registry import DocumentRegistry


def sync_source_document(
    document: Document,
    registry: DocumentRegistry,
    store: QdrantVectorStore,
    embedder,
    spec: EmbeddingSpec,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    batch_size: int = 100,
) -> SyncReport:

    # 1. Generate the current document chunks
    new_chunks = chunk_document(
        document=document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 2. Retrieve the previously indexed chunks
    stored = store.get_document_chunks(document.id)

    old_chunks = []

    for point in stored.values():
        if (
            point.document_id != document.id
            or not isinstance(point.chunk_index, int)
            or point.chunk_index < 0
        ):
            raise ValueError(
                "Invalid stored chunk position or document ID"
            )

        old_chunks.append(
            Chunk(
                id=point.chunk_id,
                document_id=point.document_id,
                content=point.content,
                chunk_index=point.chunk_index,
                section_title=point.section_title,
                metadata={
                    key: value
                    for key, value in point.metadata.items()
                    if key not in {
                        "_embedding_fingerprint",
                        "chunk_index",
                    }
                },
            )
        )

    # 3. Identify changed chunks
    delta = plan_chunk_delta(
        old_chunks=old_chunks,
        new_chunks=new_chunks,
    )

    # 4. Prepare reusable and new embeddings
    prepared = prepare_embeddings(
        delta=delta,
        stored=stored,
        embedder=embedder,
        spec=spec,
    )

    # 5. Synchronize and verify Qdrant
    report = sync_document(
        store=store,
        document_id=document.id,
        expected_chunks=new_chunks,
        prepared=prepared,
        embedding_fingerprint=spec.fingerprint(),
        batch_size=batch_size,
    )

    # 6. Record the successful version in SQLite
    registry.mark_indexed(document)

    return report


def sync_deleted_document(
    document_id: str,
    registry: DocumentRegistry,
    store: QdrantVectorStore,
    batch_size: int = 100,
) -> int:

    if store.collection_name != "technical_docs_v2":
        raise ValueError("Only V2 can be synchronized")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # 1. Identify the document's existing Qdrant points
    before = store.get_document_payloads(document_id)
    chunk_ids = set(before)

    # 2. Delete them
    if chunk_ids:
        store.delete_chunks(
            chunk_ids=chunk_ids,
            batch_size=batch_size,
        )

    # 3. Verify that no points remain
    remaining = store.get_document_payloads(document_id)

    if remaining:
        raise RuntimeError(
            f"Document {document_id} still has "
            f"{len(remaining)} Qdrant chunks"
        )

    # 4. Remove the document from SQLite
    registry.remove_indexed(document_id)

    return len(chunk_ids)
