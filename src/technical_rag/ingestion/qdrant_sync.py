
from dataclasses import dataclass

import numpy as np

from ..models import Chunk
from ..vector_store import QdrantVectorStore
from .embedding_reuse import PreparedEmbeddings


@dataclass
class SyncReport:
    document_id: str
    upserted: int
    deleted: int
    verified: int


def sync_document(
    store: QdrantVectorStore,
    document_id: str,
    expected_chunks: list[Chunk],
    prepared: PreparedEmbeddings,
    embedding_fingerprint: str,
    batch_size: int = 100,
) -> SyncReport:

    # 1. Validate the synchronization plan
    if store.collection_name != "technical_docs_v2":
        raise ValueError("Only the V2 collection can be synchronized")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not embedding_fingerprint:
        raise ValueError("Missing embedding fingerprint")

    expected_ids = [chunk.id for chunk in expected_chunks]
    upsert_ids = [chunk.id for chunk in prepared.chunks]

    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate expected chunk IDs")

    if len(set(upsert_ids)) != len(upsert_ids):
        raise ValueError("Duplicate upsert chunk IDs")

    if any(c.document_id != document_id for c in expected_chunks):
        raise ValueError("Unexpected document in expected chunks")

    if any(c.document_id != document_id for c in prepared.chunks):
        raise ValueError("Unexpected document in upserts")

    if set(expected_ids) != (
        set(upsert_ids) | set(prepared.skipped_ids)
    ):
        raise ValueError("Incomplete synchronization plan")

    if set(upsert_ids) & set(prepared.skipped_ids):
        raise ValueError("Chunk cannot be both skipped and upserted")

    if set(expected_ids) & prepared.delete_ids:
        raise ValueError("Cannot delete an expected chunk")

    if (
        prepared.vectors.ndim != 2
        or len(prepared.vectors) != len(prepared.chunks)
        or prepared.vectors.shape[1] == 0
        or not np.isfinite(prepared.vectors).all()
    ):
        raise ValueError("Invalid prepared embeddings")

    # 2. Insert or update chunks in batches
    for start in range(0, len(prepared.chunks), batch_size):
        end = start + batch_size

        store.upsert_chunks(
            chunks=prepared.chunks[start:end],
            embeddings=prepared.vectors[start:end],
            embedding_fingerprint=embedding_fingerprint,
        )

    # 3. Delete obsolete points
    store.delete_chunks(
        chunk_ids=prepared.delete_ids,
        batch_size=batch_size,
    )

    # 4. Read the final document state
    actual = store.get_document_payloads(document_id)

    if set(actual) != set(expected_ids):
        raise RuntimeError(
            "Qdrant chunk IDs do not match the expected document"
        )

    # 5. Verify each stored chunk
    for chunk in expected_chunks:
        expected_payload = {
            **chunk.metadata,
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "section_title": chunk.section_title,
            "chunk_index": chunk.chunk_index,
            "_embedding_fingerprint": embedding_fingerprint,
        }

        if actual[chunk.id] != expected_payload:
            raise RuntimeError(
                f"Qdrant payload mismatch: {chunk.id}"
            )

    # 6. Report successful synchronization
    return SyncReport(
        document_id=document_id,
        upserted=len(prepared.chunks),
        deleted=len(prepared.delete_ids),
        verified=len(expected_chunks),
    )
