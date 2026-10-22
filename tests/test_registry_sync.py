
from unittest.mock import Mock

import numpy as np
import pytest

from technical_rag.models import Chunk, Document
from technical_rag.vector_store import StoredChunk
from technical_rag.ingestion import registry_sync
from technical_rag.ingestion.embedding_reuse import EmbeddingSpec
from technical_rag.ingestion.qdrant_sync import SyncReport
from technical_rag.ingestion.registry import DocumentRegistry


def make_document(content_hash="hash-v1"):
    return Document(
        id="doc1",
        content_hash=content_hash,
        title="Test document",
        content="Example content",
        source_name="Test",
        source_url="https://example.com",
        document_type="documentation",
        metadata={},
    )


def make_chunk():
    return Chunk(
        id="chunk-A",
        document_id="doc1",
        content="Example content",
        chunk_index=0,
        section_title=None,
        metadata={},
    )


def setup_sync(monkeypatch, existing_chunks=None):
    chunk = make_chunk()

    monkeypatch.setattr(
        registry_sync,
        "chunk_document",
        lambda **kwargs: [chunk],
    )

    store = Mock()
    store.collection_name = "technical_docs_v2"
    store.get_document_chunks.return_value = existing_chunks or {}

    embedder = Mock()
    embedder.dimension = 2
    embedder.embed_documents.return_value = np.array(
        [[0.6, 0.8]]
    )

    spec = EmbeddingSpec(
        model_id="test-model",
        revision="test-revision",
        dimension=2,
    )

    return store, embedder, spec


def test_success_updates_registry(tmp_path, monkeypatch):
    registry = DocumentRegistry(tmp_path / "registry.db")
    document = make_document()

    store, embedder, spec = setup_sync(monkeypatch)

    mock_sync = Mock(
        return_value=SyncReport(
            document_id="doc1",
            upserted=1,
            deleted=0,
            verified=1,
        )
    )
    monkeypatch.setattr(
        registry_sync, "sync_document", mock_sync
    )

    registry_sync.sync_source_document(
        document, registry, store, embedder, spec
    )

    assert registry.get_status(document) == "unchanged"
    mock_sync.assert_called_once()


def test_failure_preserves_old_registry_state(
    tmp_path, monkeypatch
):
    registry = DocumentRegistry(tmp_path / "registry.db")

    old_document = make_document("old-hash")
    updated_document = make_document("new-hash")

    registry.mark_indexed(old_document)

    store, embedder, spec = setup_sync(monkeypatch)

    monkeypatch.setattr(
        registry_sync,
        "sync_document",
        Mock(side_effect=RuntimeError("Qdrant failed")),
    )

    with pytest.raises(RuntimeError, match="Qdrant failed"):
        registry_sync.sync_source_document(
            updated_document,
            registry,
            store,
            embedder,
            spec,
        )

    assert registry.get_status(updated_document) == "modified"
    assert registry.get_status(old_document) == "unchanged"


def test_existing_verified_chunks_are_reused(
    tmp_path, monkeypatch
):
    registry = DocumentRegistry(tmp_path / "registry.db")
    document = make_document()

    chunk = make_chunk()
    spec = EmbeddingSpec(
        model_id="test-model",
        revision="test-revision",
        dimension=2,
    )

    stored = StoredChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        content=chunk.content,
        section_title=chunk.section_title,
        metadata={
            "_embedding_fingerprint": spec.fingerprint(),
        },
        vector=np.array([0.6, 0.8]),
        chunk_index=0,
    )

    store, embedder, spec = setup_sync(
        monkeypatch,
        existing_chunks={chunk.id: stored},
    )

    mock_sync = Mock(
        return_value=SyncReport(
            document_id="doc1",
            upserted=0,
            deleted=0,
            verified=1,
        )
    )
    monkeypatch.setattr(
        registry_sync, "sync_document", mock_sync
    )

    registry_sync.sync_source_document(
        document, registry, store, embedder, spec
    )

    prepared = mock_sync.call_args.kwargs["prepared"]

    assert prepared.skipped_ids == ["chunk-A"]
    assert prepared.computed_count == 0
    embedder.embed_documents.assert_not_called()
    assert registry.get_status(document) == "unchanged"


def test_successful_deletion_updates_registry(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.db")
    document = make_document()
    registry.mark_indexed(document)

    store = Mock()
    store.collection_name = "technical_docs_v2"
    store.get_document_payloads.side_effect = [
        {"chunk-A": {}},
        {},
    ]

    deleted = registry_sync.sync_deleted_document(
        document.id, registry, store
    )

    assert deleted == 1
    assert registry.get_status(document) == "new"
    store.delete_chunks.assert_called_once_with(
        chunk_ids={"chunk-A"},
        batch_size=100,
    )


def test_failed_deletion_keeps_registry(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.db")
    document = make_document()
    registry.mark_indexed(document)

    store = Mock()
    store.collection_name = "technical_docs_v2"
    store.get_document_payloads.side_effect = [
        {"chunk-A": {}},
        {"chunk-A": {}},
    ]

    with pytest.raises(RuntimeError):
        registry_sync.sync_deleted_document(
            document.id, registry, store
        )

    assert registry.get_status(document) == "unchanged"
