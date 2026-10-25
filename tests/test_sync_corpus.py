
from unittest.mock import Mock, patch

from technical_rag.models import Document
from technical_rag.ingestion.registry import DocumentRegistry
from script.sync_corpus import (
    compute_index_signature,
    run_sync,
)


def make_document(metadata=None):
    return Document(
        id="doc1",
        content_hash="content-hash",
        title="Test",
        content="Example content",
        source_name="Test",
        source_url="https://example.com",
        document_type="documentation",
        metadata=metadata or {},
    )


def make_dependencies():
    store = Mock()
    store.collection_name = "technical_docs_v2"

    embedder = Mock()
    spec = Mock()
    spec.fingerprint.return_value = "embedding-v1"

    return store, embedder, spec


def test_unchanged_document_is_skipped(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.db")
    document = make_document()
    store, embedder, spec = make_dependencies()

    signature = compute_index_signature(document, spec)
    registry.mark_indexed(document, signature)

    with patch(
        "script.sync_corpus.sync_source_document"
    ) as sync:
        stats = run_sync(
            [document], registry, store, embedder, spec
        )

    sync.assert_not_called()
    assert stats["skipped"] == 1


def test_metadata_change_triggers_sync(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.db")
    original = make_document({"language": "en"})
    modified = make_document({"language": "fr"})

    store, embedder, spec = make_dependencies()

    registry.mark_indexed(
        original,
        compute_index_signature(original, spec),
    )

    with patch(
        "script.sync_corpus.sync_source_document"
    ) as sync:
        sync.return_value.upserted = 1
        sync.return_value.deleted = 0

        stats = run_sync(
            [modified], registry, store, embedder, spec
        )

    sync.assert_called_once()
    assert stats["updated"] == 1


def test_failure_stops_synchronization(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.db")
    document = make_document()
    store, embedder, spec = make_dependencies()

    with patch(
        "script.sync_corpus.sync_source_document",
        side_effect=RuntimeError("Qdrant failed"),
    ):
        try:
            run_sync(
                [document], registry, store, embedder, spec
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected synchronization failure")

    assert registry.get_status(document) == "new"
