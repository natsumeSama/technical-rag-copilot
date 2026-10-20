
from technical_rag.ingestion.registry import DocumentRegistry
from technical_rag.models import Document


def test_document_registry(tmp_path):
    db_path = tmp_path / "registry.db"
    registry = DocumentRegistry(db_path)

    document = Document(
        id="doc-1",
        content_hash="hash-v1",
        title="Test document",
        content="Original content",
        source_name="Test",
        source_url="https://example.com/doc",
        document_type="documentation",
        metadata={},
    )

    # First time: new document
    assert registry.get_status(document) == "new"

    # Simulate successful indexing
    registry.mark_indexed(document)
    assert registry.get_status(document) == "unchanged"

    # Modify the document
    updated = document.model_copy(
        update={"content_hash": "hash-v2"}
    )
    assert registry.get_status(updated) == "modified"

    # Verify persistence after reopening the database
    reopened = DocumentRegistry(db_path)
    assert reopened.get_status(document) == "unchanged"
    assert reopened.list_indexed_ids() == {"doc-1"}

    # Simulate an empty source manifest
    deleted_ids = reopened.list_indexed_ids() - set()
    assert deleted_ids == {"doc-1"}