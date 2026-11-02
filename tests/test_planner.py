
from technical_rag.ingestion.planner import build_sync_plan
from technical_rag.ingestion.registry import DocumentRegistry
from technical_rag.models import Document


def make_document(doc_id, content_hash):
    return Document(
        id=doc_id,
        content_hash=content_hash,
        title=doc_id,
        content="Example",
        source_name="Test",
        source_url=f"https://example.com/{doc_id}",
        document_type="documentation",
        metadata={},
    )


def test_sync_plan(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.db")

    # Simulate previously indexed documents
    registry.mark_indexed(make_document("A", "v1"))
    registry.mark_indexed(make_document("B", "v1"))
    registry.mark_indexed(make_document("C", "v1"))

    current_documents = [
        make_document("A", "v1"),  # Unchanged
        make_document("B", "v2"),  # Modified
        make_document("D", "v1"),  # New
        # C was removed
    ]

    plan = build_sync_plan(current_documents, registry)

    assert [d.id for d in plan.unchanged] == ["A"]
    assert [d.id for d in plan.modified] == ["B"]
    assert [d.id for d in plan.new] == ["D"]
    assert plan.deleted_ids == {"C"}

    # Planning must not modify the registry
    assert registry.get_status(current_documents[1]) == "modified"
