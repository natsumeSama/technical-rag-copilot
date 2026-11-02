
from technical_rag.ingestion.chunk_delta import plan_chunk_delta
from technical_rag.models import Chunk


def make_chunk(chunk_id, index, content):
    return Chunk(
        id=chunk_id,
        document_id="doc1",
        content=content,
        chunk_index=index,
        section_title="Search",
        metadata={},
    )


def test_chunk_delta():
    old_chunks = [
        make_chunk("A", 0, "Introduction"),
        make_chunk("B", 1, "Dense retrieval"),
        make_chunk("C", 2, "Old content"),
    ]

    new_chunks = [
        make_chunk("A", 0, "Introduction"),
        make_chunk("B2", 2, "Dense retrieval"),
        make_chunk("D", 3, "New content"),
    ]

    delta = plan_chunk_delta(old_chunks, new_chunks)

    assert [c.id for c in delta.unchanged] == ["A"]
    assert [c.id for c in delta.to_upsert] == ["B2", "D"]
    assert [c.id for c in delta.to_embed] == ["D"]
    assert delta.reuse_from == {"B2": "B"}
    assert delta.to_delete == {"B", "C"}
