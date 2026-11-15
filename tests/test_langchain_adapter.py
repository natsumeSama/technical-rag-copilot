from langchain_core.documents import Document

from technical_rag.context import ContextBuilder
from technical_rag.retrieval.langchain_adapter import (
    to_search_results,
)


def test_langchain_results_preserve_citation_metadata():
    document = Document(
        page_content="Metrics\n\nQdrant supports cosine similarity.",
        metadata={
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "original_content": "Qdrant supports cosine similarity.",
            "section_title": "Metrics",
            "document_title": "Search",
            "source_name": "Qdrant",
            "source_url": "https://qdrant.tech",
        },
    )

    results = to_search_results([(document, 0.85)])

    evidence = ContextBuilder().build_evidence(results)

    assert len(evidence) == 1
    assert evidence[0].citation_id == "S1"
    assert evidence[0].chunk_id == "chunk-1"
    assert evidence[0].content == "Qdrant supports cosine similarity."
    assert evidence[0].source_url == "https://qdrant.tech"
    assert evidence[0].retrieval_score == 0.85
