
from unittest.mock import Mock

from technical_rag.models import SearchResult
from technical_rag.pipeline import RAGPipeline


def test_pipeline_returns_answer_with_citations():
    retriever = Mock()
    llm = Mock()

    retriever.search.return_value = [
        SearchResult(
            chunk_id="chunk1",
            document_id="doc1",
            content="Qdrant supports cosine similarity.",
            section_title="Metrics",
            score=0.9,
            metadata={
                "document_title": "Search",
                "source_name": "Qdrant",
                "source_url": "https://qdrant.tech",
            },
        )
    ]

    llm.generate.return_value = (
        "Qdrant supports cosine similarity. [S1]"
    )

    rag = RAGPipeline(
        retriever=retriever,
        llm=llm,
    )

    result = rag.ask("What metric does Qdrant support?")

    assert result.answer == (
        "Qdrant supports cosine similarity. [S1]"
    )
    assert result.valid_citations == ["S1"]
    assert result.invalid_citations == []
    assert result.uncited_statements == []
    assert len(result.evidence) == 1
    assert result.grounding == []
