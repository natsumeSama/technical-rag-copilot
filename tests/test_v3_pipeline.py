"""Unit tests for the V3 RAG pipeline."""

from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from technical_rag.rag.v3_pipeline import ask_v3
from technical_rag.models import CitationSupportResult

import pytest


def test_v3_pipeline_returns_grounded_answer():
    question = "Which similarity metric does Qdrant support?"

    # Simulate a result returned by LangChain retrieval
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

    store = Mock()
    store.similarity_search_with_score.return_value = [(document, 0.9)]

    # Simulate the LLM response
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content="Qdrant supports cosine similarity. [S1]"
    )

    # Execute our real pipeline logic
    result = ask_v3(
        question=question,
        store=store,
        llm=llm,
    )

    # Verify the output
    assert result.question == question
    assert result.valid_citations == ["S1"]
    assert result.invalid_citations == []
    assert result.uncited_statements == []
    assert len(result.evidence) == 1

    assert result.evidence[0].chunk_id == "chunk-1"
    assert result.evidence[0].source_url == "https://qdrant.tech"

    # NLI grounding is not integrated yet
    assert result.grounding == []

    # Verify that the components were actually called
    store.similarity_search_with_score.assert_called_once_with(question, k=10)

    llm.invoke.assert_called_once()


@pytest.mark.parametrize("question", ["", "   "])
def test_v3_rejects_empty_questions(question):
    store = Mock()
    llm = Mock()

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        ask_v3(question, store, llm)

    store.similarity_search_with_score.assert_not_called()
    llm.invoke.assert_not_called()


def test_v3_handles_missing_evidence():
    store = Mock()
    llm = Mock()

    store.similarity_search_with_score.return_value = []

    with pytest.raises(
        RuntimeError,
        match="No evidence retrieved",
    ):
        ask_v3("What is Qdrant?", store, llm)

    llm.invoke.assert_not_called()


def test_v3_handles_empty_llm_response():
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

    store = Mock()
    store.similarity_search_with_score.return_value = [(document, 0.9)]

    llm = Mock()
    llm.invoke.return_value = AIMessage(content="")

    with pytest.raises(
        RuntimeError,
        match="LLM returned an empty answer",
    ):
        ask_v3("What metric does Qdrant support?", store, llm)


def test_v3_runs_optional_grounding():
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

    store = Mock()
    store.similarity_search_with_score.return_value = [(document, 0.9)]

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content="Qdrant supports cosine similarity. [S1]"
    )

    expected = CitationSupportResult(
        claim="Qdrant supports cosine similarity.",
        citation_id="S1",
        label="entailment",
        supported=True,
        entailment_score=0.95,
        neutral_score=0.04,
        contradiction_score=0.01,
    )

    grounding_validator = Mock()
    grounding_validator.validate.return_value = [expected]

    result = ask_v3(
        question="What metric does Qdrant support?",
        store=store,
        llm=llm,
        grounding_validator=grounding_validator,
    )

    grounding_validator.validate.assert_called_once_with(
        answer=result.answer,
        evidence=result.evidence,
    )

    assert result.grounding == [expected]
    assert result.valid_citations == ["S1"]
