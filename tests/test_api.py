"""Tests for the V3 FastAPI application."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from technical_rag.api import app
from technical_rag.models import Evidence, RAGResult


@pytest.fixture
def client(monkeypatch):
    """Mock V3 components without starting Qdrant or Ollama."""

    fake_store = Mock()
    fake_llm = Mock()

    fake_result = RAGResult(
        question="Which metric does Qdrant support?",
        answer="Qdrant supports cosine similarity. [S1]",
        evidence=[
            Evidence(
                citation_id="S1",
                chunk_id="chunk1",
                content="Qdrant supports cosine similarity.",
                document_title="Qdrant Search",
                section_title="Metrics",
                source_name="Qdrant",
                source_url="https://qdrant.tech",
                retrieval_score=0.9,
            )
        ],
        valid_citations=["S1"],
        invalid_citations=[],
        uncited_statements=[],
        grounding=[],
        timing_seconds={
            "retrieval": 0.01,
            "generation": 0.02,
            "validation": 0.01,
            "total": 0.04,
        },
    )

    fake_ask = Mock(return_value=fake_result)

    # Replace real V3 components during startup
    monkeypatch.setattr(
        "technical_rag.api.build_v3_store",
        lambda: fake_store,
    )

    monkeypatch.setattr(
        "technical_rag.api.build_v3_llm",
        lambda: fake_llm,
    )

    monkeypatch.setattr(
        "technical_rag.api.ask_v3",
        fake_ask,
    )

    with TestClient(app) as test_client:
        yield test_client, fake_ask, fake_store, fake_llm


def test_health(client):
    test_client, _, _, _ = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_success(client):
    test_client, fake_ask, fake_store, fake_llm = client

    response = test_client.post(
        "/ask",
        json={"question": "Which metric does Qdrant support?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == ("Qdrant supports cosine similarity. [S1]")
    assert data["valid_citations"] == ["S1"]
    assert data["invalid_citations"] == []
    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["citation_id"] == "S1"
    assert "timing_seconds" in data
    assert data["grounding"] == []

    fake_ask.assert_called_once_with(
        question="Which metric does Qdrant support?",
        store=fake_store,
        llm=fake_llm,
    )


def test_ask_empty_question(client):
    test_client, fake_ask, _, _ = client

    response = test_client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422
    fake_ask.assert_not_called()


def test_ask_whitespace_question(client):
    test_client, fake_ask, _, _ = client

    response = test_client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422
    fake_ask.assert_not_called()


def test_ask_missing_question(client):
    test_client, fake_ask, _, _ = client

    response = test_client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422
    fake_ask.assert_not_called()


def test_homepage(client):
    test_client, _, _, _ = client

    response = test_client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Technical RAG Copilot" in response.text
