
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from technical_rag.api import app
from technical_rag.models import Evidence, RAGResult


@pytest.fixture
def client(monkeypatch):
    # Fake RAG: no Qdrant or Ollama required
    fake_rag = Mock()

    fake_rag.ask.return_value = RAGResult(
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

    # Replace the real pipeline factory during tests
    monkeypatch.setattr(
        "technical_rag.api.build_pipeline",
        lambda enable_grounding=True: fake_rag,
    )

    with TestClient(app) as test_client:
        yield test_client, fake_rag


def test_health(client):
    test_client, _ = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_success(client):
    test_client, fake_rag = client

    response = test_client.post(
        "/ask",
        json={"question": "Which metric does Qdrant support?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "Qdrant supports cosine similarity. [S1]"
    )
    assert data["valid_citations"] == ["S1"]
    assert data["invalid_citations"] == []
    assert len(data["evidence"]) == 1
    assert "timing_seconds" in data

    fake_rag.ask.assert_called_once_with(
        "Which metric does Qdrant support?"
    )


def test_ask_empty_question(client):
    test_client, fake_rag = client

    response = test_client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422
    fake_rag.ask.assert_not_called()


def test_ask_whitespace_question(client):
    test_client, fake_rag = client

    response = test_client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422
    fake_rag.ask.assert_not_called()


def test_ask_missing_question(client):
    test_client, fake_rag = client

    response = test_client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422
    fake_rag.ask.assert_not_called()
