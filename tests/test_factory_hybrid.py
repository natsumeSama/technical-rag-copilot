
from unittest.mock import Mock

import pytest

from technical_rag import factory
from technical_rag.config import HYBRID_PROFILE
from technical_rag.retrieval.qdrant_hybrid import (
    QdrantHybridRetriever,
)


def test_factory_selects_native_hybrid(monkeypatch):
    monkeypatch.setenv(
        "RAG_COLLECTION",
        HYBRID_PROFILE.collection_name,
    )

    client = Mock()
    client.collection_exists.return_value = True
    client.count.return_value.count = 10

    monkeypatch.setattr(
        factory, "QdrantClient", Mock(return_value=client)
    )
    monkeypatch.setattr(
        factory, "EmbeddingModel", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        factory, "BM25SparseEncoder", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        factory, "OllamaLLM", Mock(return_value=Mock())
    )

    # Native hybrid must not rebuild the local BM25 corpus.
    monkeypatch.setattr(
        factory,
        "load_chunks",
        Mock(side_effect=AssertionError("Unexpected loading")),
    )

    rag = factory.build_pipeline(enable_grounding=False)

    assert isinstance(
        rag.retriever,
        QdrantHybridRetriever,
    )
    assert rag.retriever.client is client


def test_factory_rejects_missing_hybrid_collection(monkeypatch):
    monkeypatch.setenv(
        "RAG_COLLECTION",
        HYBRID_PROFILE.collection_name,
    )

    client = Mock()
    client.collection_exists.return_value = False

    monkeypatch.setattr(
        factory, "QdrantClient", Mock(return_value=client)
    )

    with pytest.raises(
        RuntimeError,
        match="Hybrid collection does not exist",
    ):
        factory.build_pipeline(enable_grounding=False)
