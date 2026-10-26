
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from qdrant_client import models

from technical_rag.retrieval.qdrant_hybrid import (
    QdrantHybridRetriever,
)


def make_retriever():
    client = Mock()

    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                score=0.04,
                payload={
                    "chunk_id": "A",
                    "document_id": "doc1",
                    "content": "Cosine similarity",
                    "section_title": "Metrics",
                    "source_name": "Qdrant",
                },
            )
        ]
    )

    dense_model = Mock()
    dense_model.embed_query.return_value = np.array(
        [0.1, 0.2]
    )

    sparse_encoder = Mock()
    sparse_encoder.encode_query.return_value = (
        models.SparseVector(
            indices=[10, 20],
            values=[1.5, 0.8],
        )
    )

    retriever = QdrantHybridRetriever(
        client=client,
        collection_name="test_hybrid",
        dense_model=dense_model,
        sparse_encoder=sparse_encoder,
    )

    return retriever, client, dense_model, sparse_encoder


def test_hybrid_search():
    retriever, client, dense, sparse = make_retriever()

    results = retriever.search(
        "similarity metrics",
        mode="hybrid",
    )

    kwargs = client.query_points.call_args.kwargs

    assert len(kwargs["prefetch"]) == 2
    assert kwargs["prefetch"][0].using == "dense"
    assert kwargs["prefetch"][1].using == "bm25"

    assert kwargs["query"].fusion == models.Fusion.RRF

    assert results[0].section_title == "Metrics"
    assert results[0].content == "Cosine similarity"

    dense.embed_query.assert_called_once()
    sparse.encode_query.assert_called_once()


def test_dense_search():
    retriever, client, dense, sparse = make_retriever()

    retriever.search("similarity metrics", mode="dense")

    kwargs = client.query_points.call_args.kwargs

    assert kwargs["using"] == "dense"
    assert kwargs["query"] == [0.1, 0.2]

    sparse.encode_query.assert_not_called()


def test_sparse_search():
    retriever, client, dense, sparse = make_retriever()

    retriever.search("similarity metrics", mode="sparse")

    kwargs = client.query_points.call_args.kwargs

    assert kwargs["using"] == "bm25"
    assert kwargs["query"].indices == [10, 20]

    dense.embed_query.assert_not_called()
