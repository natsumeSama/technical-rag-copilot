
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from technical_rag.vector_store import QdrantVectorStore


def test_get_document_chunks():
    store = QdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test_collection",
    )

    store.client = Mock()

    store.client.scroll.side_effect = [
        (
            [
                SimpleNamespace(
                    payload={
                        "chunk_id": "A",
                        "document_id": "doc1",
                        "content": "Dense retrieval",
                        "section_title": "Search",
                    },
                    vector=[0.1, 0.2],
                )
            ],
            "next-page",
        ),
        (
            [
                SimpleNamespace(
                    payload={
                        "chunk_id": "B",
                        "document_id": "doc1",
                        "content": "Sparse retrieval",
                        "section_title": "Search",
                    },
                    vector=[0.3, 0.4],
                )
            ],
            None,
        ),
    ]

    chunks = store.get_document_chunks("doc1")

    assert set(chunks) == {"A", "B"}
    assert chunks["A"].content == "Dense retrieval"

    np.testing.assert_allclose(
        chunks["A"].vector,
        [0.1, 0.2],
    )

    assert store.client.scroll.call_count == 2

    assert (
        store.client.scroll.call_args_list[1]
        .kwargs["offset"] == "next-page"
    )
