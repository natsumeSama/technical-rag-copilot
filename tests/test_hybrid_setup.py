
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from technical_rag.models import Chunk
from technical_rag.retrieval.hybrid_store import (
    create_hybrid_collection,
)
from technical_rag.retrieval.sparse_embeddings import (
    BM25SparseEncoder,
)


def test_hybrid_collection_configuration():
    client = Mock()
    client.collection_exists.return_value = False

    create_hybrid_collection(
        client=client,
        collection_name="test_hybrid",
        dense_dimension=384,
    )

    config = client.create_collection.call_args.kwargs

    assert "dense" in config["vectors_config"]
    assert "bm25" in config["sparse_vectors_config"]

    client.create_payload_index.assert_called_once()


def test_existing_collection_is_protected():
    client = Mock()
    client.collection_exists.return_value = True

    with pytest.raises(RuntimeError):
        create_hybrid_collection(
            client, "test_hybrid", 384
        )

    client.create_collection.assert_not_called()


def test_sparse_encoding():
    fake_model = Mock()
    fake_model.embed.return_value = iter([
        SimpleNamespace(
            indices=np.array([10, 20]),
            values=np.array([1.5, 0.8]),
        )
    ])

    chunk = Chunk(
        id="A",
        document_id="doc1",
        content="Hybrid retrieval",
        chunk_index=0,
        section_title="Search",
        metadata={},
    )

    encoder = BM25SparseEncoder(model=fake_model)
    vectors = encoder.encode_chunks([chunk])

    assert vectors[0].indices == [10, 20]
    assert vectors[0].values == [1.5, 0.8]

    fake_model.embed.assert_called_once_with(
        ["Search\n\nHybrid retrieval"]
    )
