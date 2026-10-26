
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from qdrant_client.models import SparseVector

from technical_rag.models import Chunk
from technical_rag.vector_store import StoredChunk
from technical_rag.ingestion.embedding_reuse import EmbeddingSpec
from technical_rag.retrieval.hybrid_index import (
    build_hybrid_points,
    index_hybrid_points,
)


def make_chunk(chunk_id, text, index):
    return Chunk(
        id=chunk_id,
        document_id="doc1",
        content=text,
        chunk_index=index,
        section_title="Search",
        metadata={},
    )


def test_reuse_dense_and_generate_sparse():
    spec = EmbeddingSpec(
        model_id="test-model",
        revision="test-revision",
        dimension=2,
    )

    old_chunk = make_chunk("A", "Dense retrieval", 0)
    new_chunk = make_chunk("B", "New retrieval", 1)

    stored = StoredChunk(
        chunk_id="A",
        document_id="doc1",
        content="Dense retrieval",
        section_title="Search",
        metadata={
            "_embedding_fingerprint": spec.fingerprint()
        },
        vector=np.array([1.0, 0.0]),
        chunk_index=0,
    )

    dense_store = Mock()
    dense_store.get_document_chunks.return_value = {
        "A": stored
    }

    dense_model = Mock()
    dense_model.dimension = 2
    dense_model.embed_documents.return_value = np.array(
        [[0.0, 1.0]]
    )

    sparse_encoder = Mock()
    sparse_encoder.encode_chunks.return_value = [
        SparseVector(indices=[10], values=[1.5]),
        SparseVector(indices=[20], values=[2.0]),
    ]

    result = build_hybrid_points(
        chunks=[old_chunk, new_chunk],
        dense_store=dense_store,
        dense_model=dense_model,
        sparse_encoder=sparse_encoder,
        spec=spec,
    )

    assert result.reused == 1
    assert result.computed == 1

    dense_model.embed_documents.assert_called_once_with(
        [new_chunk]
    )

    assert result.points[0].vector["dense"] == [1.0, 0.0]
    assert result.points[0].vector["bm25"].indices == [10]
    assert result.points[1].vector["dense"] == [0.0, 1.0]

    assert result.points[0].payload["chunk_id"] == "A"
    assert result.points[1].payload["chunk_index"] == 1


def test_upload_and_verify_points():
    client = Mock()
    client.collection_exists.return_value = True

    point = SimpleNamespace(
        id="point-A",
        vector={"dense": [1.0, 0.0], "bm25": object()},
        payload={"chunk_id": "A"},
    )

    client.scroll.side_effect = [
        ([], None),
        ([point], None),
    ]

    count = index_hybrid_points(
        client=client,
        collection_name="test_hybrid",
        points=[point],
    )

    assert count == 1
    client.upsert.assert_called_once()
    assert client.upsert.call_args.kwargs["wait"] is True


def test_unexpected_points_are_rejected():
    client = Mock()
    client.collection_exists.return_value = True

    client.scroll.return_value = (
        [SimpleNamespace(id="unexpected")],
        None,
    )

    point = SimpleNamespace(
        id="expected",
        payload={},
    )

    with pytest.raises(RuntimeError):
        index_hybrid_points(
            client, "test_hybrid", [point]
        )

    client.upsert.assert_not_called()
