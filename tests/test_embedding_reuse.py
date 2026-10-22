
from unittest.mock import Mock

import numpy as np

from technical_rag.models import Chunk
from technical_rag.vector_store import StoredChunk
from technical_rag.ingestion.chunk_delta import plan_chunk_delta
from technical_rag.ingestion.embedding_reuse import (
    EmbeddingSpec,
    prepare_embeddings,
)


def make_chunk(chunk_id, index, text):
    return Chunk(
        id=chunk_id,
        document_id="doc1",
        content=text,
        chunk_index=index,
        section_title="Search",
        metadata={},
    )


def make_stored(chunk, fingerprint):
    return StoredChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        content=chunk.content,
        section_title=chunk.section_title,
        metadata={"_embedding_fingerprint": fingerprint},
        vector=np.array([1.0, 0.0, 0.0]),
    )


def test_embedding_reuse():
    spec = EmbeddingSpec(
        model_id="test-model",
        revision="test-revision",
        dimension=3,
    )

    old = [
        make_chunk("A", 0, "Introduction"),
        make_chunk("B", 1, "Dense retrieval"),
        make_chunk("C", 2, "Old content"),
    ]

    new = [
        make_chunk("A", 0, "Introduction"),
        make_chunk("B2", 2, "Dense retrieval"),
        make_chunk("D", 3, "New content"),
    ]

    delta = plan_chunk_delta(old, new)

    stored = {
        chunk.id: make_stored(
            chunk, spec.fingerprint()
        )
        for chunk in old
    }

    embedder = Mock()
    embedder.dimension = 3
    embedder.embed_documents.return_value = np.array(
        [[0.0, 0.0, 1.0]]
    )

    result = prepare_embeddings(
        delta, stored, embedder, spec
    )

    assert result.skipped_ids == ["A"]
    assert [c.id for c in result.chunks] == ["B2", "D"]
    assert result.delete_ids == {"B", "C"}

    assert result.reused_count == 1
    assert result.computed_count == 1

    np.testing.assert_allclose(
        result.vectors,
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    embedder.embed_documents.assert_called_once_with(
        [new[2]]
    )


def test_missing_fingerprint_forces_reembedding():
    chunk = make_chunk("A", 0, "Introduction")
    spec = EmbeddingSpec(
        model_id="test-model",
        revision="test-revision",
        dimension=3,
    )

    delta = plan_chunk_delta([chunk], [chunk])

    old_point = make_stored(chunk, spec.fingerprint())
    old_point.metadata.clear()

    embedder = Mock()
    embedder.dimension = 3
    embedder.embed_documents.return_value = np.array(
        [[0.0, 1.0, 0.0]]
    )

    result = prepare_embeddings(
        delta, {"A": old_point}, embedder, spec
    )

    assert result.skipped_ids == []
    assert [c.id for c in result.chunks] == ["A"]
    assert result.reused_count == 0
    assert result.computed_count == 1
