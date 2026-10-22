
from unittest.mock import Mock

import numpy as np
import pytest

from technical_rag.models import Chunk
from technical_rag.vector_store import QdrantVectorStore


def test_embedding_fingerprint_is_stored():
    store = QdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test_collection",
    )
    store.client = Mock()

    chunk = Chunk(
        id="chunk1",
        document_id="doc1",
        content="Example text",
        chunk_index=3,
        section_title="Search",
        metadata={"language": "en"},
    )

    store.upsert_chunks(
        chunks=[chunk],
        embeddings=np.array([[0.1, 0.2]]),
        embedding_fingerprint="test-fingerprint",
    )

    call = store.client.upsert.call_args
    point = call.kwargs["points"][0]

    assert point.payload["chunk_index"] == 3
    assert (
        point.payload["_embedding_fingerprint"]
        == "test-fingerprint"
    )
    assert call.kwargs["wait"] is True


def test_bootstrap_cannot_overwrite_collection():
    store = QdrantVectorStore(
        url="http://localhost:6333",
        collection_name="technical_docs_v2",
    )
    store.client = Mock()
    store.client.collection_exists.return_value = True

    with pytest.raises(RuntimeError):
        store.create_collection(
            vector_size=384,
            fail_if_exists=True,
        )

    store.client.create_collection.assert_not_called()
