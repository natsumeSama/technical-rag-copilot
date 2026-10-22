
from unittest.mock import Mock

import numpy as np
import pytest

from technical_rag.models import Chunk
from technical_rag.ingestion.embedding_reuse import PreparedEmbeddings
from technical_rag.ingestion.qdrant_sync import sync_document


def make_chunk(chunk_id, text, index):
    return Chunk(
        id=chunk_id,
        document_id="doc1",
        content=text,
        chunk_index=index,
        section_title="Search",
        metadata={},
    )


def payload(chunk):
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "section_title": chunk.section_title,
        "chunk_index": chunk.chunk_index,
        "_embedding_fingerprint": "fingerprint1",
    }


def test_successful_sync():
    a = make_chunk("A", "Introduction", 0)
    b2 = make_chunk("B2", "Updated section", 1)
    d = make_chunk("D", "New section", 2)

    prepared = PreparedEmbeddings(
        chunks=[b2, d],
        vectors=np.array([
            [0.1, 0.2],
            [0.3, 0.4],
        ]),
        skipped_ids=["A"],
        delete_ids={"B", "C"},
        reused_count=1,
        computed_count=1,
    )

    store = Mock()
    store.collection_name = "technical_docs_v2"
    store.get_document_payloads.return_value = {
        "A": payload(a),
        "B2": payload(b2),
        "D": payload(d),
    }

    report = sync_document(
        store=store,
        document_id="doc1",
        expected_chunks=[a, b2, d],
        prepared=prepared,
        embedding_fingerprint="fingerprint1",
    )

    assert report.upserted == 2
    assert report.deleted == 2
    assert report.verified == 3

    store.upsert_chunks.assert_called_once()
    store.delete_chunks.assert_called_once_with(
        chunk_ids={"B", "C"},
        batch_size=100,
    )

    operations = [
        call[0]
        for call in store.method_calls
    ]

    assert operations == [
        "upsert_chunks",
        "delete_chunks",
        "get_document_payloads",
    ]


def test_sync_detects_incomplete_deletion():
    a = make_chunk("A", "Introduction", 0)

    prepared = PreparedEmbeddings(
        chunks=[],
        vectors=np.empty((0, 2)),
        skipped_ids=["A"],
        delete_ids={"B"},
        reused_count=0,
        computed_count=0,
    )

    store = Mock()
    store.collection_name = "technical_docs_v2"

    store.get_document_payloads.return_value = {
        "A": payload(a),
        "B": payload(make_chunk("B", "Obsolete", 1)),
    }

    with pytest.raises(RuntimeError):
        sync_document(
            store=store,
            document_id="doc1",
            expected_chunks=[a],
            prepared=prepared,
            embedding_fingerprint="fingerprint1",
        )


def test_invalid_plan_cannot_write():
    a = make_chunk("A", "Introduction", 0)

    prepared = PreparedEmbeddings(
        chunks=[],
        vectors=np.empty((0, 2)),
        skipped_ids=[],
        delete_ids=set(),
        reused_count=0,
        computed_count=0,
    )

    store = Mock()
    store.collection_name = "technical_docs_v2"

    with pytest.raises(ValueError):
        sync_document(
            store=store,
            document_id="doc1",
            expected_chunks=[a],
            prepared=prepared,
            embedding_fingerprint="fingerprint1",
        )

    store.upsert_chunks.assert_not_called()
    store.delete_chunks.assert_not_called()
