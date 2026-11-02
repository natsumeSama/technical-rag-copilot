
"""Verify hybrid synchronization against real Qdrant."""

from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from technical_rag.config import QDRANT_URL
from technical_rag.retrieval.hybrid_store import (
    create_hybrid_collection,
)
from technical_rag.retrieval.hybrid_index import (
    index_hybrid_points,
)


def make_point(point_id: str, content: str, token: int):
    return PointStruct(
        id=point_id,
        vector={
            "dense": [1.0, 0.0],
            "bm25": SparseVector(
                indices=[token],
                values=[1.0],
            ),
        },
        payload={
            "chunk_id": point_id,
            "document_id": "integration_test",
            "content": content,
        },
    )


def main():
    client = QdrantClient(url=QDRANT_URL)

    # Unique temporary collection: production data is untouched.
    collection = f"rag_sync_test_{uuid4().hex}"

    a_id = str(uuid4())
    b_id = str(uuid4())

    try:
        create_hybrid_collection(
            client=client,
            collection_name=collection,
            dense_dimension=2,
        )

        # Initial indexing: two points.
        index_hybrid_points(
            client,
            collection,
            [
                make_point(a_id, "original text", 10),
                make_point(b_id, "obsolete text", 20),
            ],
        )
        print("PASS: Initial indexing")

        # Update A and remove B.
        index_hybrid_points(
            client,
            collection,
            [make_point(a_id, "updated text", 30)],
        )

        stored = client.retrieve(
            collection_name=collection,
            ids=[a_id],
            with_payload=True,
        )

        assert len(stored) == 1
        assert stored[0].payload["content"] == "updated text"

        count = client.count(
            collection_name=collection,
            exact=True,
        ).count

        assert count == 1

        print("PASS: Updated point")
        print("PASS: Deleted stale point")
        print("PASS: Final collection verified")

    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.close()


if __name__ == "__main__":
    main()
