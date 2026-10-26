
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)


def create_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dimension: int,
) -> None:

    if dense_dimension <= 0:
        raise ValueError("Invalid dense dimension")

    if client.collection_exists(collection_name):
        raise RuntimeError(
            f"Collection already exists: {collection_name}"
        )

    client.create_collection(
        collection_name=collection_name,

        vectors_config={
            "dense": VectorParams(
                size=dense_dimension,
                distance=Distance.COSINE,
            ),
        },

        sparse_vectors_config={
            "bm25": SparseVectorParams(
                modifier=Modifier.IDF,
            ),
        },
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="document_id",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )
