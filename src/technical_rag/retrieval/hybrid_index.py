
"""Build and verify a dense + sparse Qdrant index."""

from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client.models import PointStruct

from ..embedding_text import embedding_text
from ..models import Chunk
from ..ingestion.embedding_reuse import EmbeddingSpec


@dataclass
class HybridBuild:
    points: list[PointStruct]
    reused: int
    computed: int


def build_hybrid_points(
    chunks: list[Chunk],
    dense_store,
    dense_model,
    sparse_encoder,
    spec: EmbeddingSpec,
) -> HybridBuild:

    if not chunks:
        raise ValueError("No chunks to index")

    if len({c.id for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")

    if dense_model.dimension != spec.dimension:
        raise ValueError("Dense embedding dimension mismatch")

    # 1. Load old vectors, grouped by document
    old_vectors = {}

    for document_id in {c.document_id for c in chunks}:
        stored = dense_store.get_document_chunks(document_id)

        old_vectors[document_id] = {
            embedding_text(point): point
            for point in stored.values()
        }

    # 2. Reuse compatible dense vectors
    fingerprint = spec.fingerprint()
    dense_vectors = [None] * len(chunks)
    missing = []
    missing_positions = []
    reused = 0

    for position, chunk in enumerate(chunks):
        old = old_vectors[chunk.document_id].get(
            embedding_text(chunk)
        )

        if old is not None:
            vector = np.asarray(
                old.vector, dtype=np.float32
            )

            compatible = (
                old.document_id == chunk.document_id
                and old.metadata.get(
                    "_embedding_fingerprint"
                ) == fingerprint
                and vector.shape == (spec.dimension,)
                and np.isfinite(vector).all()
            )

            if compatible:
                dense_vectors[position] = vector.copy()
                reused += 1
                continue

        missing.append(chunk)
        missing_positions.append(position)

    # 3. Calculate only missing dense embeddings
    if missing:
        calculated = np.asarray(
            dense_model.embed_documents(missing),
            dtype=np.float32,
        )

        if (
            calculated.shape
            != (len(missing), spec.dimension)
            or not np.isfinite(calculated).all()
        ):
            raise ValueError("Invalid dense embeddings")

        for position, vector in zip(
            missing_positions, calculated
        ):
            dense_vectors[position] = vector

    # 4. Generate sparse vectors for every chunk
    sparse_vectors = sparse_encoder.encode_chunks(chunks)

    if len(sparse_vectors) != len(chunks):
        raise ValueError("Missing sparse embeddings")

    # 5. Build Qdrant points containing both vectors
    points = []

    for chunk, dense, sparse in zip(
        chunks, dense_vectors, sparse_vectors
    ):
        if (
            len(sparse.indices) == 0
            or len(sparse.indices) != len(sparse.values)
            or len(set(sparse.indices)) != len(sparse.indices)
            or not np.isfinite(sparse.values).all()
        ):
            raise ValueError("Invalid sparse embedding")

        points.append(
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.id)),
                vector={
                    "dense": dense.tolist(),
                    "bm25": sparse,
                },
                payload={
                    **chunk.metadata,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "_embedding_fingerprint": fingerprint,
                    "_sparse_model": "Qdrant/bm25",
                },
            )
        )

    return HybridBuild(
        points=points,
        reused=reused,
        computed=len(missing),
    )


def read_points(client, collection_name, with_vectors=False):
    """Read every point from one collection."""
    result = {}
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=with_vectors,
            with_vectors=with_vectors,
        )

        for point in points:
            point_id = str(point.id)

            if point_id in result:
                raise ValueError("Duplicate Qdrant point ID")

            result[point_id] = point

        if offset is None:
            break

    return result


def index_hybrid_points(
    client,
    collection_name: str,
    points: list[PointStruct],
    batch_size: int = 100,
) -> int:

    if batch_size <= 0:
        raise ValueError("Invalid batch size")

    if not client.collection_exists(collection_name):
        raise RuntimeError("Hybrid collection does not exist")

    expected = {str(point.id): point for point in points}

    if len(expected) != len(points):
        raise ValueError("Duplicate point IDs")

    # 1. Protect against unexpected existing data
    existing = read_points(client, collection_name)

    if set(existing) - set(expected):
        raise RuntimeError(
            "Hybrid collection contains unexpected points"
        )

    # 2. Upload chunks in batches
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]

        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )

    # 3. Verify every stored point
    actual = read_points(
        client,
        collection_name,
        with_vectors=True,
    )

    if set(actual) != set(expected):
        raise RuntimeError("Hybrid index has missing or extra points")

    for point_id, expected_point in expected.items():
        stored = actual[point_id]

        if (
            not isinstance(stored.vector, dict)
            or "dense" not in stored.vector
            or "bm25" not in stored.vector
            or stored.payload != expected_point.payload
        ):
            raise RuntimeError(
                f"Invalid hybrid point: {point_id}"
            )

    return len(actual)
