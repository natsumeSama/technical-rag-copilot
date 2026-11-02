from qdrant_client import QdrantClient

from uuid import NAMESPACE_URL, uuid5

import numpy as np
from .models import Chunk, SearchResult
from dataclasses import dataclass

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
    Distance,
    VectorParams,
    PointStruct,
)
@dataclass
class StoredChunk:
    chunk_id: str
    document_id: str
    content: str
    section_title: str | None
    metadata: dict[str, object]
    vector: np.ndarray
    chunk_index: int | None = None
class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
    ):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def create_collection(
        self,
        vector_size: int,
        *,
        fail_if_exists: bool = False,
    ) -> None:
        if self.client.collection_exists(
            collection_name=self.collection_name
        ):
            if fail_if_exists:
                raise RuntimeError(
                    f"Collection already exists: {self.collection_name}"
                )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        *,
        embedding_fingerprint: str | None = None,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings"
            )
        if not chunks:
            return

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or infinity")
        points = []

        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(
                uuid5(NAMESPACE_URL, chunk.id)
            )

            payload = {
                **chunk.metadata,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
            }

            if embedding_fingerprint is not None:
                payload["_embedding_fingerprint"] = embedding_fingerprint

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload,
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_embedding,
        limit: int = 5,
    ) -> list[SearchResult]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit,
            with_payload=True,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            result = SearchResult(
                chunk_id=payload["chunk_id"],
                document_id=payload["document_id"],
                content=payload["content"],
                section_title=payload.get("section_title"),
                score=point.score,
                metadata={
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "chunk_id",
                        "document_id",
                        "content",
                        "section_title",
                    }
                },
            )

            results.append(result)

        return results

    def reset_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(
            collection_name=self.collection_name
        ):
            self.client.delete_collection(
                collection_name=self.collection_name
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


    def get_document_chunks(
        self,
        document_id: str,
    ) -> dict[str, StoredChunk]:

        stored_chunks = {}
        offset = None

        # 1. Only select points belonging to this document
        document_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

        # 2. Retrieve all matching points, page by page
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=document_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            # 3. Extract each chunk and its embedding
            for point in points:
                payload = point.payload or {}

                chunk_id = payload.get("chunk_id")
                content = payload.get("content")
                vector = point.vector

                if (
                    not isinstance(chunk_id, str)
                    or payload.get("document_id") != document_id
                    or not isinstance(content, str)
                    or not isinstance(vector, list)
                    or not vector
                ):
                    raise ValueError("Invalid stored chunk")

                if chunk_id in stored_chunks:
                    raise ValueError(
                        f"Duplicate chunk ID: {chunk_id}"
                    )

                excluded = {
                    "chunk_id",
                    "document_id",
                    "content",
                    "section_title",
                }

                stored_chunks[chunk_id] = StoredChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=content,
                    section_title=payload.get("section_title"),
                    metadata={
                        key: value
                        for key, value in payload.items()
                        if key not in excluded
                    },
                    vector=np.asarray(vector, dtype=np.float32),
                    chunk_index=payload.get("chunk_index"),
                )

            # 4. Stop when all pages have been retrieved
            if offset is None:
                break

        return stored_chunks

    
    def get_document_payloads(
        self,
        document_id: str,
    ) -> dict[str, dict]:

        result = {}
        offset = None

        document_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=document_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")

                if not isinstance(chunk_id, str):
                    raise ValueError("Missing chunk ID")

                if payload.get("document_id") != document_id:
                    raise ValueError("Unexpected document ID")

                if chunk_id in result:
                    raise ValueError(
                        f"Duplicate chunk ID: {chunk_id}"
                    )

                result[chunk_id] = payload

            if offset is None:
                break

        return result


    def delete_chunks(
        self,
        chunk_ids: set[str],
        batch_size: int = 100,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        ordered_ids = sorted(chunk_ids)

        for start in range(0, len(ordered_ids), batch_size):
            batch = ordered_ids[start:start + batch_size]

            point_ids = [
                str(uuid5(NAMESPACE_URL, chunk_id))
                for chunk_id in batch
            ]

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(
                    points=point_ids,
                ),
                wait=True,
            )
