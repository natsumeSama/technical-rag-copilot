from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client.models import PointStruct

from .models import Chunk, SearchResult

class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
    ):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def create_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(
            collection_name=self.collection_name
        ):
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
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings"
            )

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(
                uuid5(NAMESPACE_URL, chunk.id)
            )

            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "section_title": chunk.section_title,
                **chunk.metadata,
            }

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload,
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
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