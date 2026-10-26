
"""Dense, sparse and hybrid retrieval using Qdrant."""

from typing import Literal

from qdrant_client import QdrantClient, models

from ..models import SearchResult


SearchMode = Literal["dense", "sparse", "hybrid"]


class QdrantHybridRetriever:

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_model,
        sparse_encoder,
    ):
        self.client = client
        self.collection_name = collection_name
        self.dense_model = dense_model
        self.sparse_encoder = sparse_encoder

    def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        limit: int = 5,
        candidate_limit: int = 20,
    ) -> list[SearchResult]:

        # 1. Validate the request
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if mode not in ("dense", "sparse", "hybrid"):
            raise ValueError("Invalid search mode")

        if limit <= 0 or candidate_limit < limit:
            raise ValueError("Invalid search limits")

        # 2. Generate only the query vectors we need
        dense_vector = None
        sparse_vector = None

        if mode in ("dense", "hybrid"):
            dense_vector = self.dense_model.embed_query(
                query
            ).tolist()

        if mode in ("sparse", "hybrid"):
            sparse_vector = self.sparse_encoder.encode_query(
                query
            )

        # 3. Execute the selected search
        if mode == "dense":
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                limit=limit,
                with_payload=True,
            )

        elif mode == "sparse":
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_vector,
                using="bm25",
                limit=limit,
                with_payload=True,
            )

        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=candidate_limit,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using="bm25",
                        limit=candidate_limit,
                    ),
                ],
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF,
                ),
                limit=limit,
                with_payload=True,
            )

        # 4. Convert Qdrant results into our RAG model
        results = []

        system_fields = {
            "chunk_id",
            "document_id",
            "content",
            "section_title",
            "chunk_index",
            "_embedding_fingerprint",
            "_sparse_model",
        }

        for point in response.points:
            payload = point.payload or {}

            results.append(
                SearchResult(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    content=payload["content"],
                    section_title=payload.get("section_title"),
                    score=point.score,
                    metadata={
                        key: value
                        for key, value in payload.items()
                        if key not in system_fields
                    },
                )
            )

        return results
