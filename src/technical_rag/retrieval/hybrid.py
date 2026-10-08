from ..embeddings import EmbeddingModel
from ..models import SearchResult
from ..vector_store import QdrantVectorStore
from .bm25 import BM25Retriever


class HybridRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: QdrantVectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        limit: int = 5,
        candidate_limit: int = 10,
    ) -> list[SearchResult]:

        query_embedding = self.embedding_model.embed_query(query)

        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=candidate_limit,
        )

        bm25_results = self.bm25_retriever.search(
            query=query,
            limit=candidate_limit,
        )

        fused = {}

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            fused[result.chunk_id] = {
                "result": result,
                "score": 1 / (self.rrf_k + rank),
                "dense_rank": rank,
                "bm25_rank": None,
            }

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            contribution = 1 / (
                self.rrf_k + rank
            )

            if result.chunk_id in fused:
                fused[result.chunk_id]["score"] += contribution
                fused[result.chunk_id]["bm25_rank"] = rank

            else:
                fused[result.chunk_id] = {
                    "result": result,
                    "score": contribution,
                    "dense_rank": None,
                    "bm25_rank": rank,
                }

        ranked = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        results = []

        for item in ranked[:limit]:
            result = item["result"]

            metadata = {
                **result.metadata,
                "dense_rank": item["dense_rank"],
                "bm25_rank": item["bm25_rank"],
            }

            results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    section_title=result.section_title,
                    score=item["score"],
                    metadata=metadata,
                )
            )

        return results