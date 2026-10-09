from sentence_transformers import CrossEncoder

from ..models import SearchResult


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        limit: int = 5,
    ) -> list[SearchResult]:

        if not results:
            return []

        pairs = []

        for result in results:
            if result.section_title:
                text = (
                    f"{result.section_title}\n\n"
                    f"{result.content}"
                )
            else:
                text = result.content

            pairs.append(
                (query, text)
            )

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):
            metadata = {
                **result.metadata,
                "retrieval_score": result.score,
            }

            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    section_title=result.section_title,
                    score=float(score),
                    metadata=metadata,
                )
            )

        reranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked[:limit]