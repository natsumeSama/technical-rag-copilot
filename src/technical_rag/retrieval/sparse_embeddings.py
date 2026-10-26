
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from ..embedding_text import embedding_text
from ..models import Chunk


class BM25SparseEncoder:
    def __init__(self, model=None):
        self.model = (
            model
            if model is not None
            else SparseTextEmbedding(
                model_name="Qdrant/bm25"
            )
        )

    def encode_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[SparseVector]:

        if not chunks:
            return []

        texts = [
            embedding_text(chunk)
            for chunk in chunks
        ]

        embeddings = self.model.embed(texts)

        return [
            SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )
            for embedding in embeddings
        ]

    def encode_query(self, query: str) -> SparseVector:
        embedding = next(self.model.query_embed(query))

        return SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )
