import numpy as np
from sentence_transformers import SentenceTransformer

from .models import Chunk
from .embedding_text import embedding_text

class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
    ):
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
        )
        
    def embed_documents(self, chunks: list[Chunk]) -> np.ndarray:
        texts = [embedding_text(chunk) for chunk in chunks]

        embeddings = self.model.encode_document(
            texts,
            normalize_embeddings=True,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode_query(
            query,
            normalize_embeddings=True,
        )

        return embedding

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()

    