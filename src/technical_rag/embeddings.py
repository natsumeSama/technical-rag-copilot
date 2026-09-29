import numpy as np
from sentence_transformers import SentenceTransformer

from .models import Chunk


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, chunks: list[Chunk]) -> np.ndarray:
        texts = []

        for chunk in chunks:
            if chunk.section_title:
                text = f"{chunk.section_title}\n\n{chunk.content}"
            else:
                text = chunk.content

            texts.append(text)

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