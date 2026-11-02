
from qdrant_client import QdrantClient

from technical_rag.config import (
    HYBRID_PROFILE,
    QDRANT_URL,
)
from technical_rag.embeddings import EmbeddingModel
from technical_rag.retrieval.hybrid_store import (
    create_hybrid_collection,
)


def main():
    dense_model = EmbeddingModel(
        HYBRID_PROFILE.embedding_model_id,
        revision=HYBRID_PROFILE.embedding_revision,
    )

    client = QdrantClient(url=QDRANT_URL)

    create_hybrid_collection(
        client=client,
        collection_name=HYBRID_PROFILE.collection_name,
        dense_dimension=dense_model.dimension,
    )

    print(
        f"Created collection: {HYBRID_PROFILE.collection_name}"
    )
    print("Dense vector: dense")
    print("Sparse vector: bm25")


if __name__ == "__main__":
    main()
