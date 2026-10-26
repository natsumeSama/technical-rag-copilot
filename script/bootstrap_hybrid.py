
from qdrant_client import QdrantClient

from technical_rag.config_v2 import (
    MODEL_ID,
    MODEL_REVISION,
    QDRANT_URL,
)
from technical_rag.embeddings import EmbeddingModel
from technical_rag.retrieval.hybrid_store import (
    create_hybrid_collection,
)


COLLECTION_NAME = "technical_docs_hybrid_exp"


def main():
    dense_model = EmbeddingModel(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    client = QdrantClient(url=QDRANT_URL)

    create_hybrid_collection(
        client=client,
        collection_name=COLLECTION_NAME,
        dense_dimension=dense_model.dimension,
    )

    print(f"Created collection: {COLLECTION_NAME}")
    print("Dense vector: dense")
    print("Sparse vector: bm25")


if __name__ == "__main__":
    main()
