
"""Compare dense, sparse and hybrid retrieval."""

import argparse

from qdrant_client import QdrantClient

from technical_rag.config_v2 import (
    MODEL_ID,
    MODEL_REVISION,
    QDRANT_URL,
)
from technical_rag.embeddings import EmbeddingModel
from technical_rag.retrieval.sparse_embeddings import (
    BM25SparseEncoder,
)
from technical_rag.retrieval.qdrant_hybrid import (
    QdrantHybridRetriever,
)
from technical_rag.config import HYBRID_PROFILE

COLLECTION_NAME = HYBRID_PROFILE.collection_name


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        default="Which similarity metrics does Qdrant support?",
    )

    args = parser.parse_args()

    retriever = QdrantHybridRetriever(
        client=QdrantClient(url=QDRANT_URL),
        collection_name=COLLECTION_NAME,
        dense_model=EmbeddingModel(
            MODEL_ID,
            revision=MODEL_REVISION,
        ),
        sparse_encoder=BM25SparseEncoder(),
    )

    for mode in ("dense", "sparse", "hybrid"):
        print(f"\n{'=' * 50}")
        print(f"{mode.upper()} SEARCH")
        print("=" * 50)

        results = retriever.search(
            query=args.query,
            mode=mode,
            limit=5,
        )

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"{result.section_title} "
                f"(score={result.score:.4f})"
            )


if __name__ == "__main__":
    main()
