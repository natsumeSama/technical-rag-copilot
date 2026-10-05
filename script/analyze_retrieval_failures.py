import json
from pathlib import Path

from technical_rag.embeddings import EmbeddingModel
from technical_rag.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.json"

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "technical_docs_v1"


with open(EVAL_PATH, "r", encoding="utf-8") as file:
    eval_data = json.load(file)


embedding_model = EmbeddingModel(MODEL_NAME)

vector_store = QdrantVectorStore(
    url="http://localhost:6333",
    collection_name=COLLECTION_NAME,
)


for example in eval_data:
    question = example["question"]

    relevant_targets = {
        (
            target["document_title"],
            target["section_title"],
        )
        for target in example["relevant_targets"]
    }

    query_embedding = embedding_model.embed_query(question)

    results = vector_store.search(
        query_embedding=query_embedding,
        limit=3,
    )

    retrieved_targets = [
        (
            result.metadata.get("document_title"),
            result.section_title,
        )
        for result in results
    ]

    success = any(
        target in relevant_targets
        for target in retrieved_targets
    )

    if success:
        continue

    print("\n" + "=" * 100)
    print(f"FAILED QUESTION: {question}")
    print(f"EXPECTED: {relevant_targets}")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Rank {rank} ---")
        print(
            f"{result.metadata.get('document_title')} "
            f"| {result.section_title}"
        )
        print(f"Score: {result.score:.4f}")
        print(result.content[:800])