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


hit_at_1 = 0
hit_at_3 = 0
reciprocal_ranks = []


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

    first_relevant_rank = None

    for rank, target in enumerate(
        retrieved_targets,
        start=1,
    ):
        if target in relevant_targets:
            first_relevant_rank = rank
            break

    if first_relevant_rank == 1:
        hit_at_1 += 1

    if first_relevant_rank is not None:
        hit_at_3 += 1
        reciprocal_ranks.append(
            1 / first_relevant_rank
        )
    else:
        reciprocal_ranks.append(0)

    print("\n" + "=" * 80)
    print(f"Question: {question}")
    print(f"Expected: {relevant_targets}")

    for rank, target in enumerate(
        retrieved_targets,
        start=1,
    ):
        marker = "✓" if target in relevant_targets else " "
        print(
            f"{marker} Rank {rank}: "
            f"{target[0]} | {target[1]}"
        )

    print(
        f"First relevant rank: "
        f"{first_relevant_rank}"
    )


n_questions = len(eval_data)

hit_at_1 /= n_questions
hit_at_3 /= n_questions
mrr = sum(reciprocal_ranks) / n_questions


print("\n" + "=" * 80)
print("DENSE RETRIEVAL BASELINE")
print("=" * 80)
print(f"Questions: {n_questions}")
print(f"Hit@1:    {hit_at_1:.3f}")
print(f"Hit@3:    {hit_at_3:.3f}")
print(f"MRR:      {mrr:.3f}")