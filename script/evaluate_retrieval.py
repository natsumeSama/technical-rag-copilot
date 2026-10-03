import json
from pathlib import Path

from technical_rag.embeddings import EmbeddingModel
from technical_rag.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.json"

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


with open(EVAL_PATH, "r", encoding="utf-8") as file:
    eval_data = json.load(file)


embedding_model = EmbeddingModel(MODEL_NAME)

vector_store = QdrantVectorStore(
    url="http://localhost:6333",
    collection_name="technical_docs",
)


recall_at_1 = 0
recall_at_3 = 0
reciprocal_ranks = []


for example in eval_data:
    question = example["question"]
    relevant_section = example["relevant_section"]

    query_embedding = embedding_model.embed_query(question)

    results = vector_store.search(
        query_embedding=query_embedding,
        limit=3,
    )

    retrieved_sections = [
        result.section_title
        for result in results
    ]

    if relevant_section in retrieved_sections[:1]:
        recall_at_1 += 1

    if relevant_section in retrieved_sections[:3]:
        recall_at_3 += 1

    rank = None

    for index, section in enumerate(retrieved_sections, start=1):
        if section == relevant_section:
            rank = index
            break

    if rank is not None:
        reciprocal_ranks.append(1 / rank)
    else:
        reciprocal_ranks.append(0)

    print(f"\nQuestion: {question}")
    print(f"Expected: {relevant_section}")
    print(f"Retrieved: {retrieved_sections}")
    print(f"Rank: {rank}")


n_questions = len(eval_data)

recall_at_1 /= n_questions
recall_at_3 /= n_questions
mrr = sum(reciprocal_ranks) / n_questions


print("\n" + "=" * 50)
print(f"Recall@1: {recall_at_1:.3f}")
print(f"Recall@3: {recall_at_3:.3f}")
print(f"MRR:      {mrr:.3f}")