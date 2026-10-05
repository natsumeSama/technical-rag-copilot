import json
from pathlib import Path

from technical_rag.chunking import chunk_document
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.retrieval.bm25 import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"
EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------
# Build the same corpus used by our retrieval system
# --------------------------------------------------

with open(SOURCES_PATH, "r", encoding="utf-8") as file:
    sources = json.load(file)

all_chunks = []

for source in sources:
    document = load_markdown(
        file_path=str(PROJECT_ROOT / source["file_path"]),
        source_name=source["source_name"],
        source_url=source["source_url"],
        document_type=source["document_type"],
        metadata=source["metadata"],
    )

    chunks = chunk_document(
        document=document,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks.extend(chunks)


print(f"BM25 corpus: {len(all_chunks)} chunks")


# -------------------
# Build BM25 index
# -------------------

retriever = BM25Retriever()
retriever.index(all_chunks)


# -------------------
# Load evaluation set
# -------------------

with open(EVAL_PATH, "r", encoding="utf-8") as file:
    eval_data = json.load(file)


hit_at_1 = 0
hit_at_3 = 0
reciprocal_ranks = []


# -------------------
# Evaluation
# -------------------

for example in eval_data:
    question = example["question"]

    relevant_targets = {
        (
            target["document_title"],
            target["section_title"],
        )
        for target in example["relevant_targets"]
    }

    results = retriever.search(
        query=question,
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


# -------------------
# Final metrics
# -------------------

n_questions = len(eval_data)

hit_at_1 /= n_questions
hit_at_3 /= n_questions
mrr = sum(reciprocal_ranks) / n_questions


print("\n" + "=" * 80)
print("BM25 RETRIEVAL BASELINE")
print("=" * 80)
print(f"Questions: {n_questions}")
print(f"Hit@1:    {hit_at_1:.3f}")
print(f"Hit@3:    {hit_at_3:.3f}")
print(f"MRR:      {mrr:.3f}")