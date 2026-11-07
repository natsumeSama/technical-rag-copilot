
"""Compare V2 and V3 hybrid retrieval quality."""

import json

import ir_measures
from ir_measures import Success, RR
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)
from qdrant_client import QdrantClient

from technical_rag.config import (
    HYBRID_PROFILE,
    PROJECT_ROOT,
    QDRANT_URL,
    SPARSE_MODEL_ID,
)
from technical_rag.embeddings import EmbeddingModel
from technical_rag.retrieval.qdrant_hybrid import QdrantHybridRetriever
from technical_rag.retrieval.sparse_embeddings import BM25SparseEncoder


K = 3
CANDIDATES = 20
V3_COLLECTION = "technical_docs_v3_hybrid"

cases = json.loads(
    (PROJECT_ROOT / "data/evaluation/retrieval_eval.json")
    .read_text(encoding="utf-8")
)

assert cases, "Evaluation dataset is empty"

client = QdrantClient(url=QDRANT_URL)

# Ensure that we are comparing complete indexes.
counts = [
    client.count(collection_name=name, exact=True).count
    for name in (
        HYBRID_PROFILE.collection_name,
        V3_COLLECTION,
    )
]
assert counts[0] == counts[1], (
    f"Collection sizes differ: {counts}"
)

# V2: existing hybrid retrieval implementation
v2 = QdrantHybridRetriever(
    client=client,
    collection_name=HYBRID_PROFILE.collection_name,
    dense_model=EmbeddingModel(
        HYBRID_PROFILE.embedding_model_id,
        revision=HYBRID_PROFILE.embedding_revision,
    ),
    sparse_encoder=BM25SparseEncoder(),
)

# V3: LangChain hybrid retrieval
v3 = QdrantVectorStore(
    client=client,
    collection_name=V3_COLLECTION,
    embedding=HuggingFaceEmbeddings(
        model_name=HYBRID_PROFILE.embedding_model_id,
        model_kwargs={
            "revision": HYBRID_PROFILE.embedding_revision
        },
        encode_kwargs={"normalize_embeddings": True},
    ),
    sparse_embedding=FastEmbedSparse(
        model_name=SPARSE_MODEL_ID
    ),
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="bm25",
)


def target_id(title, section):
    return f"{title}::{section}"


def rank_scores(targets):
    """Convert rankings to an ir-measures run."""
    scores = {}
    for rank, (title, section) in enumerate(targets[:K], 1):
        key = target_id(title, section)
        scores.setdefault(key, float(K + 1 - rank))
    return scores


qrels = {}
v2_run = {}
v3_run = {}

for index, case in enumerate(cases, 1):
    qid = f"q{index:02d}"
    question = case["question"]

    qrels[qid] = {
        target_id(t["document_title"], t["section_title"]): 1
        for t in case["relevant_targets"]
    }


    # Retrieve more candidates to handle tied scores fairly.
    old_results = v2.search(
        question,
        mode="hybrid",
        limit=CANDIDATES,
        candidate_limit=CANDIDATES,
    )

    new_results = v3.similarity_search_with_score(
        question,
        k=CANDIDATES,
    )

    # Higher score first; use stable chunk ID to break ties.
    old_results = sorted(
        old_results,
        key=lambda r: (-r.score, r.chunk_id),
    )[:K]

    new_results = sorted(
        new_results,
        key=lambda item: (
            -item[1],
            item[0].metadata["chunk_id"],
        ),
    )[:K]
         
    v2_run[qid] = rank_scores([
        (r.metadata.get("document_title"), r.section_title)
        for r in old_results
    ])

    v3_run[qid] = rank_scores([
        (
            doc.metadata.get("document_title"),
            doc.metadata.get("section_title"),
        )
        for doc, _ in new_results
    ])

    if v2_run[qid] != v3_run[qid]:
        print(f"Ranking difference: {qid} - {question}")


metrics = [Success@1, Success@3, RR]

print(f"\nEvaluated {len(cases)} questions")
print(f"Collection sizes: {counts}")

for name, run in [("V2", v2_run), ("V3", v3_run)]:
    scores = ir_measures.calc_aggregate(
        metrics, qrels, run
    )
    print(f"\n{name}")
    for metric, value in scores.items():
        label = "MRR@3" if metric == RR else str(metric)
        print(f"{label}: {value:.4f}")
