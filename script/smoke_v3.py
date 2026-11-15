"""Smoke test the complete V3 RAG workflow."""

from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient

from technical_rag.config import (
    QDRANT_URL,
    V2_PROFILE,
    SPARSE_MODEL_ID,
)
from technical_rag.context import ContextBuilder
from technical_rag.generation.citations import CitationValidator
from technical_rag.generation.prompt import RAGPromptBuilder
from technical_rag.models import RAGResult
from technical_rag.retrieval.langchain_adapter import (
    to_search_results,
)
from technical_rag.retrieval.v3_store import build_v3_store
from technical_rag.generation.v3_llm import build_v3_llm

COLLECTION = "technical_docs_v3_hybrid"
QUESTION = "Which similarity metrics does Qdrant support?"


def main():
    started = perf_counter()

    store = build_v3_store()
    # 2. Retrieve and prepare evidence
    retrieved = store.similarity_search_with_score(
        QUESTION,
        k=10,
    )

    ranked = sorted(
        retrieved,
        key=lambda item: (
            -item[1],
            item[0].metadata["chunk_id"],
        ),
    )[:6]

    results = to_search_results(ranked)

    context_builder = ContextBuilder()
    evidence = context_builder.build_evidence(
        results,
        max_evidence=3,
    )

    assert evidence, "No evidence retrieved"

    retrieval_s = perf_counter() - started

    # 3. Build the existing RAG prompt
    context = context_builder.format_context(evidence)

    system_prompt, user_prompt = RAGPromptBuilder().build(
        question=QUESTION,
        context=context,
    )

    # 4. Generate with ChatOllama
    llm = build_v3_llm()

    generation_started = perf_counter()

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    answer = response.content
    assert isinstance(answer, str) and answer.strip()

    generation_s = perf_counter() - generation_started

    # 5. Validate citations
    validation_started = perf_counter()

    validator = CitationValidator()

    valid, invalid = validator.validate(
        answer=answer,
        evidence=evidence,
    )

    uncited = validator.extract_uncited_statements(answer)

    validation_s = perf_counter() - validation_started

    # 6. Preserve our structured RAG output
    result = RAGResult(
        question=QUESTION,
        answer=answer,
        evidence=evidence,
        valid_citations=valid,
        invalid_citations=invalid,
        uncited_statements=uncited,
        grounding=[],
        timing_seconds={
            "retrieval": round(retrieval_s, 3),
            "generation": round(generation_s, 3),
            "validation": round(validation_s, 3),
            "total": round(perf_counter() - started, 3),
        },
    )

    print("\nANSWER:")
    print(result.answer)
    print("\nValid citations:", result.valid_citations)
    print("Invalid citations:", result.invalid_citations)
    print("Evidence count:", len(result.evidence))
    print("Timings:", result.timing_seconds)

    assert result.valid_citations
    assert not result.invalid_citations

    print("\nV3 RAG SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
