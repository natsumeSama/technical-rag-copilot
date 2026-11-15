"""Reusable V3 RAG workflow."""

from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore

from technical_rag.context import ContextBuilder
from technical_rag.generation.citations import (
    CitationValidator,
    CitationGroundingValidator,
)
from technical_rag.generation.prompt import RAGPromptBuilder
from technical_rag.models import RAGResult
from technical_rag.retrieval.langchain_adapter import to_search_results


def ask_v3(
    question: str,
    store: QdrantVectorStore,
    llm: ChatOllama,
    grounding_validator: CitationGroundingValidator | None = None,
) -> RAGResult:
    if not question.strip():
        raise ValueError("Question cannot be empty")

    started = perf_counter()

    # 1. Retrieve and prepare evidence
    retrieved = store.similarity_search_with_score(question, k=10)

    ranked = sorted(
        retrieved,
        key=lambda item: (
            -item[1],
            item[0].metadata["chunk_id"],
        ),
    )[:6]

    context_builder = ContextBuilder()

    evidence = context_builder.build_evidence(
        to_search_results(ranked),
        max_evidence=3,
    )

    if not evidence:
        raise RuntimeError("No evidence retrieved")

    retrieval_s = perf_counter() - started

    # 2. Build the prompt
    system_prompt, user_prompt = RAGPromptBuilder().build(
        question=question,
        context=context_builder.format_context(evidence),
    )

    # 3. Generate the answer
    generation_started = perf_counter()

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    answer = response.content

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("LLM returned an empty answer")

    generation_s = perf_counter() - generation_started

    # 4. Validate citations
    validation_started = perf_counter()

    validator = CitationValidator()

    valid, invalid = validator.validate(
        answer=answer,
        evidence=evidence,
    )

    uncited = validator.extract_uncited_statements(answer)

    # 5. Optional semantic grounding assessment
    grounding = []

    if grounding_validator is not None:
        grounding = grounding_validator.validate(
            answer=answer,
            evidence=evidence,
        )

    validation_s = perf_counter() - validation_started

    # 5. Return our structured result
    return RAGResult(
        question=question,
        answer=answer,
        evidence=evidence,
        valid_citations=valid,
        invalid_citations=invalid,
        uncited_statements=uncited,
        grounding=grounding,
        timing_seconds={
            "retrieval": round(retrieval_s, 3),
            "generation": round(generation_s, 3),
            "validation": round(validation_s, 3),
            "total": round(perf_counter() - started, 3),
        },
    )
