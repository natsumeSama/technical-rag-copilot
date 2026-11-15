"""Smoke test the reusable V3 RAG pipeline."""

from technical_rag.generation.v3_llm import build_v3_llm
from technical_rag.rag.v3_pipeline import ask_v3
from technical_rag.retrieval.v3_store import build_v3_store
from technical_rag.config import NLI_MODEL_ID
from technical_rag.generation.citations import (
    CitationGroundingValidator,
)


def main():
    store = build_v3_store()
    llm = build_v3_llm()
    grounding_validator = CitationGroundingValidator(
        model_name=NLI_MODEL_ID,
    )
    result = ask_v3(
        question="Which similarity metrics does Qdrant support?",
        store=store,
        llm=llm,
        grounding_validator=grounding_validator,
    )

    print("\nANSWER:")
    print(result.answer)

    print("\nValid citations:", result.valid_citations)
    print("Invalid citations:", result.invalid_citations)
    print("\nGrounding results:")

    for item in result.grounding:
        print(
            f"{item.citation_id}: {item.label} "
            f"(entailment={item.entailment_score:.3f})"
        )

    assert result.grounding, "No grounding results produced"
    print("Evidence count:", len(result.evidence))
    print("Timings:", result.timing_seconds)

    assert result.valid_citations
    assert not result.invalid_citations

    print("\nV3 RAG SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
