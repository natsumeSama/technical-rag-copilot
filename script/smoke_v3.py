"""Smoke test the reusable V3 RAG pipeline."""

from technical_rag.generation.v3_llm import build_v3_llm
from technical_rag.rag.v3_pipeline import ask_v3
from technical_rag.retrieval.v3_store import build_v3_store


def main():
    store = build_v3_store()
    llm = build_v3_llm()

    result = ask_v3(
        question="Which similarity metrics does Qdrant support?",
        store=store,
        llm=llm,
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
