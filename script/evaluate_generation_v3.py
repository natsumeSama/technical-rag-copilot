"""Evaluate the complete V3 RAG pipeline."""

import json

from technical_rag.config import PROJECT_ROOT, NLI_MODEL_ID
from technical_rag.generation.citations import (
    CitationGroundingValidator,
)
from technical_rag.generation.v3_llm import build_v3_llm
from technical_rag.rag.v3_pipeline import ask_v3
from technical_rag.retrieval.v3_store import build_v3_store

EVAL_PATH = PROJECT_ROOT / "data/evaluation/generation_eval.json"

OUTPUT_PATH = PROJECT_ROOT / "data/evaluation/results/generation_results_v3.json"


def main():
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    if not cases:
        raise ValueError("Evaluation dataset is empty")

    # Initialize components once
    store = build_v3_store()
    llm = build_v3_llm()
    grounding_validator = CitationGroundingValidator(
        model_name=NLI_MODEL_ID,
    )

    records = []

    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] " f"{case['id']}: {case['question']}")

        result = ask_v3(
            question=case["question"],
            store=store,
            llm=llm,
            grounding_validator=grounding_validator,
        )

        record = {
            "id": case["id"],
            "category": case["category"],
            "expected_behavior": case["expected_behavior"],
            "expected_facts": case.get("expected_facts", []),
            **result.model_dump(),
            "manual_review": {
                "behavior_correct": None,
                "answer_correct": None,
                "notes": "",
            },
        }

        records.append(record)

        # Save progress after each question
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(
                records,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\nANSWER:")
        print(result.answer)
        print("Valid citations:", result.valid_citations)
        print("Invalid citations:", result.invalid_citations)
        print("Uncited statements:", len(result.uncited_statements))

        print("\nNLI grounding:")
        for item in result.grounding:
            print(
                f"  [{item.citation_id}] "
                f"{item.label} "
                f"(entailment={item.entailment_score:.3f})"
            )

        print(f"Total time: " f"{result.timing_seconds['total']:.2f}s")

    print(f"\nSaved {len(records)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
