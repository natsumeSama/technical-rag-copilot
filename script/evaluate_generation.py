
"""Evaluate the complete RAG pipeline."""

import json
from pathlib import Path

from technical_rag.factory import build_pipeline


ROOT = Path(__file__).resolve().parents[1]

EVAL_PATH = ROOT / "data/evaluation/generation_eval.json"

OUTPUT_PATH = (
    ROOT / "data/evaluation/results/generation_results_pipeline.json"
)


def save_results(results: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    if not cases:
        raise ValueError("Evaluation dataset is empty")

    # Initialize the shared RAG pipeline once
    rag = build_pipeline()

    records = []

    for index, case in enumerate(cases, start=1):
        print(
            f"\n[{index}/{len(cases)}] "
            f"{case['id']}: {case['question']}"
        )

        # Run the complete RAG pipeline
        result = rag.ask(case["question"])

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
        save_results(records)

        print(f"\nAnswer:\n{result.answer}")
        print(f"Valid citations: {result.valid_citations}")
        print(f"Invalid citations: {result.invalid_citations}")
        print(f"Uncited statements: {len(result.uncited_statements)}")
        print(f"NLI comparisons: {len(result.grounding)}")
        print(f"Total time: {result.timing_seconds['total']:.2f}s")

    print(f"\nSaved {len(records)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
