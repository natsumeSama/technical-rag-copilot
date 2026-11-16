"""Investigate how evidence segmentation affects NLI predictions."""

import json
import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter

from technical_rag.config import NLI_MODEL_ID, PROJECT_ROOT
from technical_rag.generation.citations import CitationGroundingValidator
from technical_rag.models import Evidence

RESULTS_PATH = PROJECT_ROOT / "data/evaluation/results/generation_results_v3.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=NLI_MODEL_ID,
        help="Pretrained NLI model to evaluate",
    )
    args = parser.parse_args()
    records = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    by_id = {record["id"]: record for record in records}

    a1 = next(
        item for item in by_id["A1"]["evidence"] if item["section_title"] == "Metrics"
    )

    a2 = next(
        item
        for item in by_id["A2"]["evidence"]
        if item["section_title"] == "Reciprocal Rank Fusion (RRF)"
    )

    cases = [
        (
            "A1 supporting",
            a1,
            "Qdrant supports cosine similarity.",
            "entailment",
        ),
        (
            "A1 contradicting",
            a1,
            "Qdrant does not support cosine similarity.",
            "contradiction",
        ),
        (
            "A2 supporting",
            a2,
            "RRF considers the positions of results within each query "
            "and boosts those that appear closer to the top "
            "in multiple sets of results.",
            "entailment",
        ),
        (
            "A2 contradicting",
            a2,
            "RRF ignores the positions of results within each query.",
            "contradiction",
        ),
        (
            "A1 unrelated",
            a1,
            "Qdrant was founded in 1997.",
            "neutral",
        ),
        (
            "A1 partially false",
            a1,
            "Qdrant supports cosine similarity, "
            "but it does not support Manhattan distance.",
            "contradiction",
        ),
        (
            "A2 unrelated",
            a2,
            "RRF was invented in 1975.",
            "neutral",
        ),
        (
            "A2 partially false",
            a2,
            "RRF considers the positions of results, "
            "but gives higher scores to lower-ranked results.",
            "contradiction",
        ),
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=75,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    validator = CitationGroundingValidator(args.model)
    print("NLI model:", args.model)

    for name, source, claim, expected in cases:
        print(f"\n{'=' * 65}")
        print(name)
        print("Expected:", expected)

        evidence = Evidence.model_validate(
            {
                **source,
                "citation_id": "S1",
            }
        )

        def evaluate(premise):
            short_evidence = evidence.model_copy(update={"content": premise})
            results = validator.validate(
                answer=f"{claim} [S1]",
                evidence=[short_evidence],
            )
            return results[0]

        full_result = evaluate(source["content"])

        print("\nFull chunk:")
        print(
            f"  {full_result.label} "
            f"(E={full_result.entailment_score:.3f}, "
            f"C={full_result.contradiction_score:.3f})"
        )

        segments = splitter.split_text(source["content"])
        predictions = []

        print(f"\nSmaller passages: {len(segments)}")

        for index, segment in enumerate(segments, start=1):
            result = evaluate(segment)
            predictions.append((segment, result))

            print(
                f"  Passage {index}: {result.label} "
                f"(E={result.entailment_score:.3f}, "
                f"C={result.contradiction_score:.3f})"
            )

        # Diagnostic only: inspect the most supportive passage.
        best_segment, best_result = max(
            predictions,
            key=lambda item: item[1].entailment_score,
        )

        print("\nHighest-entailment passage:")
        print(best_segment[:450])
        print("Predicted:", best_result.label)
        print(
            "Entailment score:",
            round(best_result.entailment_score, 3),
        )


if __name__ == "__main__":
    main()
