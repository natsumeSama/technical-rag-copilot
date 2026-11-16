"""Diagnose NLI grounding using real V3 evaluation evidence."""

import json

from technical_rag.config import NLI_MODEL_ID, PROJECT_ROOT
from technical_rag.generation.citations import (
    CitationGroundingValidator,
)
from technical_rag.models import Evidence

RESULTS_PATH = PROJECT_ROOT / "data/evaluation/results/generation_results_v3.json"


def main():
    records = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    by_id = {record["id"]: record for record in records}

    # Reuse the exact evidence retrieved during V3 evaluation.
    a1_source = next(
        item for item in by_id["A1"]["evidence"] if item["section_title"] == "Metrics"
    )

    a2_source = next(
        item
        for item in by_id["A2"]["evidence"]
        if item["section_title"] == "Reciprocal Rank Fusion (RRF)"
    )

    metric_claim = "Qdrant supports cosine similarity."

    rrf_claim = (
        "RRF considers the positions of results within each "
        "query and boosts those that appear closer to the top "
        "in multiple sets of results."
    )

    probes = [
        (
            "A1: short supporting evidence",
            a1_source,
            metric_claim,
            metric_claim,
            "entailment",
        ),
        (
            "A1: full supporting evidence",
            a1_source,
            a1_source["content"],
            metric_claim,
            "entailment",
        ),
        (
            "A1: contradiction",
            a1_source,
            metric_claim,
            "Qdrant does not support cosine similarity.",
            "contradiction",
        ),
        (
            "A1: unrelated claim",
            a1_source,
            metric_claim,
            "Qdrant earned 2 billion dollars in 2025.",
            "neutral",
        ),
        (
            "A2: short supporting evidence",
            a2_source,
            rrf_claim,
            rrf_claim,
            "entailment",
        ),
        (
            "A2: full supporting evidence",
            a2_source,
            a2_source["content"],
            rrf_claim,
            "entailment",
        ),
    ]

    # Load the pretrained NLI model only once.
    validator = CitationGroundingValidator(model_name=NLI_MODEL_ID)

    print("Model:", NLI_MODEL_ID)
    print("Label mapping:", validator.id2label)
    print("Max sequence length:", validator.model.max_seq_length)

    correct = 0

    for name, source, premise, claim, expected in probes:
        evidence = Evidence.model_validate(
            {
                **source,
                "citation_id": "S1",
                "content": premise,
            }
        )

        # Count tokens before truncation.
        tokens = validator.model.tokenizer(
            premise,
            claim,
            truncation=False,
        )

        token_count = len(tokens["input_ids"])

        # Run our existing validator, unchanged.
        results = validator.validate(
            answer=f"{claim} [S1]",
            evidence=[evidence],
        )

        assert len(results) == 1

        result = results[0]
        matches = result.label == expected
        correct += matches

        print(f"\n{name}")
        print(f"Expected: {expected}")
        print(f"Predicted: {result.label}")
        print(f"Match: {matches}")
        print(f"Input tokens: {token_count}")
        print(f"Entailment: {result.entailment_score:.3f}")
        print(f"Neutral: {result.neutral_score:.3f}")
        print(f"Contradiction: " f"{result.contradiction_score:.3f}")

    print(f"\nExpected-label matches: " f"{correct}/{len(probes)}")


if __name__ == "__main__":
    main()
