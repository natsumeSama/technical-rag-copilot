
import json
from pathlib import Path

from technical_rag.models import Evidence
from technical_rag.generation.citations import CitationGroundingValidator
import numpy as np



ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data/evaluation/grounding_eval.json"

MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

with open(EVAL_PATH, encoding="utf-8") as f:
    cases = json.load(f)

validator = CitationGroundingValidator(model_name=MODEL)

correct = 0
labels = ["entailment", "contradiction", "neutral"]
stats = {label: {"correct": 0, "total": 0} for label in labels}

for case in cases:
    evidence = Evidence(
        citation_id="S1",
        chunk_id=case["id"],
        content=case["evidence"],
        document_title="Qdrant Documentation",
        section_title="Evaluation",
        source_name="Qdrant",
        source_url="",
        retrieval_score=1.0,
    )

    answer = f'{case["claim"]} [S1]'

    results = validator.validate(
        answer=answer,
        evidence=[evidence],
    )

    prediction = results[0].label
    expected = case["expected_label"]

    # Direct NLI prediction, without document metadata
    raw_logits = np.asarray(
        validator.model.predict([
            (case["evidence"], case["claim"])
        ])
    )

    raw_prediction = validator.id2label[
        int(np.argmax(raw_logits[0]))
    ]

    print(
        f'{case["id"]}: '
        f'Expected={expected:<14} '
        f'Wrapped={prediction:<14} '
        f'Raw={raw_prediction:<14}'
    )

    passed = prediction == expected

    stats[expected]["total"] += 1

    if passed:
        correct += 1
        stats[expected]["correct"] += 1

    print(
        f'{case["id"]}: '
        f'Expected={expected:<14} '
        f'Predicted={prediction:<14} '
        f'{"PASS" if passed else "FAIL"}'
    )

print("\n" + "=" * 60)
print("GROUNDING EVALUATION")
print("=" * 60)

print(f"Accuracy: {correct}/{len(cases)} ({correct / len(cases):.1%})")

for label, values in stats.items():
    print(
        f'{label}: '
        f'{values["correct"]}/{values["total"]}'
    )
