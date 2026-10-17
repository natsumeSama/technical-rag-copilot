import re

import numpy as np
from sentence_transformers import CrossEncoder

from ..models import (
    CitationSupportResult,
    Evidence,
)


class CitationValidator:
    CITATION_PATTERN = re.compile(r"\[(S\d+)\]")

    def extract_citations(
        self,
        answer: str,
    ) -> list[str]:
        citations = self.CITATION_PATTERN.findall(answer)

        return list(dict.fromkeys(citations))

    def validate(
        self,
        answer: str,
        evidence: list[Evidence],
    ) -> tuple[list[str], list[str]]:
        cited_ids = self.extract_citations(answer)

        available_ids = {
            item.citation_id
            for item in evidence
        }

        valid_ids = [
            citation_id
            for citation_id in cited_ids
            if citation_id in available_ids
        ]

        invalid_ids = [
            citation_id
            for citation_id in cited_ids
            if citation_id not in available_ids
        ]

        return valid_ids, invalid_ids

    def extract_claim_citation_pairs(
        self,
        answer: str,
    ) -> list[tuple[str, str]]:
        pairs = []

        # Split the answer into individual sentences or lines.
        segments = re.split(
            r"(?<=[.!?])\s+(?!\[S\d+\])|(?<=\])\s+(?=[A-Z])|\n+",
            answer.strip(),
        )

        for segment in segments:
            segment = segment.strip()

            if not segment:
                continue

            citation_ids = self.CITATION_PATTERN.findall(
                segment
            )

            if not citation_ids:
                continue

            # Remove citation markers from the claim text.
            claim = self.CITATION_PATTERN.sub(
                "",
                segment,
            )

            # Remove bullet characters if present.
            claim = re.sub(
                r"^[\-\*\u2022]\s*",
                "",
                claim,
            )

            # Clean punctuation/spaces left after removing citations.
            claim = re.sub(
                r"\s+,",
                ",",
                claim,
            )

            claim = re.sub(
                r",\s*\.",
                ".",
                claim,
            )

            claim = re.sub(
                r"\s+\.",
                ".",
                claim,
            )

            claim = re.sub(
                r"\s{2,}",
                " ",
                claim,
            )

            claim = claim.strip()

            if not claim:
                continue

            citation_ids = list(
                dict.fromkeys(citation_ids)
            )

            for citation_id in citation_ids:
                pairs.append(
                    (
                        claim,
                        citation_id,
                    )
                )

        return pairs

class CitationGroundingValidator:
    def __init__(
        self,
        model_name: str,
    ):
        self.model = CrossEncoder(model_name)

        config = self.model.model.config

        self.id2label = {
            int(index): label.lower()
            for index, label
            in config.id2label.items()
        }

        self.label2id = {
            label.lower(): int(index)
            for label, index
            in config.label2id.items()
        }

        self.citation_validator = CitationValidator()

    def validate(
        self,
        answer: str,
        evidence: list[Evidence],
    ) -> list[CitationSupportResult]:

        evidence_by_id = {
            item.citation_id: item
            for item in evidence
        }

        claim_citation_pairs = (
            self.citation_validator
            .extract_claim_citation_pairs(answer)
        )

        model_pairs = []
        metadata = []

        for claim, citation_id in claim_citation_pairs:

            evidence_item = evidence_by_id.get(
                citation_id
            )

            if evidence_item is None:
                continue

            premise = (
                f"Document: "
                f"{evidence_item.document_title}\n"
                f"Section: "
                f"{evidence_item.section_title}\n"
                f"{evidence_item.content}"
            )

            model_pairs.append(
                (
                    premise,
                    claim,
                )
            )

            metadata.append(
                (
                    claim,
                    citation_id,
                )
            )

        if not model_pairs:
            return []

        logits = np.asarray(
            self.model.predict(model_pairs)
        )

        probabilities = self._softmax(logits)

        results = []

        entailment_index = self.label2id[
            "entailment"
        ]

        neutral_index = self.label2id[
            "neutral"
        ]

        contradiction_index = self.label2id[
            "contradiction"
        ]

        for (
            claim_data,
            scores,
            probabilities_row,
        ) in zip(
            metadata,
            logits,
            probabilities,
        ):
            claim, citation_id = claim_data

            best_index = int(
                np.argmax(scores)
            )

            label = self.id2label[
                best_index
            ]

            results.append(
                CitationSupportResult(
                    claim=claim,
                    citation_id=citation_id,
                    label=label,
                    supported=(
                        label == "entailment"
                    ),
                    entailment_score=float(
                        probabilities_row[
                            entailment_index
                        ]
                    ),
                    neutral_score=float(
                        probabilities_row[
                            neutral_index
                        ]
                    ),
                    contradiction_score=float(
                        probabilities_row[
                            contradiction_index
                        ]
                    ),
                )
            )

        return results

    @staticmethod
    def _softmax(
        scores: np.ndarray,
    ) -> np.ndarray:
        shifted_scores = (
            scores
            - np.max(
                scores,
                axis=1,
                keepdims=True,
            )
        )

        exponentials = np.exp(
            shifted_scores
        )

        return (
            exponentials
            / exponentials.sum(
                axis=1,
                keepdims=True,
            )
        )