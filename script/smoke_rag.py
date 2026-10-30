
from technical_rag.factory import build_pipeline


def main():
    rag = build_pipeline()

    result = rag.ask(
        "Which similarity metrics does Qdrant support?"
    )

    print("\nQUESTION")
    print(result.question)

    print("\nANSWER")
    print(result.answer)

    print("\nEVIDENCE")
    for item in result.evidence:
        print(
            f"[{item.citation_id}] "
            f"{item.document_title} | {item.section_title}"
        )

    print("\nCITATION VALIDATION")
    print("Valid:", result.valid_citations)
    print("Invalid:", result.invalid_citations)
    print("Uncited:", result.uncited_statements)

    print("\nNLI GROUNDING")
    for item in result.grounding:
        print(
            f"[{item.citation_id}] "
            f"{item.claim} -> {item.label}"
        )

    print("\nPERFORMANCE")
    print(result.timing_seconds)


if __name__ == "__main__":
    main()
