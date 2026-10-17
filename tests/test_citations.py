
from technical_rag.generation.citations import CitationValidator


def test_citation_after_period():
    validator = CitationValidator()

    answer = "Qdrant supports cosine similarity. [S1]"

    pairs = validator.extract_claim_citation_pairs(answer)

    assert pairs == [
        ("Qdrant supports cosine similarity.", "S1")
    ]


def test_multiple_sentences_with_citations():
    validator = CitationValidator()

    answer = (
        "Qdrant supports cosine similarity. [S1] "
        "It also supports dot product. [S2]"
    )

    pairs = validator.extract_claim_citation_pairs(answer)

    assert pairs == [
        ("Qdrant supports cosine similarity.", "S1"),
        ("It also supports dot product.", "S2"),
    ]


def test_uncited_statements():
    validator = CitationValidator()

    answer = (
        "Qdrant supports cosine similarity. [S1]\n"
        "Qdrant supports dot product. [S2]\n"
        "Qdrant automatically encrypts every vector."
    )

    uncited = validator.extract_uncited_statements(answer)

    assert uncited == [
        "Qdrant automatically encrypts every vector."
    ]


def test_all_statements_cited():
    validator = CitationValidator()

    answer = (
        "Qdrant supports cosine similarity. [S1]\n"
        "Qdrant supports dot product. [S2]"
    )

    assert validator.extract_uncited_statements(answer) == []
