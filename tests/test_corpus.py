
import json

import pytest

from technical_rag.ingestion.corpus import (
    load_documents,
    load_chunks,
)


@pytest.fixture
def sample_corpus(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "search.md").write_text(
        "# Search\n\nQdrant supports cosine similarity.",
        encoding="utf-8",
    )

    source = {
        "file_path": "docs/search.md",
        "source_name": "Qdrant",
        "source_url": "https://example.com/search",
        "document_type": "documentation",
        "metadata": {"language": "en"},
    }

    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps([source]),
        encoding="utf-8",
    )

    return manifest, tmp_path, source


def test_load_documents(sample_corpus):
    manifest, root, _ = sample_corpus

    documents = load_documents(manifest, root)

    assert len(documents) == 1
    assert documents[0].source_name == "Qdrant"
    assert "cosine similarity" in documents[0].content


def test_load_chunks(sample_corpus):
    manifest, root, _ = sample_corpus

    documents = load_documents(manifest, root)

    chunks = load_chunks(
        sources_path=manifest,
        project_root=root,
        chunk_size=1000,
        chunk_overlap=150,
    )

    assert len(chunks) >= 1
    assert chunks[0].document_id == documents[0].id
    assert "cosine similarity" in chunks[0].content


def test_duplicate_documents_are_rejected(sample_corpus):
    manifest, root, source = sample_corpus

    manifest.write_text(
        json.dumps([source, source]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate document ID"):
        load_documents(manifest, root)


def test_empty_manifest_is_rejected(sample_corpus):
    manifest, root, _ = sample_corpus

    manifest.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        load_documents(manifest, root)
