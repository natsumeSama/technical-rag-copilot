
"""Load the documentation corpus from its source manifest."""

import json
from pathlib import Path

from ..chunking import chunk_document
from ..models import Chunk, Document
from .markdown import load_markdown


def load_documents(
    sources_path: Path,
    project_root: Path,
) -> list[Document]:
    """Load and validate all source documents."""

    sources = json.loads(
        sources_path.read_text(encoding="utf-8")
    )

    if not isinstance(sources, list) or not sources:
        raise ValueError("Source manifest must be a non-empty list")

    documents = []
    seen_ids = set()

    for source in sources:
        document = load_markdown(
            file_path=str(project_root / source["file_path"]),
            source_name=source["source_name"],
            source_url=source["source_url"],
            document_type=source["document_type"],
            metadata=source["metadata"],
        )

        if document.id in seen_ids:
            raise ValueError(
                f"Duplicate document ID: {document.id}"
            )

        seen_ids.add(document.id)
        documents.append(document)

    return documents


def load_chunks(
    sources_path: Path,
    project_root: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Load the corpus and split its documents into chunks."""

    documents = load_documents(
        sources_path=sources_path,
        project_root=project_root,
    )

    chunks = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    if not chunks:
        raise ValueError("No chunks found in corpus")

    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")

    return chunks
