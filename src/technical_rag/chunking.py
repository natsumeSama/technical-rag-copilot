from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from hashlib import sha256
from .models import Document,Chunk

from hashlib import sha256

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .models import Chunk, Document


def chunk_document(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True,
    )

    sections = markdown_splitter.split_text(document.content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    parts = text_splitter.split_documents(sections)

    chunks: list[Chunk] = []

    for chunk_index, part in enumerate(parts):
        section_title = (
            part.metadata.get("h3")
            or part.metadata.get("h2")
            or part.metadata.get("h1")
        )

        identifier = (
            document.id
            + str(chunk_index)
            + part.page_content
        )

        chunk_id = sha256(
            identifier.encode("utf-8")
        ).hexdigest()

        chunk_metadata = {
            "document_title": document.title,
            "source_name": document.source_name,
            "source_url": document.source_url,
            "document_type": document.document_type,
            **document.metadata,
            **part.metadata,
        }

        chunk = Chunk(
            id=chunk_id,
            document_id=document.id,
            content=part.page_content,
            chunk_index=chunk_index,
            section_title=section_title,
            metadata=chunk_metadata,
        )

        chunks.append(chunk)

    return chunks