"""Convert LangChain retrieval results to our RAG evidence format."""

from langchain_core.documents import Document

from technical_rag.models import SearchResult


def to_search_results(
    results: list[tuple[Document, float]],
) -> list[SearchResult]:
    converted = []

    system_fields = {
        "chunk_id",
        "document_id",
        "original_content",
        "section_title",
        "chunk_index",
        "_id",
        "_collection_name",
    }

    for document, score in results:
        metadata = document.metadata

        converted.append(
            SearchResult(
                chunk_id=metadata["chunk_id"],
                document_id=metadata["document_id"],
                content=metadata["original_content"],
                section_title=metadata.get("section_title"),
                score=score,
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key not in system_fields
                },
            )
        )

    return converted
