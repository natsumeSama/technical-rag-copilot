from .models import Evidence, SearchResult


class ContextBuilder:
    def build_evidence(
        self,
        results: list[SearchResult],
    ) -> list[Evidence]:
        evidence = []
        seen_chunk_ids = set()

        for result in results:
            if result.chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(result.chunk_id)

            citation_id = f"S{len(evidence) + 1}"

            evidence.append(
                Evidence(
                    citation_id=citation_id,
                    chunk_id=result.chunk_id,
                    content=result.content,
                    document_title=result.metadata.get(
                        "document_title",
                        "Unknown",
                    ),
                    section_title=result.section_title,
                    source_name=result.metadata.get(
                        "source_name",
                        "Unknown",
                    ),
                    source_url=result.metadata.get(
                        "source_url",
                        "",
                    ),
                    retrieval_score=result.score,
                )
            )

        return evidence

    def format_context(
        self,
        evidence: list[Evidence],
    ) -> str:
        blocks = []

        for item in evidence:
            section = item.section_title or "N/A"

            block = (
                f"[{item.citation_id}]\n"
                f"Document: {item.document_title}\n"
                f"Section: {section}\n"
                f"Source: {item.source_name}\n"
                f"URL: {item.source_url}\n"
                f"Content:\n"
                f"{item.content}"
            )

            blocks.append(block)

        return "\n\n".join(blocks)