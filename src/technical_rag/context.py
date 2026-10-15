from .models import Evidence, SearchResult


class ContextBuilder:
    def build_evidence(
        self,
        results: list[SearchResult],
        max_evidence: int = 3,
    ) -> list[Evidence]:
        evidence = []

        seen_chunk_ids = set()
        seen_sections = set()

        for result in results:
            # 1. Skip exact duplicate chunks
            if result.chunk_id in seen_chunk_ids:
                continue

            document_title = result.metadata.get(
                "document_title",
                "Unknown",
            )

            # 2. Build a section identity
            section_key = (
                document_title,
                result.section_title,
            )

            # 3. Skip duplicate sections,
            # but only if the section title actually exists
            if (
                result.section_title is not None
                and section_key in seen_sections
            ):
                continue

            # 4. Mark this chunk as seen
            seen_chunk_ids.add(result.chunk_id)

            # 5. Mark this section as seen
            if result.section_title is not None:
                seen_sections.add(section_key)

            citation_id = f"S{len(evidence) + 1}"

            evidence.append(
                Evidence(
                    citation_id=citation_id,
                    chunk_id=result.chunk_id,
                    content=result.content,
                    document_title=document_title,
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
            if len(evidence) >= max_evidence:
                break

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