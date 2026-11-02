from technical_rag.vector_store import QdrantVectorStore


vector_store = QdrantVectorStore(
    url="http://localhost:6333",
    collection_name="technical_docs_v1",
)

points, _ = vector_store.client.scroll(
    collection_name="technical_docs_v1",
    limit=1000,
    with_payload=True,
    with_vectors=False,
)

sections = set()

for point in points:
    payload = point.payload or {}

    sections.add(
        (
            payload.get("source_name"),
            payload.get("document_title"),
            payload.get("section_title"),
        )
    )

for source, document, section in sorted(
    sections,
    key=lambda item: (
        str(item[0]),
        str(item[1]),
        str(item[2]),
    ),
):
    print(
        f"{source} | {document} | {section}"
    )