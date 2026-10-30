import json
from pathlib import Path

from technical_rag.chunking import chunk_document
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.retrieval.bm25 import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


with open(SOURCES_PATH, "r", encoding="utf-8") as file:
    sources = json.load(file)


all_chunks = []

for source in sources:
    document = load_markdown(
        file_path=str(PROJECT_ROOT / source["file_path"]),
        source_name=source["source_name"],
        source_url=source["source_url"],
        document_type=source["document_type"],
        metadata=source["metadata"],
    )

    chunks = chunk_document(
        document=document,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks.extend(chunks)


print(f"BM25 corpus: {len(all_chunks)} chunks")


retriever = BM25Retriever()
retriever.index(all_chunks)


queries = [
    "How does Qdrant filtering work?",
    "What is hybrid search?",
    "What is the difference between retrieval and reranking?",
    "How does semantic search work with Sentence Transformers?",
    "What distance metrics can Qdrant use?",
]


for query in queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")

    results = retriever.search(
        query=query,
        limit=3,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Section: {result.section_title}")
        print(
            f"Document: "
            f"{result.metadata.get('document_title')}"
        )
        print(
            f"Source: "
            f"{result.metadata.get('source_name')}"
        )
        print(
            f"Content: "
            f"{result.content[:250]}..."
        )