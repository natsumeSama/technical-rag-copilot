import json
from pathlib import Path

from technical_rag.chunking import chunk_document
from technical_rag.embeddings import EmbeddingModel
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.retrieval.bm25 import BM25Retriever
from technical_rag.retrieval.hybrid import HybridRetriever
from technical_rag.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "technical_docs_v1"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# Build BM25 corpus
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


bm25_retriever = BM25Retriever()
bm25_retriever.index(all_chunks)


# Dense components
embedding_model = EmbeddingModel(MODEL_NAME)

vector_store = QdrantVectorStore(
    url="http://localhost:6333",
    collection_name=COLLECTION_NAME,
)


# Hybrid
hybrid_retriever = HybridRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
    bm25_retriever=bm25_retriever,
)


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

    results = hybrid_retriever.search(
        query=query,
        limit=3,
        candidate_limit=10,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"RRF score: {result.score:.6f}")
        print(
            f"Dense rank: "
            f"{result.metadata.get('dense_rank')}"
        )
        print(
            f"BM25 rank: "
            f"{result.metadata.get('bm25_rank')}"
        )
        print(f"Section: {result.section_title}")
        print(
            f"Document: "
            f"{result.metadata.get('document_title')}"
        )
        print(
            f"Content: "
            f"{result.content[:250]}..."
        )