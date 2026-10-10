import json
from pathlib import Path

from technical_rag.chunking import chunk_document
from technical_rag.context import ContextBuilder
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

embedding_model = EmbeddingModel(MODEL_NAME)

vector_store = QdrantVectorStore(
    url="http://localhost:6333",
    collection_name=COLLECTION_NAME,
)

hybrid_retriever = HybridRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
    bm25_retriever=bm25_retriever,
)

context_builder = ContextBuilder()


query = "Which similarity metrics are available in Qdrant?"

results = hybrid_retriever.search(
    query=query,
    limit=3,
    candidate_limit=10,
)

evidence = context_builder.build_evidence(results)

context = context_builder.format_context(evidence)


print("=" * 80)
print(f"QUERY: {query}")
print("=" * 80)

print(context)