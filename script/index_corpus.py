import json
from pathlib import Path

from technical_rag.chunking import chunk_document
from technical_rag.embeddings import EmbeddingModel
from technical_rag.ingestion.markdown import load_markdown
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
    file_path = PROJECT_ROOT / source["file_path"]

    document = load_markdown(
        file_path=str(file_path),
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

    print(
        f"{document.title}: "
        f"{len(chunks)} chunks"
    )


print(f"\nTotal chunks: {len(all_chunks)}")


embedding_model = EmbeddingModel(MODEL_NAME)

embeddings = embedding_model.embed_documents(all_chunks)


vector_store = QdrantVectorStore(
    url="http://localhost:6333",
    collection_name=COLLECTION_NAME,
)

vector_store.create_collection(
    vector_size=embedding_model.dimension,
)

vector_store.upsert_chunks(
    chunks=all_chunks,
    embeddings=embeddings,
)

print(
    f"Indexed {len(all_chunks)} chunks "
    f"into '{COLLECTION_NAME}'."
)