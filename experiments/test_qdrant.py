from pathlib import Path

from technical_rag.chunking import chunk_document
from technical_rag.embeddings import EmbeddingModel
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample.md"

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


document = load_markdown(
    file_path=str(SAMPLE_PATH),
    source_name="Local Sample",
    source_url="local://sample.md",
    document_type="documentation",
    metadata={"language": "en"},
)

chunks = chunk_document(
    document=document,
    chunk_size=200,
    chunk_overlap=30,
)

embedding_model = EmbeddingModel(MODEL_NAME)

embeddings = embedding_model.embed_documents(chunks)

vector_store = QdrantVectorStore(
    url="http://localhost:6333",
    collection_name="technical_docs",
)

vector_store.create_collection(
    vector_size=embedding_model.dimension,
)

vector_store.upsert_chunks(
    chunks=chunks,
    embeddings=embeddings,
)

print(f"Indexed {len(chunks)} chunks into Qdrant.")

query = "How does sparse retrieval work?"

query_embedding = embedding_model.embed_query(query)

results = vector_store.search(
    query_embedding=query_embedding,
    limit=4,
)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Section: {result.section_title}")
    print(result.content)
    print("-" * 60)