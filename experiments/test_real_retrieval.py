from technical_rag.embeddings import EmbeddingModel
from technical_rag.vector_store import QdrantVectorStore


MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "technical_docs_v1"


embedding_model = EmbeddingModel(MODEL_NAME)

vector_store = QdrantVectorStore(
    url="http://localhost:6333",
    collection_name=COLLECTION_NAME,
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

    query_embedding = embedding_model.embed_query(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        limit=3,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Section: {result.section_title}")
        print(f"Document: {result.metadata.get('document_title')}")
        print(f"Source: {result.metadata.get('source_name')}")
        print(f"URL: {result.metadata.get('source_url')}")
        print(f"Content: {result.content[:250]}...")