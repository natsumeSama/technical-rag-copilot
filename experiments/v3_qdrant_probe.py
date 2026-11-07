
"""Read-only LangChain hybrid retrieval compatibility probe."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient

from technical_rag.config import (
    HYBRID_PROFILE,
    QDRANT_URL,
    SPARSE_MODEL_ID,
)

embeddings = HuggingFaceEmbeddings(
    model_name=HYBRID_PROFILE.embedding_model_id,
    model_kwargs={
        "revision": HYBRID_PROFILE.embedding_revision
    },
    encode_kwargs={"normalize_embeddings": True},
)

store = QdrantVectorStore(
    client=QdrantClient(url=QDRANT_URL),
    collection_name=HYBRID_PROFILE.collection_name,
    embedding=embeddings,
    sparse_embedding=FastEmbedSparse(
        model_name=SPARSE_MODEL_ID,
    ),
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="bm25",
    content_payload_key="content",
)

results = store.similarity_search_with_score(
    "Which similarity metrics does Qdrant support?",
    k=3,
)

assert results, "No documents retrieved"

for document, score in results:
    print(f"\nScore: {score:.4f}")
    print("Content:", document.page_content[:200])
    print("Metadata:", document.metadata)
