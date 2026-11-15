"""Configure the V3 LangChain Qdrant vector store."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient

from technical_rag.config import (
    QDRANT_URL,
    SPARSE_MODEL_ID,
    V2_PROFILE,
)

V3_COLLECTION = "technical_docs_v3_hybrid"


def build_v3_store() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL)

    if not client.collection_exists(V3_COLLECTION):
        raise RuntimeError(f"Missing V3 collection: {V3_COLLECTION}")

    embeddings = HuggingFaceEmbeddings(
        model_name=V2_PROFILE.embedding_model_id,
        model_kwargs={"revision": V2_PROFILE.embedding_revision},
        encode_kwargs={"normalize_embeddings": True},
    )

    return QdrantVectorStore(
        client=client,
        collection_name=V3_COLLECTION,
        embedding=embeddings,
        sparse_embedding=FastEmbedSparse(
            model_name=SPARSE_MODEL_ID,
        ),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="bm25",
    )
