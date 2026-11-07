
"""Verify LangChain metadata persistence in Qdrant."""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)
from qdrant_client import QdrantClient, models

from technical_rag.config import V2_PROFILE, SPARSE_MODEL_ID

embeddings = HuggingFaceEmbeddings(
    model_name=V2_PROFILE.embedding_model_id,
    model_kwargs={"revision": V2_PROFILE.embedding_revision},
    encode_kwargs={"normalize_embeddings": True},
)

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="v3_metadata_probe",
    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    },
)

store = QdrantVectorStore(
    client=client,
    collection_name="v3_metadata_probe",
    embedding=embeddings,
    sparse_embedding=FastEmbedSparse(model_name=SPARSE_MODEL_ID),
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="bm25",
)

document = Document(
    page_content="Metrics\n\nQdrant supports cosine similarity.",
    metadata={
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "section_title": "Metrics",
        "chunk_index": 0,
        "original_content": "Qdrant supports cosine similarity.",
    },
)

store.add_documents([document])

results = store.similarity_search(
    "Which similarity metric does Qdrant support?",
    k=1,
)

assert len(results) == 1
assert results[0].metadata["chunk_id"] == "chunk-1"
assert results[0].metadata["document_id"] == "doc-1"
assert results[0].metadata["original_content"] == document.metadata["original_content"]

print("Retrieved content:", results[0].page_content)
print("Retrieved metadata:", results[0].metadata)
print("\nMetadata compatibility: PASS")
