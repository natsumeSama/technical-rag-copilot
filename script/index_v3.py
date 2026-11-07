
"""Initial V3 corpus indexing with LangChain."""

from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient, models

from technical_rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PROJECT_ROOT,
    QDRANT_URL,
    SOURCES_PATH,
    SPARSE_MODEL_ID,
    V2_PROFILE,
)
from technical_rag.embedding_text import embedding_text
from technical_rag.ingestion.corpus import load_chunks


COLLECTION = "technical_docs_v3_hybrid"


def main():
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION):
        raise RuntimeError(
            f"{COLLECTION} already exists; refusing to overwrite"
        )

    chunks = load_chunks(
        sources_path=SOURCES_PATH,
        project_root=PROJECT_ROOT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=V2_PROFILE.embedding_model_id,
        model_kwargs={
            "revision": V2_PROFILE.embedding_revision
        },
        encode_kwargs={"normalize_embeddings": True},
    )

    dimension = len(embeddings.embed_query("dimension check"))

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="metadata.document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
        sparse_embedding=FastEmbedSparse(
            model_name=SPARSE_MODEL_ID
        ),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="bm25",
    )

    documents = [
        Document(
            page_content=embedding_text(chunk),
            metadata={
                **chunk.metadata,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "original_content": chunk.content,
            },
        )
        for chunk in chunks
    ]

    ids = [
        str(uuid5(NAMESPACE_URL, chunk.id))
        for chunk in chunks
    ]

    store.add_documents(documents, ids=ids)

    count = client.count(
        collection_name=COLLECTION,
        exact=True,
    ).count

    assert count == len(chunks)

    results = store.similarity_search(
        "Which similarity metrics does Qdrant support?",
        k=3,
    )

    assert results
    assert all(
        doc.metadata.get("chunk_id")
        and doc.metadata.get("document_id")
        and doc.metadata.get("source_url")
        for doc in results
    )

    print(f"Indexed chunks: {count}")
    print("Hybrid retrieval: PASS")
    print("Citation metadata: PASS")


if __name__ == "__main__":
    main()
