
"""Build the configured RAG pipeline."""

from qdrant_client import QdrantClient

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HYBRID_PROFILE,
    LLM_MODEL_ID,
    NLI_MODEL_ID,
    PROJECT_ROOT,
    QDRANT_URL,
    SOURCES_PATH,
    get_pipeline_profile,
)
from .embeddings import EmbeddingModel
from .generation.citations import CitationGroundingValidator
from .generation.llm import OllamaLLM
from .ingestion.corpus import load_chunks
from .pipeline import RAGPipeline
from .retrieval.bm25 import BM25Retriever
from .retrieval.hybrid import HybridRetriever
from .retrieval.qdrant_hybrid import QdrantHybridRetriever
from .retrieval.sparse_embeddings import BM25SparseEncoder
from .vector_store import QdrantVectorStore


def build_pipeline(
    enable_grounding: bool = True,
) -> RAGPipeline:

    profile = get_pipeline_profile()

    if profile == HYBRID_PROFILE:
        # Native Qdrant hybrid retrieval
        client = QdrantClient(url=QDRANT_URL)

        if not client.collection_exists(profile.collection_name):
            raise RuntimeError(
                "Hybrid collection does not exist. "
                "Run bootstrap_hybrid.py first."
            )

        count = client.count(
            collection_name=profile.collection_name,
            exact=True,
        ).count

        if count == 0:
            raise RuntimeError(
                "Hybrid collection is empty. "
                "Run index_hybrid.py first."
            )

        embedding_model = EmbeddingModel(
            profile.embedding_model_id,
            revision=profile.embedding_revision,
        )

        retriever = QdrantHybridRetriever(
            client=client,
            collection_name=profile.collection_name,
            dense_model=embedding_model,
            sparse_encoder=BM25SparseEncoder(),
        )

    else:
        # Existing V1 / V2 retrieval baseline
        chunks = load_chunks(
            sources_path=SOURCES_PATH,
            project_root=PROJECT_ROOT,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        bm25 = BM25Retriever()
        bm25.index(chunks)

        embedding_model = EmbeddingModel(
            profile.embedding_model_id,
            revision=profile.embedding_revision,
        )

        vector_store = QdrantVectorStore(
            url=QDRANT_URL,
            collection_name=profile.collection_name,
        )

        retriever = HybridRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            bm25_retriever=bm25,
        )

    llm = OllamaLLM(model=LLM_MODEL_ID)

    grounding_validator = None

    if enable_grounding:
        grounding_validator = CitationGroundingValidator(
            model_name=NLI_MODEL_ID,
        )

    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        grounding_validator=grounding_validator,
    )
