
"""Build the RAG pipeline from shared configuration."""

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
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
from .vector_store import QdrantVectorStore


def build_pipeline(
    enable_grounding: bool = True,
) -> RAGPipeline:
    """Initialize the configured RAG pipeline."""

    # 1. Select the configuration
    profile = get_pipeline_profile()

    # 2. Load document chunks
    all_chunks = load_chunks(
        sources_path=SOURCES_PATH,
        project_root=PROJECT_ROOT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # 3. Build BM25 retrieval
    bm25 = BM25Retriever()
    bm25.index(all_chunks)

    # 4. Initialize dense retrieval
    embedding_model = EmbeddingModel(
        profile.embedding_model_id,
        revision=profile.embedding_revision,
    )

    vector_store = QdrantVectorStore(
        url=QDRANT_URL,
        collection_name=profile.collection_name,
    )

    # 5. Combine dense and BM25 retrieval
    retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25,
    )

    # 6. Initialize the LLM
    llm = OllamaLLM(model=LLM_MODEL_ID)

    # 7. Optional grounding validation
    grounding_validator = None

    if enable_grounding:
        grounding_validator = CitationGroundingValidator(
            model_name=NLI_MODEL_ID,
        )

    # 8. Assemble the pipeline
    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        grounding_validator=grounding_validator,
    )
