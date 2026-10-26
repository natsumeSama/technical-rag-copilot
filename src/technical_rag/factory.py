

import os

from .embeddings import EmbeddingModel
from .generation.citations import CitationGroundingValidator
from .generation.llm import OllamaLLM
from .pipeline import RAGPipeline
from .retrieval.bm25 import BM25Retriever
from .retrieval.hybrid import HybridRetriever
from .vector_store import QdrantVectorStore
from .config_v2 import (
    MODEL_REVISION,
    PROJECT_ROOT,
    SOURCES_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    QDRANT_URL,
)
from .ingestion.corpus import load_chunks

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
LLM_MODEL = "qwen3:1.7b"

COLLECTION_NAME = os.getenv(
    "RAG_COLLECTION",
    "technical_docs_v1",
)


def build_pipeline(
    enable_grounding: bool = True,
) -> RAGPipeline:
    """Initialize all RAG components and return a ready-to-use pipeline."""

    all_chunks = load_chunks(
        sources_path=SOURCES_PATH,
        project_root=PROJECT_ROOT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    # 2. Build sparse retrieval
    bm25 = BM25Retriever()
    bm25.index(all_chunks)

    # 3. Initialize dense retrieval
    revision = (
        MODEL_REVISION
        if COLLECTION_NAME == "technical_docs_v2"
        else None
    )

    embedding_model = EmbeddingModel(
        EMBEDDING_MODEL,
        revision=revision,
    )
    vector_store = QdrantVectorStore(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    # 4. Build hybrid retrieval (RRF)
    retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_retriever=bm25,
    )

    # 5. Initialize LLM
    llm = OllamaLLM(model=LLM_MODEL)

    # 6. Optional NLI grounding
    grounding_validator = None

    if enable_grounding:
        grounding_validator = CitationGroundingValidator(
            model_name=NLI_MODEL,
        )

    # 7. Assemble the pipeline
    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        grounding_validator=grounding_validator,
    )
