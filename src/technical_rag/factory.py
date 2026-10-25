
import json
import os

from .config_v2 import MODEL_REVISION
from pathlib import Path

from .chunking import chunk_document
from .embeddings import EmbeddingModel
from .generation.citations import CitationGroundingValidator
from .generation.llm import OllamaLLM
from .ingestion.markdown import load_markdown
from .pipeline import RAGPipeline
from .retrieval.bm25 import BM25Retriever
from .retrieval.hybrid import HybridRetriever
from .vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
LLM_MODEL = "qwen3:1.7b"

COLLECTION_NAME = os.getenv(
    "RAG_COLLECTION",
    "technical_docs_v1",
)

QDRANT_URL = "http://localhost:6333"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def build_pipeline(
    enable_grounding: bool = True,
) -> RAGPipeline:
    """Initialize all RAG components and return a ready-to-use pipeline."""

    # 1. Load documents and build chunks
    sources = json.loads(
        SOURCES_PATH.read_text(encoding="utf-8")
    )

    all_chunks = []

    for source in sources:
        document = load_markdown(
            file_path=str(PROJECT_ROOT / source["file_path"]),
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

    if not all_chunks:
        raise ValueError("No chunks found in the document corpus")

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
