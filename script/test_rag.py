import json
from pathlib import Path

from technical_rag.chunking import chunk_document
from technical_rag.context import ContextBuilder
from technical_rag.embeddings import EmbeddingModel
from technical_rag.generation.llm import OllamaLLM
from technical_rag.generation.prompt import RAGPromptBuilder
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.retrieval.bm25 import BM25Retriever
from technical_rag.retrieval.hybrid import HybridRetriever
from technical_rag.vector_store import QdrantVectorStore
from technical_rag.generation.citations import (
    CitationGroundingValidator,
    CitationValidator,
)
from technical_rag.pipeline import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
LLM_MODEL = "qwen3:1.7b"

COLLECTION_NAME = "technical_docs_v1"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150



# --------------------------------------------------
# Load corpus for BM25
# --------------------------------------------------

with open(SOURCES_PATH, "r", encoding="utf-8") as file:
    sources = json.load(file)


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


# --------------------------------------------------
# Build retrieval
# --------------------------------------------------

bm25_retriever = BM25Retriever()
bm25_retriever.index(all_chunks)

embedding_model = EmbeddingModel(EMBEDDING_MODEL)

vector_store = QdrantVectorStore(
    url="http://localhost:6333",
    collection_name=COLLECTION_NAME,
)

retriever = HybridRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
    bm25_retriever=bm25_retriever,
)


# --------------------------------------------------
# Generation components
# --------------------------------------------------

# Initialize the RAG pipeline
rag = RAGPipeline(
    retriever=retriever,
    llm=OllamaLLM(model=LLM_MODEL),
    grounding_validator=CitationGroundingValidator(
        model_name=NLI_MODEL,
    ),
)

# Ask a question
result = rag.ask(
    "Which similarity metrics does Qdrant support?"
)

print("\nQUESTION")
print(result.question)

print("\nANSWER")
print(result.answer)

print("\nEVIDENCE")
for item in result.evidence:
    print(
        f"[{item.citation_id}] "
        f"{item.document_title} | {item.section_title}"
    )

print("\nCITATION VALIDATION")
print("Valid:", result.valid_citations)
print("Invalid:", result.invalid_citations)
print("Uncited:", result.uncited_statements)

print("\nNLI GROUNDING")
for item in result.grounding:
    print(
        f"[{item.citation_id}] "
        f"{item.claim} -> {item.label}"
    )
