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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
LLM_MODEL = "qwen3.5:4b"

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

context_builder = ContextBuilder()
prompt_builder = RAGPromptBuilder()

llm = OllamaLLM(
    model=LLM_MODEL,
)


# --------------------------------------------------
# Ask the RAG
# --------------------------------------------------

question = "Which similarity metrics are available in Qdrant?"

results = retriever.search(
    query=question,
    limit=3,
    candidate_limit=10,
)

evidence = context_builder.build_evidence(results)

context = context_builder.format_context(evidence)

system_prompt, user_prompt = prompt_builder.build(
    question=question,
    context=context,
)

answer = llm.generate(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
)


# --------------------------------------------------
# Display
# --------------------------------------------------

print("\n" + "=" * 80)
print("QUESTION")
print("=" * 80)
print(question)

print("\n" + "=" * 80)
print("RETRIEVED EVIDENCE")
print("=" * 80)

for item in evidence:
    print(
        f"[{item.citation_id}] "
        f"{item.document_title} | "
        f"{item.section_title}"
    )

print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)
print(answer)