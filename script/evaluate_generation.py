
"""Evaluate the complete RAG pipeline."""

import json
from pathlib import Path

from technical_rag.chunking import chunk_document
from technical_rag.embeddings import EmbeddingModel
from technical_rag.generation.citations import CitationGroundingValidator
from technical_rag.generation.llm import OllamaLLM
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.pipeline import RAGPipeline
from technical_rag.retrieval.bm25 import BM25Retriever
from technical_rag.retrieval.hybrid import HybridRetriever
from technical_rag.vector_store import QdrantVectorStore


ROOT = Path(__file__).resolve().parents[1]

SOURCES_PATH = ROOT / "data/sources.json"
EVAL_PATH = ROOT / "data/evaluation/generation_eval.json"
OUTPUT_PATH = (
    ROOT / "data/evaluation/results/generation_results_pipeline.json"
)

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
LLM_MODEL = "qwen3:1.7b"

COLLECTION_NAME = "technical_docs_v1"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def build_pipeline() -> RAGPipeline:
    """Initialize and return our RAG pipeline."""

    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    chunks = []

    for source in sources:
        document = load_markdown(
            file_path=str(ROOT / source["file_path"]),
            source_name=source["source_name"],
            source_url=source["source_url"],
            document_type=source["document_type"],
            metadata=source["metadata"],
        )

        chunks.extend(
            chunk_document(
                document=document,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        )

    bm25 = BM25Retriever()
    bm25.index(chunks)

    retriever = HybridRetriever(
        embedding_model=EmbeddingModel(EMBEDDING_MODEL),
        vector_store=QdrantVectorStore(
            url="http://localhost:6333",
            collection_name=COLLECTION_NAME,
        ),
        bm25_retriever=bm25,
    )

    return RAGPipeline(
        retriever=retriever,
        llm=OllamaLLM(model=LLM_MODEL),
        grounding_validator=CitationGroundingValidator(
            model_name=NLI_MODEL,
        ),
    )


def save_results(results: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    if not cases:
        raise ValueError("Evaluation dataset is empty")

    rag = build_pipeline()
    records = []

    for index, case in enumerate(cases, start=1):
        print(
            f"\n[{index}/{len(cases)}] "
            f"{case['id']}: {case['question']}"
        )

        # One method runs the complete RAG
        result = rag.ask(case["question"])

        record = {
            "id": case["id"],
            "category": case["category"],
            "expected_behavior": case["expected_behavior"],
            "expected_facts": case.get("expected_facts", []),
            **result.model_dump(),
            "manual_review": {
                "behavior_correct": None,
                "answer_correct": None,
                "notes": "",
            },
        }

        records.append(record)
        save_results(records)

        print(f"\nAnswer:\n{result.answer}")
        print(f"Valid citations: {result.valid_citations}")
        print(f"Invalid citations: {result.invalid_citations}")
        print(f"Uncited statements: {len(result.uncited_statements)}")
        print(f"NLI comparisons: {len(result.grounding)}")
        print(f"Total time: {result.timing_seconds['total']:.2f}s")

    print(f"\nSaved {len(records)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
