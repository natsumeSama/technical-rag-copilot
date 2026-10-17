"""Run the Technical RAG Copilot's generation benchmark locally.

Usage: uv run python script/evaluate_generation.py
"""

import json
from pathlib import Path
from time import perf_counter

from technical_rag.chunking import chunk_document
from technical_rag.context import ContextBuilder
from technical_rag.embeddings import EmbeddingModel
from technical_rag.generation.citations import (
    CitationGroundingValidator,
    CitationValidator,
)
from technical_rag.generation.llm import OllamaLLM
from technical_rag.generation.prompt import RAGPromptBuilder
from technical_rag.ingestion.markdown import load_markdown
from technical_rag.retrieval.bm25 import BM25Retriever
from technical_rag.retrieval.hybrid import HybridRetriever
from technical_rag.vector_store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"
EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "generation_eval.json"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "results" / "generation_results.json"
)

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
LLM_MODEL = "qwen3:1.7b"
COLLECTION_NAME = "technical_docs_v1"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_chunks():
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    chunks = []

    for source in sources:
        document = load_markdown(
            file_path=str(PROJECT_ROOT / source["file_path"]),
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
    return chunks


def save_results(results):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    if not EVAL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing evaluation dataset: {EVAL_PATH}. "
            "Create generation_eval.json before running this script."
        )

    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("The evaluation dataset is empty.")

    # Initialize everything once; reuse it for all questions.
    bm25 = BM25Retriever()
    bm25.index(load_chunks())

    retriever = HybridRetriever(
        embedding_model=EmbeddingModel(EMBEDDING_MODEL),
        vector_store=QdrantVectorStore(
            url="http://localhost:6333",
            collection_name=COLLECTION_NAME,
        ),
        bm25_retriever=bm25,
    )
    context_builder = ContextBuilder()
    prompt_builder = RAGPromptBuilder()
    citation_validator = CitationValidator()
    grounding_validator = CitationGroundingValidator(model_name=NLI_MODEL)
    llm = OllamaLLM(model=LLM_MODEL)

    results = []
    for index, case in enumerate(cases, start=1):
        question = case["question"]
        print(f"\n[{index}/{len(cases)}] {case['id']}: {question}", flush=True)
        started = perf_counter()

        search_results = retriever.search(
            query=question,
            limit=6,
            candidate_limit=10,
        )
        evidence = context_builder.build_evidence(
            search_results,
            max_evidence=3,
        )
        retrieval_s = perf_counter() - started

        system_prompt, user_prompt = prompt_builder.build(
            question=question,
            context=context_builder.format_context(evidence),
        )
        generated_at = perf_counter()
        answer = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        generation_s = perf_counter() - generated_at

        validated_at = perf_counter()
        valid_citations, invalid_citations = citation_validator.validate(
            answer=answer,
            evidence=evidence,
        )
        uncited_statements = citation_validator.extract_uncited_statements(answer)
        grounding_results = grounding_validator.validate(
            answer=answer,
            evidence=evidence,
        )
        validation_s = perf_counter() - validated_at

        record = {
            "id": case["id"],
            "category": case["category"],
            "question": question,
            "expected_behavior": case["expected_behavior"],
            "expected_facts": case.get("expected_facts", []),
            "answer": answer,
            "evidence": [item.model_dump() for item in evidence],
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "uncited_statements": uncited_statements,
            "grounding": [item.model_dump() for item in grounding_results],
            "timing_seconds": {
                "retrieval": round(retrieval_s, 3),
                "generation": round(generation_s, 3),
                "validation": round(validation_s, 3),
                "total": round(perf_counter() - started, 3),
            },
            "manual_review": {
                "behavior_correct": None,
                "answer_correct": None,
                "notes": "",
            },
        }
        results.append(record)
        save_results(results)  # Save after every question, even if a later run fails.

        print(f"Answer: {answer}")
        print(f"Invalid citations: {invalid_citations}")
        print(f"Uncited statements: {len(uncited_statements)}")
        print(f"NLI comparisons: {len(grounding_results)}")
        print(f"Total time: {record['timing_seconds']['total']:.2f}s")

    print(f"\nSaved {len(results)} cases to {OUTPUT_PATH}")
    print("Review answer correctness and abstentions manually before scoring.")


if __name__ == "__main__":
    main()
