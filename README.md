# Technical RAG Copilot

An experimental retrieval-augmented generation (RAG) system for technical documentation, developed to study **AI engineering**: retrieval, indexing, evaluation, grounding, reproducibility, and the evolution of document ingestion pipelines.

**Project status:** V1 baseline implemented; V2 incremental indexing and an experimental Qdrant-native hybrid retriever implemented; unified retrieval evaluation, production-style hybrid synchronization, and MLOps/agentic ingestion are planned.

This repository is an engineering learning project, **not** a production-ready document assistant.

## 1. Architecture

The current question-answering pipeline is:

```text
Question
   |
   v
Dense retrieval (Sentence Transformers + Qdrant)
        +
Local BM25 retrieval
   |
   v
Reciprocal Rank Fusion (Python)
   |
   v
Evidence selection (up to 3 passages)
   |
   v
Prompt construction -> Ollama (Qwen3 1.7B)
   |
   v
Citation checks + optional NLI grounding assessment
   |
   v
Structured answer, evidence, diagnostics, timings
```

The experimental Qdrant-native retriever is a **separate path**:

```text
Question
   |-- dense query embedding ----> Qdrant named vector: dense
   |-- sparse BM25 encoding -----> Qdrant sparse vector: bm25
                                      |
                                      v
                               Qdrant RRF fusion
                                      |
                                      v
                                 Ranked chunks
```

The main `factory.py` still constructs the original Python-based hybrid retriever. Selecting `technical_docs_hybrid_exp` through `RAG_COLLECTION` is intentionally unsupported until the native retriever has been integrated and evaluated.

## 2. Implemented components

| Area | Current implementation |
| --- | --- |
| Document ingestion | Markdown files listed in `data/sources.json`; shared document/chunk loaders |
| Chunking | Header-aware Markdown splitting and recursive splitting (configured at 1,000 characters, 150 overlap) |
| Dense embeddings | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| Dense index | Qdrant |
| Lexical retrieval | Local BM25 for the existing pipeline |
| Hybrid baseline | Python reciprocal rank fusion (RRF) of dense and BM25 rankings |
| Experimental hybrid | Qdrant named dense vectors and FastEmbed `Qdrant/bm25` sparse vectors, fused with Qdrant RRF |
| Generation | Local Ollama model `qwen3:1.7b` |
| Grounding | Citation validation and optional NLI-based grounding assessment |
| Incremental V2 indexing | SQLite document registry, index signatures, chunk-level delta planning, compatible embedding reuse, Qdrant verification |
| Testing | `pytest` unit tests; several older manual evaluation scripts |

### Collection profiles

Configuration lives in `src/technical_rag/config.py`:

| Profile | Qdrant collection | Notes |
| --- | --- | --- |
| V1 | `technical_docs_v1` | Historical baseline; embedding revision is not pinned |
| V2 | `technical_docs_v2` | Pinned dense model revision; incremental synchronizer |
| Hybrid experiment | `technical_docs_hybrid_exp` | Separate dense+sparse index; not selected by the main factory |

`config_v2.py` currently provides backward-compatible names for scripts not yet migrated to the new configuration API.

## 3. Repository layout

```text
src/technical_rag/
  config.py               # Shared settings and collection profiles
  factory.py              # Current end-to-end pipeline assembly
  pipeline.py             # Retrieval -> evidence -> generation -> validation
  ingestion/              # Markdown loading, registry, deltas, vector reuse, sync
  retrieval/              # BM25, Python RRF, Qdrant-native hybrid, reranking
  generation/             # LLM, prompts, citations and grounding
  vector_store.py         # Dense Qdrant operations

script/                    # Active indexing, diagnostics and evaluation commands
experiments/               # Archived exploratory scripts (not automated tests)
tests/                     # Automated pytest tests
data/sources.json          # Corpus source manifest
data/evaluation/           # Manually curated evaluation cases
```

The files under `experiments/` are historical and may use older configurations or perform side effects; they should not be treated as maintained operational commands.

## 4. Local setup

**Requirements:** Python 3.13+, [`uv`](https://docs.astral.sh/uv/), Docker with Qdrant, and Ollama. Local model downloads and document files are required. All commands below run from the repository root.

Install dependencies:

```bash
uv sync
```

Start a local Qdrant server if one is not already running:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Qdrant is configured at `http://localhost:6333`. If you already have a Qdrant container, use it rather than creating a duplicate. The example stores its data in a Docker-managed volume.

Install the configured Ollama model:

```bash
ollama pull qwen3:1.7b
```

Ensure Ollama is running (`ollama serve` if your installation does not already run it as a service).

### Corpus requirements

The manifest `data/sources.json` references local Markdown files under `data/raw/`. **The raw documentation files are not committed to Git**, so a fresh checkout is not immediately ready for indexing. Populate the paths declared by the manifest before running corpus-dependent scripts.

This repository does not yet have an automated, verified document acquisition pipeline; this is part of the planned agentic-ingestion work.

## 5. V2 indexing workflow

The dense V2 collection must be created **once**:

```bash
uv run python script/bootstrap_v2.py
```

**Do not rerun the bootstrap on an existing collection.** It intentionally refuses to overwrite `technical_docs_v2`.

Inspect content-based changes:

```bash
uv run python script/scan_corpus.py
```

`scan_corpus.py` compares document content hashes only. It does **not** detect changes to metadata, embedding models or chunking settings. The synchronization workflow uses a richer index signature.

Synchronize the V2 corpus incrementally:

```bash
uv run python script/sync_corpus.py
```

The synchronizer:

1. Loads the current source manifest and documents.
2. Uses index signatures to skip unchanged documents.
3. Computes chunk deltas for changed documents.
4. Reuses compatible stored dense embeddings when possible.
5. Upserts/deletes points in Qdrant, verifies the result, then updates SQLite state.

The local SQLite registry is stored under `data/state/` and is not committed. Its state is part of a local deployment and should not be mistaken for a complete remote source of truth.

**Important limitation:** `sync_corpus.py` updates the dense V2 collection only. It does not automatically synchronize `technical_docs_hybrid_exp`; that experimental index can become stale after a document update.

## 6. Experimental native hybrid retrieval

The following commands require the V2 dense collection to exist and the corpus to be available.

Create the experimental hybrid collection **once**:

```bash
uv run python script/bootstrap_hybrid.py
```

Populate/verify its dense and sparse vectors:

```bash
uv run python script/index_hybrid.py
```

Compare query-time dense, sparse and RRF-fused rankings:

```bash
uv run python script/compare_hybrid.py --query "Which similarity metrics does Qdrant support?"
```

This comparison prints example rankings; **it is not a statistically meaningful benchmark**. Its hybrid scores are fusion scores, not calibrated answer probabilities.

## 7. Run the RAG pipeline

The main pipeline defaults to the V1 collection (`technical_docs_v1`). To use the V2 dense collection with the **existing Python-based hybrid retriever**, select V2 before launching the pipeline.

PowerShell:

```powershell
$env:RAG_COLLECTION = "technical_docs_v2"
uv run python script/smoke_rag.py
```

Windows Command Prompt:

```bat
set RAG_COLLECTION=technical_docs_v2
uv run python script/smoke_rag.py
```

Alternatively, start the FastAPI application:

```bash
uv run uvicorn technical_rag.api:app --reload
```

Its `/ask` endpoint returns an answer, retrieved evidence, citation diagnostics, optional NLI grounding results and stage timings. The API is a development interface, not a hardened service.

## 8. Evaluation and testing

Run automated tests:

```bash
uv run python -m pytest -q
```

The existing evaluation files include:

- `data/evaluation/retrieval_eval.json`: 20 question/target annotations for retrieval evaluation.
- `data/evaluation/generation_eval.json`: cases for generated-answer review.
- `data/evaluation/grounding_eval.json`: evidence/claim pairs for NLI assessment.

Legacy evaluation scripts remain under `script/evaluate_*.py`. They are useful for historical experiments but are **not yet one standardized benchmark**. The unit tests do not by themselves establish real-world answer accuracy, corpus freshness or end-to-end reliability under a running Qdrant/Ollama stack.

### Evaluation principles for the next milestone

To compare methods fairly, we should keep the corpus, queries, target labels, model versions and candidate budgets fixed. Planned retrieval metrics include Hit@k/Recall@k and MRR, with per-question failure analysis and latency reporting. Any claims of improvement should be tied to reproducible evaluation results rather than a handful of example queries.

## 9. Known limitations

- The current main pipeline does not yet use the Qdrant-native hybrid retriever.
- V2 document sync does not propagate updates to the experimental hybrid collection.
- A new checkout needs local raw documentation files before indexing or corpus-based inference can run.
- Some historical evaluation and experiment scripts retain older configuration assumptions.
- V1 does not pin an embedding revision, limiting exact reproducibility of historical runs.
- No unified experiment-tracking or continuous evaluation workflow exists yet.
- No automated agentic discovery, source verification, approval or refresh workflow exists yet.
- Citation/NLI checks are useful diagnostics but do not guarantee factual correctness.

## 10. Roadmap

1. **Unified retrieval benchmark:** one evaluation runner and comparable reports for dense, BM25, Python RRF, Qdrant RRF and reranking.
2. **Hybrid integration:** select retrieval strategy through configuration; synchronize both indexes safely; test changes and deletions.
3. **AI engineering / LLMOps:** reproducible runs, dataset and index versioning, CI regression checks, observability and latency analysis.
4. **Agentic ingestion:** controlled source discovery and fetching, trust/quality checks, deduplication, approval and incremental updates.

The focus is to learn how to **measure and maintain** a RAG system, not merely make a chatbot interface.
