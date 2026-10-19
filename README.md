# Technical RAG Copilot — V1

An experimental, evidence-grounded **Retrieval-Augmented Generation (RAG)** system for answering questions about technical documentation. Built as a hands-on AI engineering project to explore document processing, hybrid search, LLM generation, citation validation, and evaluation.

**Status:** V1 baseline. The next iteration focuses on more reliable ingestion, advanced retrieval, agentic document acquisition, and LLMOps.

## What it does

Given a technical question, the system:

1. Retrieves relevant passages using **dense semantic search** (Sentence Transformers + Qdrant) and **sparse BM25 search**.
2. Combines their rankings with **Reciprocal Rank Fusion (RRF)**.
3. Selects evidence passages and assigns source IDs such as `[S1]` and `[S2]`.
4. Generates an answer with a local **Qwen3 1.7B** model through Ollama, prompting it to use only retrieved evidence.
5. Checks citation IDs and runs optional **NLI-based evidence support diagnostics**.
6. Returns a structured response, including evidence, citation diagnostics, and execution timings.

A **FastAPI** backend exposes the pipeline, and a lightweight HTML chat page offers a simple way to try it locally.

## Architecture

```text
Local Markdown documentation
           |
    Parsing + chunking
           |
      Text chunks
        /      \
       v        v
Sentence        BM25
Transformers     index
   + Qdrant       |
       \        /
        v      v
      Hybrid retrieval
         (RRF)
           |
    Evidence selection
           |
   Qwen3 via Ollama
           |
 Citation checks + optional NLI
           |
      RAG response
     (FastAPI / UI)
```

## Tech stack

| Layer | Technology |
| --- | --- |
| Language / models | Python, Pydantic |
| Document processing | `python-frontmatter`, LangChain text splitters |
| Dense retrieval | Sentence Transformers, Qdrant |
| Sparse retrieval | `bm25s`, PyStemmer |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Optional reranking experiments | CrossEncoder |
| Local generation | Ollama, `qwen3:1.7b` |
| Grounding diagnostics | NLI CrossEncoder |
| API and demo | FastAPI, HTML |
| Tests | pytest |

The current embedding model is `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`; the experimental reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`; and the NLI model is `cross-encoder/nli-MiniLM2-L6-H768`.

## Getting started

### Requirements

- **Python 3.13+** and [uv](https://docs.astral.sh/uv/)
- [Docker](https://www.docker.com/) for Qdrant
- [Ollama](https://ollama.com/) for local LLM inference
- Local copies of the documentation files listed in [`data/sources.json`](data/sources.json)

> **Important:** The raw Markdown corpus is intentionally excluded from Git (`data/raw/`). A fresh clone cannot index or launch the full pipeline until you place those files at the paths specified by `data/sources.json`. That manifest includes their original source URLs.

### 1. Install dependencies

```bash
git clone https://github.com/natsumeSama/technical-rag-copilot.git
cd technical-rag-copilot
uv sync
```

### 2. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

If the container already exists and is stopped, use `docker start qdrant` instead.

### 3. Prepare Ollama

```bash
ollama pull qwen3:1.7b
```

Make sure the Ollama service is running. The embedding and NLI models are downloaded on first use through Sentence Transformers.

### 4. Add the documentation corpus

Use [`data/sources.json`](data/sources.json) as the manifest. Download the seven referenced technical Markdown files from their source URLs and save each one to its matching `data/raw/...` path.

### 5. Build the Qdrant index

```bash
uv run python script/index_corpus.py
```

> **Warning:** In V1, this script **deletes and recreates** the Qdrant collection `technical_docs_v1`. Do not use it against a collection whose contents you need to preserve.

BM25 is rebuilt from the same local Markdown corpus when the application initializes.

### 6. Run the API and demo UI

```bash
uv run uvicorn technical_rag.api:app --reload
```

- Demo chat: http://127.0.0.1:8000/
- Interactive API docs: http://127.0.0.1:8000/docs
- Health endpoint: `GET /health`
- Question endpoint: `POST /ask`

Example request (PowerShell):

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/ask' -Method Post -ContentType 'application/json' -Body '{"question":"Which similarity metrics does Qdrant support?"}'
```

Or use the Python pipeline directly:

```python
from technical_rag.factory import build_pipeline

rag = build_pipeline()
result = rag.ask("Which similarity metrics does Qdrant support?")
print(result.answer)
```

## Evaluation and experiments

V1 includes scripts to evaluate **dense retrieval**, **BM25**, **hybrid RRF**, **cross-encoder reranking**, **NLI grounding diagnostics**, and the **end-to-end answer pipeline**. The evaluation datasets are stored in `data/evaluation/`.

```bash
# Tests that use mocked external components
uv run pytest

# Retrieval evaluation (requires the indexed corpus)
uv run python script/evaluate_hybrid.py
uv run python script/evaluate_reranker.py

# Generation evaluation (requires Qdrant and Ollama)
uv run python script/evaluate_generation.py
```

The retrieval scripts report **Hit@1, Hit@3, and Mean Reciprocal Rank (MRR)**. Early experiments favored the hybrid RRF baseline over the tested reranking configuration, but the benchmark is small. Generation experiments revealed cases where a syntactically valid citation did not actually support the corresponding claim. NLI judgments were also imperfect.

**These are exploratory results, not production-grade reliability claims.** The benchmark is too small to generalize performance, and the evaluation scripts should be rerun on a larger, reviewed dataset.

## Repository layout

```text
src/technical_rag/
  ingestion/          Markdown loading
  retrieval/          BM25, hybrid fusion, reranking
  generation/         Ollama, prompts, citation / NLI checks
  chunking.py         Header-aware chunking
  embeddings.py       Embedding interface
  vector_store.py     Qdrant integration
  context.py          Evidence selection and formatting
  pipeline.py         End-to-end RAG execution
  factory.py          Component wiring
  api.py              FastAPI endpoints
  static/index.html   Lightweight chat UI
script/                Indexing, experiments, evaluation
tests/                Automated tests
data/sources.json      Documentation manifest
data/evaluation/       Evaluation datasets
```

## Known V1 limitations

- **Static ingestion:** only manually prepared Markdown files; no PDF/HTML pipeline or automatic source updates.
- **Full Qdrant rebuilds:** no incremental indexing, persistent document registry, or coordinated versioning with BM25.
- **Small corpus and evaluations:** insufficient for reliable estimates of real-world performance.
- **Basic context selection:** duplicate-section filtering can omit useful passages.
- **Limited grounding verification:** citation syntax checks and NLI scores do not guarantee factual support.
- **Local-first operation:** no production deployment, continuous monitoring, or managed model serving.

## What comes next — V2

V2 will develop this baseline into a more capable **document intelligence system**, focusing on:

1. Versioned, incremental ingestion and reliable index synchronization.
2. Better retrieval and evaluation on harder, larger benchmarks.
3. Tool-using document discovery with human approval for updates.
4. Reproducible experiments, tracing, and automated checks (LLMOps/MLOps).

The goal is to learn and demonstrate AI engineering decisions through experiments and maintainable code—not to prioritize a polished frontend.
