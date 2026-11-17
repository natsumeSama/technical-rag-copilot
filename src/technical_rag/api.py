from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .models import RAGResult
from .generation.v3_llm import build_v3_llm
from .retrieval.v3_store import build_v3_store
from .rag.v3_pipeline import ask_v3

HTML_FILE = Path(__file__).resolve().parent / "static" / "index.html"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize V3 components once at startup
    app.state.store = build_v3_store()
    app.state.llm = build_v3_llm()

    yield

    # Release references on shutdown
    del app.state.store
    del app.state.llm


app = FastAPI(
    title="Technical RAG Copilot",
    description="Evidence-grounded technical question answering",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(HTML_FILE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=RAGResult)
def ask(payload: AskRequest, request: Request) -> RAGResult:
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be blank",
        )

    return ask_v3(
        question=question,
        store=request.app.state.store,
        llm=request.app.state.llm,
    )
