
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .factory import build_pipeline
from .models import RAGResult

from pathlib import Path
from fastapi.responses import FileResponse

HTML_FILE = Path(__file__).resolve().parent / "static" / "index.html"

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the RAG once when the server starts
    app.state.rag = build_pipeline(enable_grounding=True)

    yield

    # Release our reference during shutdown
    del app.state.rag


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

    rag = request.app.state.rag

    return rag.ask(question)
