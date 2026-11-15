"""Configure the V3 LangChain chat model."""

from langchain_ollama import ChatOllama

from technical_rag.config import LLM_MODEL_ID


def build_v3_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL_ID,
        temperature=0,
        num_predict=256,
        num_ctx=2048,
        reasoning=False,
    )
