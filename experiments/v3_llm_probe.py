
"""Verify LangChain integration with our local Ollama model."""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from technical_rag.config import LLM_MODEL_ID
from technical_rag.generation.prompt import RAGPromptBuilder


# 1. Configure the same model and settings as V2
llm = ChatOllama(
    model=LLM_MODEL_ID,
    temperature=0,
    num_predict=256,
    num_ctx=2048,
    reasoning=False,
)

# 2. Reuse our existing RAG prompt
system_prompt, user_prompt = RAGPromptBuilder().build(
    question="Which similarity metric does Qdrant support?",
    context="[S1] Qdrant supports cosine similarity.",
)

# 3. Send messages to the model
response = llm.invoke([
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt),
])

# 4. Inspect the result
assert isinstance(response.content, str)
assert response.content.strip()

print("\nANSWER:")
print(response.content)

print("\nMODEL METADATA:")
print("Model:", response.response_metadata.get("model"))
print("Generated tokens:", response.response_metadata.get("eval_count"))
print("Generation duration:", response.response_metadata.get("eval_duration"))
