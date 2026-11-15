"""Compare V2 Ollama and V3 ChatOllama behavior."""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from technical_rag.config import LLM_MODEL_ID
from technical_rag.generation.llm import OllamaLLM
from technical_rag.generation.prompt import RAGPromptBuilder
from technical_rag.generation.citations import CitationValidator

# Existing V2 integration
v2 = OllamaLLM(model=LLM_MODEL_ID)

# New V3 integration
v3 = ChatOllama(
    model=LLM_MODEL_ID,
    temperature=0,
    num_predict=256,
    num_ctx=2048,
    reasoning=False,
)

prompt_builder = RAGPromptBuilder()
citation_validator = CitationValidator()

evidence = (
    "[S1] Qdrant supports cosine similarity, "
    "dot product, Euclidean distance, "
    "and Manhattan distance."
)

cases = [
    {
        "id": "A1",
        "question": "Which similarity metrics does Qdrant support?",
    },
    {
        "id": "U1",
        "question": "What was Qdrant's total revenue in 2025?",
    },
]

for case in cases:
    system_prompt, user_prompt = prompt_builder.build(
        question=case["question"],
        context=evidence,
    )

    # V2 generation
    old_answer = v2.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # V3 generation
    response = v3.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    new_answer = response.content

    assert isinstance(new_answer, str)
    assert new_answer.strip()

    print(f"\n{'=' * 60}")
    print(f"CASE: {case['id']}")
    print(f"QUESTION: {case['question']}")

    for name, answer in [
        ("V2", old_answer),
        ("V3", new_answer),
    ]:
        print(f"\n{name} ANSWER:")
        print(answer)
        print(
            "Citations:",
            citation_validator.extract_citations(answer),
        )
