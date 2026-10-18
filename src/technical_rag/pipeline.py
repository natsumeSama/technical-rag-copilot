
from .context import ContextBuilder
from .generation.citations import (
    CitationGroundingValidator,
    CitationValidator,
)
from .generation.llm import OllamaLLM
from .generation.prompt import RAGPromptBuilder
from .models import RAGResult
from .retrieval.hybrid import HybridRetriever
from time import perf_counter

class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: OllamaLLM,
        grounding_validator: CitationGroundingValidator | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.grounding_validator = grounding_validator

        self.context_builder = ContextBuilder()
        self.prompt_builder = RAGPromptBuilder()
        self.citation_validator = CitationValidator()

    def ask(self, question: str) -> RAGResult:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        started = perf_counter()
        # 1. Retrieve relevant chunks
        results = self.retriever.search(
            query=question,
            limit=6,
            candidate_limit=10,
        )

        # 2. Select evidence
        evidence = self.context_builder.build_evidence(
            results,
            max_evidence=3,
        )
        retrieval_s = perf_counter() - started
        # 3. Build the LLM prompt
        context = self.context_builder.format_context(evidence)

        system_prompt, user_prompt = self.prompt_builder.build(
            question=question,
            context=context,
        )
        generation_started = perf_counter()
        # 4. Generate the answer
        answer = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        generation_s = perf_counter() - generation_started
        validation_started = perf_counter()

        # 5. Validate citations
        valid_citations, invalid_citations = (
            self.citation_validator.validate(
                answer=answer,
                evidence=evidence,
            )
        )

        uncited_statements = (
            self.citation_validator.extract_uncited_statements(
                answer
            )
        )

        # 6. Optional NLI grounding assessment
        grounding = []

        if self.grounding_validator is not None:
            grounding = self.grounding_validator.validate(
                answer=answer,
                evidence=evidence,
            )
        validation_s = perf_counter() - validation_started
        # 7. Return structured result
        return RAGResult(
            question=question,
            answer=answer,
            evidence=evidence,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            uncited_statements=uncited_statements,
            grounding=grounding,
            timing_seconds={
                "retrieval": round(retrieval_s, 3),
                "generation": round(generation_s, 3),
                "validation": round(validation_s, 3),
                "total": round(perf_counter() - started, 3),
            },
        )
