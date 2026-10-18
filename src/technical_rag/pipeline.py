
from .context import ContextBuilder
from .generation.citations import (
    CitationGroundingValidator,
    CitationValidator,
)
from .generation.llm import OllamaLLM
from .generation.prompt import RAGPromptBuilder
from .models import RAGResult
from .retrieval.hybrid import HybridRetriever


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

        # 3. Build the LLM prompt
        context = self.context_builder.format_context(evidence)

        system_prompt, user_prompt = self.prompt_builder.build(
            question=question,
            context=context,
        )

        # 4. Generate the answer
        answer = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

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

        # 7. Return structured result
        return RAGResult(
            question=question,
            answer=answer,
            evidence=evidence,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            uncited_statements=uncited_statements,
            grounding=grounding,
        )
