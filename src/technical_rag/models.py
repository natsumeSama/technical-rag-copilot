from pydantic import BaseModel

class Document(BaseModel):
    id : str
    content_hash : str
    title : str
    content : str
    source_name : str
    source_url : str
    document_type : str
    metadata: dict[str, object]

class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    chunk_index: int
    section_title: str | None
    metadata: dict[str, object]

class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    section_title: str | None = None
    score: float
    metadata: dict[str, object]

class Evidence(BaseModel):
    citation_id: str
    chunk_id: str
    content: str
    document_title: str
    section_title: str | None
    source_name: str
    source_url: str
    retrieval_score: float

class CitationSupportResult(BaseModel):
    claim: str
    citation_id: str
    label: str
    supported: bool
    entailment_score: float
    neutral_score: float
    contradiction_score: float


class RAGResult(BaseModel):
    question: str
    answer: str
    evidence: list[Evidence]
    valid_citations: list[str]
    invalid_citations: list[str]
    uncited_statements: list[str]
    grounding: list[CitationSupportResult]
    timing_seconds: dict[str, float]
