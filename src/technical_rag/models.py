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