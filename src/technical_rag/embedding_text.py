
from .models import Chunk


def embedding_text(chunk: Chunk) -> str:
    if chunk.section_title:
        return f"{chunk.section_title}\n\n{chunk.content}"

    return chunk.content
