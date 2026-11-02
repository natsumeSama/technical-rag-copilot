
from dataclasses import dataclass

from ..embedding_text import embedding_text
from ..models import Chunk


@dataclass
class ChunkDelta:
    unchanged: list[Chunk]
    to_upsert: list[Chunk]
    to_embed: list[Chunk]
    reuse_from: dict[str, str]
    to_delete: set[str]


def plan_chunk_delta(
    old_chunks: list[Chunk],
    new_chunks: list[Chunk],
) -> ChunkDelta:
    old_by_id = {chunk.id: chunk for chunk in old_chunks}
    new_by_id = {chunk.id: chunk for chunk in new_chunks}

    if len(old_by_id) != len(old_chunks):
        raise ValueError("Duplicate old chunk IDs")

    if len(new_by_id) != len(new_chunks):
        raise ValueError("Duplicate new chunk IDs")

    # Find existing embeddings by their exact input text.
    # This assumes the embedding configuration has not changed.
    old_embeddings = {
        embedding_text(chunk): chunk.id
        for chunk in old_chunks
    }

    unchanged = []
    to_upsert = []
    to_embed = []
    reuse_from = {}

    for chunk in new_chunks:
        previous = old_by_id.get(chunk.id)

        if previous == chunk:
            unchanged.append(chunk)
            continue

        to_upsert.append(chunk)

        source_id = old_embeddings.get(embedding_text(chunk))

        if source_id is not None:
            reuse_from[chunk.id] = source_id
        else:
            to_embed.append(chunk)

    to_delete = set(old_by_id) - set(new_by_id)

    return ChunkDelta(
        unchanged=unchanged,
        to_upsert=to_upsert,
        to_embed=to_embed,
        reuse_from=reuse_from,
        to_delete=to_delete,
    )
