
from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import numpy as np

from ..embedding_text import embedding_text
from ..models import Chunk
from ..vector_store import StoredChunk
from .chunk_delta import ChunkDelta


@dataclass(frozen=True)
class EmbeddingSpec:
    model_id: str
    revision: str
    dimension: int
    normalize: bool = True
    input_format: str = "section-title-v1"
    encode_method: str = "encode_document"

    def fingerprint(self) -> str:
        data = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(data.encode("utf-8")).hexdigest()


@dataclass
class PreparedEmbeddings:
    chunks: list[Chunk]
    vectors: np.ndarray
    skipped_ids: list[str]
    delete_ids: set[str]
    reused_count: int
    computed_count: int


def prepare_embeddings(
    delta: ChunkDelta,
    stored: dict[str, StoredChunk],
    embedder,
    spec: EmbeddingSpec,
) -> PreparedEmbeddings:

    if embedder.dimension != spec.dimension:
        raise ValueError(
            "Embedding dimension differs from configuration"
        )

    fingerprint = spec.fingerprint()

    # 1. Determine whether an old vector can be reused
    def reusable(
        chunk: Chunk,
        source: StoredChunk | None,
    ) -> bool:
        if source is None:
            return False

        if source.document_id != chunk.document_id:
            return False

        if source.metadata.get(
            "_embedding_fingerprint"
        ) != fingerprint:
            return False

        if embedding_text(source) != embedding_text(chunk):
            return False

        vector = np.asarray(source.vector)

        return (
            vector.shape == (spec.dimension,)
            and bool(np.isfinite(vector).all())
        )

    # 2. Identify points that do not need updating
    to_upsert = list(delta.to_upsert)
    skipped_ids = []

    for chunk in delta.unchanged:
        source = stored.get(chunk.id)

        old_metadata = (
            {
                k: v
                for k, v in source.metadata.items()
                if k not in {
                    "_embedding_fingerprint",
                    "chunk_index",
                }
            }
            if source is not None
            else None
        )

        if (
            reusable(chunk, source)
            and old_metadata == chunk.metadata
        ):
            skipped_ids.append(chunk.id)
        else:
            to_upsert.append(chunk)

    # 3. Reuse available vectors
    vectors = [None] * len(to_upsert)
    missing_chunks = []
    missing_positions = []
    reused_count = 0

    for position, chunk in enumerate(to_upsert):
        source_id = delta.reuse_from.get(
            chunk.id, chunk.id
        )
        source = stored.get(source_id)

        if reusable(chunk, source):
            vectors[position] = np.asarray(
                source.vector,
                dtype=np.float32,
            ).copy()
            reused_count += 1
        else:
            missing_chunks.append(chunk)
            missing_positions.append(position)

    # 4. Embed only missing chunks
    if missing_chunks:
        calculated = np.asarray(
            embedder.embed_documents(missing_chunks),
            dtype=np.float32,
        )

        if (
            calculated.shape
            != (len(missing_chunks), spec.dimension)
            or not np.isfinite(calculated).all()
        ):
            raise ValueError(
                "Embedding model returned invalid vectors"
            )

        for position, vector in zip(
            missing_positions, calculated
        ):
            vectors[position] = vector

    # 5. Assemble the result
    matrix = (
        np.stack(vectors)
        if vectors
        else np.empty(
            (0, spec.dimension),
            dtype=np.float32,
        )
    )

    return PreparedEmbeddings(
        chunks=to_upsert,
        vectors=matrix,
        skipped_ids=skipped_ids,
        delete_ids=set(delta.to_delete),
        reused_count=reused_count,
        computed_count=len(missing_chunks),
    )
