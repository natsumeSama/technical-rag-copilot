
"""Compare V2 embeddings with LangChain embeddings."""

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from technical_rag.config import V2_PROFILE
from technical_rag.embeddings import EmbeddingModel
from technical_rag.models import Chunk
from technical_rag.embedding_text import embedding_text


model_id = V2_PROFILE.embedding_model_id
revision = V2_PROFILE.embedding_revision

# Existing V2 implementation
v2 = EmbeddingModel(
    model_name=model_id,
    revision=revision,
)

# Supported LangChain integration
v3 = HuggingFaceEmbeddings(
    model_name=model_id,
    model_kwargs={"revision": revision},
    encode_kwargs={"normalize_embeddings": True},
)

chunk = Chunk(
    id="test-chunk",
    document_id="test-document",
    content="Qdrant supports cosine similarity.",
    chunk_index=0,
    section_title="Metrics",
    metadata={},
)

document_text = embedding_text(chunk)
query = "Which similarity metrics does Qdrant support?"


def compare(name, old, new):
    old = np.asarray(old, dtype=np.float32)
    new = np.asarray(new, dtype=np.float32)

    assert old.shape == new.shape

    cosine = np.dot(old, new) / (
        np.linalg.norm(old) * np.linalg.norm(new)
    )

    print(f"\n{name}")
    print("Dimension:", len(old))
    print("Cosine similarity:", round(float(cosine), 8))
    print(
        "Maximum absolute difference:",
        float(np.max(np.abs(old - new))),
    )


compare(
    "Document embeddings",
    v2.embed_documents([chunk])[0],
    v3.embed_documents([document_text])[0],
)

compare(
    "Query embeddings",
    v2.embed_query(query),
    v3.embed_query(query),
)
