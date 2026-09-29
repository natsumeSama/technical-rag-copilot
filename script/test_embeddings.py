import numpy as np
from pathlib import Path

from technical_rag.ingestion.markdown import load_markdown
from technical_rag.chunking import chunk_document
from technical_rag.embeddings import EmbeddingModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample.md"


document = load_markdown(
    file_path=str(SAMPLE_PATH),
    source_name="Local Sample",
    source_url="local://sample.md",
    document_type="documentation",
    metadata={"language": "en"},
)

chunks = chunk_document(
    document=document,
    chunk_size=200,
    chunk_overlap=30,
)

embedding_model = EmbeddingModel(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

document_embeddings = embedding_model.embed_documents(chunks)

query = "What retrieval approach relies on lexical matching?"
query_embedding = embedding_model.embed_query(query)

scores = document_embeddings @ query_embedding

ranking = np.argsort(scores)[::-1]

for index in ranking:
    print(f"Score: {scores[index]:.4f}")
    print(f"Section: {chunks[index].section_title}")
    print(chunks[index].content)
    print("-" * 60)