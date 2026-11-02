from pathlib import Path

from technical_rag.ingestion.markdown import load_markdown
from technical_rag.chunking import chunk_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample.md"


document = load_markdown(
    file_path=str(SAMPLE_PATH),
    source_name="Local Sample",
    source_url="local://sample.md",
    document_type="documentation",
    metadata={
        "language": "en",
    },
)

chunks = chunk_document(
    document=document,
    chunk_size=80,
    chunk_overlap=20,
)


print(f"Document: {document.title}")
print(f"Number of chunks: {len(chunks)}")
print("=" * 60)

for chunk in chunks:
    print(f"\nCHUNK {chunk.chunk_index}")
    print(f"Section: {chunk.section_title}")
    print(f"Metadata: {chunk.metadata}")
    print(chunk.content)
    print("-" * 60)