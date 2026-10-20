
import json
from collections import Counter
from pathlib import Path

from technical_rag.ingestion.markdown import load_markdown
from technical_rag.ingestion.registry import DocumentRegistry


ROOT = Path(__file__).resolve().parents[1]

SOURCES_PATH = ROOT / "data/sources.json"
REGISTRY_PATH = ROOT / "data/state/registry.db"


def main():
    registry = DocumentRegistry(REGISTRY_PATH)

    sources = json.loads(
        SOURCES_PATH.read_text(encoding="utf-8")
    )

    statuses = Counter()

    for source in sources:
        document = load_markdown(
            file_path=str(ROOT / source["file_path"]),
            source_name=source["source_name"],
            source_url=source["source_url"],
            document_type=source["document_type"],
            metadata=source["metadata"],
        )

        status = registry.get_status(document)
        statuses[status] += 1

        print(f"{document.title}: {status.upper()}")

    print("\nSCAN SUMMARY")
    for status in ("new", "modified", "unchanged"):
        print(f"{status}: {statuses[status]}")


if __name__ == "__main__":
    main()
