
import json
from collections import Counter
from pathlib import Path

from technical_rag.ingestion.markdown import load_markdown
from technical_rag.ingestion.registry import DocumentRegistry
from technical_rag.config_v2 import REGISTRY_PATH

ROOT = Path(__file__).resolve().parents[1]

SOURCES_PATH = ROOT / "data/sources.json"


def main():
    print(
        "CONTENT-ONLY SCAN "
        "(metadata and model changes are not checked)"
    )
    registry = DocumentRegistry(REGISTRY_PATH)

    sources = json.loads(
        SOURCES_PATH.read_text(encoding="utf-8")
    )

    statuses = Counter()
    current_ids = set()
    for source in sources:
        document = load_markdown(
            file_path=str(ROOT / source["file_path"]),
            source_name=source["source_name"],
            source_url=source["source_url"],
            document_type=source["document_type"],
            metadata=source["metadata"],
        )
        current_ids.add(document.id)
        status = registry.get_status(document)
        statuses[status] += 1
        print(f"{document.title}: {status.upper()}")

    deleted_ids = registry.list_indexed_ids() - current_ids

    statuses["deleted"] = len(deleted_ids)

    for document_id in deleted_ids:
        print(f"{document_id}: DELETED")

    print("\nSCAN SUMMARY")
    
    for status in ("new", "modified", "unchanged", "deleted"):
        print(f"{status}: {statuses[status]}")


if __name__ == "__main__":
    main()
