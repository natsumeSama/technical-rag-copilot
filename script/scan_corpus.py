
from collections import Counter

from technical_rag.ingestion.registry import DocumentRegistry
from technical_rag.config_v2 import (
    PROJECT_ROOT,
    SOURCES_PATH,
    REGISTRY_PATH,
)
from technical_rag.ingestion.corpus import load_documents


def main():
    print(
        "CONTENT-ONLY SCAN "
        "(metadata and model changes are not checked)"
    )
    registry = DocumentRegistry(REGISTRY_PATH)
    documents = load_documents(
        sources_path=SOURCES_PATH,
        project_root=PROJECT_ROOT,
    )

    statuses = Counter()
    current_ids = set()

    for document in documents:
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
