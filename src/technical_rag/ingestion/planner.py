
from dataclasses import dataclass

from ..models import Document
from .registry import DocumentRegistry


@dataclass
class SyncPlan:
    new: list[Document]
    modified: list[Document]
    unchanged: list[Document]
    deleted_ids: set[str]


def build_sync_plan(
    documents: list[Document],
    registry: DocumentRegistry,
) -> SyncPlan:
    plan = SyncPlan([], [], [], set())
    current_ids = set()

    for document in documents:
        if document.id in current_ids:
            raise ValueError(f"Duplicate document ID: {document.id}")

        current_ids.add(document.id)
        status = registry.get_status(document)

        if status == "new":
            plan.new.append(document)
        elif status == "modified":
            plan.modified.append(document)
        else:
            plan.unchanged.append(document)

    plan.deleted_ids = registry.list_indexed_ids() - current_ids

    return plan
