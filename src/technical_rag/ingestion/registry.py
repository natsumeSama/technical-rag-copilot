
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from ..models import Document


DocumentStatus = Literal["new", "unchanged", "modified"]


class DocumentRegistry:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS indexed_documents (
                    document_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL
                )
            """)
            conn.commit()

    def get_status(self, document: Document) -> DocumentStatus:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                """
                SELECT content_hash
                FROM indexed_documents
                WHERE document_id = ?
                """,
                (document.id,),
            ).fetchone()

        if row is None:
            return "new"

        if row[0] == document.content_hash:
            return "unchanged"

        return "modified"

    def mark_indexed(self, document: Document) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO indexed_documents (document_id, content_hash)
                VALUES (?, ?)
                ON CONFLICT(document_id)
                DO UPDATE SET content_hash = excluded.content_hash
                """,
                (document.id, document.content_hash),
            )
            conn.commit()


    def list_indexed_ids(self) -> set[str]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT document_id FROM indexed_documents"
            ).fetchall()

        return {row[0] for row in rows}
