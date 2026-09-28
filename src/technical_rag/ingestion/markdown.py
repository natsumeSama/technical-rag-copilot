from pathlib import Path
from hashlib import sha256

from ..models import Document


def load_markdown(
    file_path: str,
    source_name: str,
    source_url: str,
    document_type: str,
    metadata: dict[str, object],
) -> Document:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document path does not exist: {file_path}")

    identity = source_url + document_type
    document_id = sha256(identity.encode("utf-8")).hexdigest()

    title = path.stem
    content = path.read_text(encoding="utf-8")
    content_hash = sha256(content.encode("utf-8")).hexdigest()

    document = Document(
        id=document_id,
        content_hash=content_hash,
        title=title,
        content=content,
        source_name=source_name,
        source_url=source_url,
        document_type=document_type,
        metadata=metadata,
    )

    return document