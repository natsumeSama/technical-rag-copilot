from hashlib import sha256
from pathlib import Path

import frontmatter

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
        raise FileNotFoundError(
            f"Document path does not exist: {file_path}"
        )

    post = frontmatter.load(path)

    content = post.content

    document_metadata = {
        **post.metadata,
        **metadata,
    }

    title = post.metadata.get("title", path.stem)

    identity = source_url + document_type
    document_id = sha256(
        identity.encode("utf-8")
    ).hexdigest()

    content_hash = sha256(
        content.encode("utf-8")
    ).hexdigest()

    return Document(
        id=document_id,
        content_hash=content_hash,
        title=title,
        content=content,
        source_name=source_name,
        source_url=source_url,
        document_type=document_type,
        metadata=document_metadata,
    )