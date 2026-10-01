from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
    ):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def create_collection(self, vector_size: int) -> None:
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )