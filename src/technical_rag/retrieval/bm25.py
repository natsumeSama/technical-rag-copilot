import bm25s
import Stemmer

from ..models import Chunk, SearchResult


class BM25Retriever:
    def __init__(self):
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = None
        self.corpus = []

    def index(self, chunks: list[Chunk]) -> None:
        texts = []
        corpus = []

        for chunk in chunks:
            if chunk.section_title:
                text = f"{chunk.section_title}\n\n{chunk.content}"
            else:
                text = chunk.content

            texts.append(text)

            corpus.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "section_title": chunk.section_title,
                    "metadata": chunk.metadata,
                }
            )

        corpus_tokens = bm25s.tokenize(
            texts,
            stopwords="en",
            stemmer=self.stemmer,
        )

        self.retriever = bm25s.BM25(
            corpus=corpus,
        )

        self.retriever.index(corpus_tokens)

        self.corpus = corpus

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:

        if self.retriever is None:
            raise RuntimeError(
                "BM25 index has not been built yet"
            )

        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=self.stemmer,
        )

        documents, scores = self.retriever.retrieve(
            query_tokens,
            corpus=self.corpus,
            k=limit,
        )

        results = []

        for document, score in zip(
            documents[0],
            scores[0],
        ):
            results.append(
                SearchResult(
                    chunk_id=document["chunk_id"],
                    document_id=document["document_id"],
                    content=document["content"],
                    section_title=document["section_title"],
                    score=float(score),
                    metadata=document["metadata"],
                )
            )

        return results