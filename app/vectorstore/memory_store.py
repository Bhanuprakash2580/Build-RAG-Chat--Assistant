import logging
from dataclasses import dataclass

from app.services.embeddings import normalize

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    id: str
    title: str
    source: str
    text: str
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: list[dict] = []

    def add(self, chunks: list[dict[str, str]], embeddings: list[list[float]]) -> None:
        self._records = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._records.append({**chunk, "embedding": normalize(embedding)})

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        query = normalize(query_embedding)
        results: list[SearchResult] = []

        for record in self._records:
            score = float(sum(left * right for left, right in zip(query, record["embedding"])))
            logger.info(
                "Similarity score %.4f for %s (%s)",
                score,
                record["title"],
                record["id"],
            )
            results.append(
                SearchResult(
                    id=record["id"],
                    title=record["title"],
                    source=record["source"],
                    text=record["text"],
                    score=score,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self._records)
