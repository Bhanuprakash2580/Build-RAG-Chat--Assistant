import logging
import math
import re
from collections import Counter

from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

from app.config import Settings
from app.utils.errors import AppError

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        self._vocabulary: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        if (
            settings.use_openai
            and settings.embedding_provider.lower() == "openai"
            and settings.embedding_api_key
        ):
            self._client = OpenAI(
                api_key=settings.embedding_api_key,
                timeout=settings.request_timeout_seconds,
            )

    @property
    def uses_external_api(self) -> bool:
        return self._client is not None

    def fit_local_fallback(self, texts: list[str]) -> None:
        tokenized = [self._tokenize(text) for text in texts]
        terms = sorted({term for tokens in tokenized for term in tokens})
        self._vocabulary = {term: index for index, term in enumerate(terms)}
        document_count = max(len(texts), 1)
        document_frequency = Counter(
            term for tokens in tokenized for term in set(tokens)
        )
        self._idf = {
            term: math.log((1 + document_count) / (1 + document_frequency[term])) + 1
            for term in terms
        }
        logger.warning("Using local TF-IDF fallback for embeddings.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._client:
            return self._embed_with_openai(texts)
        self.fit_local_fallback(texts)
        return [self._embed_local(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if self._client:
            return self._embed_with_openai([text])[0]
        if not self._vocabulary:
            raise AppError("Embedding service is not initialized", status_code=500)
        return self._embed_local(text)

    def _embed_with_openai(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except AuthenticationError as exc:
            raise AppError("Invalid embedding API key", status_code=401) from exc
        except RateLimitError as exc:
            raise AppError("Embedding rate limit exceeded", status_code=429) from exc
        except APIConnectionError as exc:
            raise AppError("Embedding request timeout or connection error", status_code=504) from exc
        except Exception as exc:
            raise AppError("Embedding provider failed", status_code=502) from exc

    def _embed_local(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        vector = [0.0] * len(self._vocabulary)
        for term, count in counts.items():
            index = self._vocabulary.get(term)
            if index is not None:
                vector[index] = float(count) * self._idf.get(term, 1.0)
        return vector

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", text.lower())
        tokens = [word for word in words if len(word) > 2 and word not in STOP_WORDS]
        bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        return tokens + bigrams


STOP_WORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "that",
    "this",
    "can",
    "are",
    "how",
    "what",
    "where",
    "when",
    "who",
    "why",
    "into",
    "your",
    "you",
}


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
