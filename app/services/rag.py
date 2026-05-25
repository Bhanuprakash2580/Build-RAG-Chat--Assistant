from app.config import Settings
from app.models.schemas import SourceChunk
from app.prompts.rag_prompt import build_rag_prompt
from app.services.embeddings import EmbeddingService
from app.services.history import ConversationHistory
from app.services.llm import LLMResult, LLMService
from app.vectorstore.memory_store import InMemoryVectorStore, SearchResult

FALLBACK_REPLY = "I could not find enough information in the knowledge base to answer this question."


class RAGService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        vector_store: InMemoryVectorStore,
        history: ConversationHistory,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.history = history

    def answer(self, session_id: str, message: str) -> tuple[LLMResult, list[SourceChunk]]:
        query_embedding = self.embedding_service.embed_query(message)
        retrieved = self.vector_store.search(query_embedding, self.settings.top_k)
        qualified = [
            result
            for result in retrieved
            if result.score >= self.settings.similarity_threshold
        ]

        sources = [self._to_source(result) for result in qualified]
        if not qualified:
            result = LLMResult(text=FALLBACK_REPLY, tokens_used=0)
            self.history.add(session_id, message, result.text)
            return result, []

        context = self._format_context(qualified)
        history = self.history.format_for_prompt(session_id)
        prompt = build_rag_prompt(context=context, history=history, question=message)
        result = self.llm_service.generate(prompt=prompt, context=context, question=message)
        self.history.add(session_id, message, result.text)
        return result, sources

    def _format_context(self, results: list[SearchResult]) -> str:
        return "\n\n".join(
            f"[{index}] Title: {result.title}\nSource: {result.source}\nContent: {result.text}"
            for index, result in enumerate(results, start=1)
        )

    def _to_source(self, result: SearchResult) -> SourceChunk:
        return SourceChunk(
            title=result.title,
            chunkId=result.id,
            score=round(result.score, 4),
            text=result.text,
        )
