from functools import lru_cache

from app.config import get_settings
from app.services.documents import build_chunks, load_documents
from app.services.embeddings import EmbeddingService
from app.services.history import ConversationHistory
from app.services.llm import LLMService
from app.services.rag import RAGService
from app.vectorstore.memory_store import InMemoryVectorStore


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    documents = load_documents()
    chunks = build_chunks(documents)
    embedding_service = EmbeddingService(settings)
    embeddings = embedding_service.embed_documents([chunk["text"] for chunk in chunks])
    vector_store = InMemoryVectorStore()
    vector_store.add(chunks, embeddings)
    history = ConversationHistory(max_pairs=settings.max_history_pairs)
    llm_service = LLMService(settings)
    return RAGService(
        settings=settings,
        embedding_service=embedding_service,
        llm_service=llm_service,
        vector_store=vector_store,
        history=history,
    )
