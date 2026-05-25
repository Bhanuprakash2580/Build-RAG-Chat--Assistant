from fastapi import APIRouter, Depends

from app.main_dependencies import get_rag_service
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services.rag import RAGService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    result, sources = rag_service.answer(
        session_id=request.sessionId.strip(),
        message=request.message.strip(),
    )
    return ChatResponse(
        reply=result.text,
        tokensUsed=result.tokens_used,
        retrievedChunks=len(sources),
        sources=sources,
    )
