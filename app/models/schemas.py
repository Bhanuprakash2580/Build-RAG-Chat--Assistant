from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    sessionId: str = Field(..., min_length=1, description="Client session identifier")
    message: str = Field(..., min_length=1, description="User message")


class SourceChunk(BaseModel):
    title: str
    chunkId: str
    score: float
    text: str


class ChatResponse(BaseModel):
    reply: str
    tokensUsed: int | None = None
    retrievedChunks: int
    sources: list[SourceChunk]


class HealthResponse(BaseModel):
    status: str
