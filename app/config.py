from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8-sig",
        extra="ignore",
    )

    use_openai: bool = Field(default=False, alias="USE_OPENAI")
    llm_provider: str = Field(default="local", alias="LLM_PROVIDER")
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    top_k: int = Field(default=3, alias="TOP_K")
    similarity_threshold: float = Field(default=0.25, alias="SIMILARITY_THRESHOLD")
    request_timeout_seconds: float = Field(default=30.0, alias="REQUEST_TIMEOUT_SECONDS")
    max_history_pairs: int = Field(default=5, alias="MAX_HISTORY_PAIRS")


@lru_cache
def get_settings() -> Settings:
    return Settings()
