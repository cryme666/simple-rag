from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict  # Import this

# This resolves to the folder containing 'app', which is your project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    # Required fields (no default values)
    groq_api_key: str

    # Optional fields with defaults
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/ragdb"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    distance_threshold: float = 0.5

    bm25_top_k: int = 5

    # Query transformation (for normal_chat retrieval)
    query_transform_enabled: bool = True
    query_transform_split_clarify_enabled: bool = True
    query_transform_fallback_with_context_enabled: bool = True
    query_transform_max_questions: int = 3
    query_transform_min_query_len: int = 8
    query_transform_fallback_max_prev_user_messages: int = 5

    # Modern Pydantic V2 configuration
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,  # This helps match .env keys to class variables
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
