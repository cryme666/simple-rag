from functools import lru_cache

from groq import AsyncGroq
from fastapi import HTTPException

from app.config import get_settings


@lru_cache
def get_groq_client() -> AsyncGroq:
    settings = get_settings()
    if not (settings.groq_api_key or "").strip():
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured. Set GROQ_API_KEY in the environment.",
        )
    return AsyncGroq(api_key=settings.groq_api_key)
