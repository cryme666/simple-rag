import os
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.routers import chat, ingest, scrape
from app.services.embedding import warmup_embedding_model
from app.services.nltk_setup import ensure_nltk_stopwords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_nltk_stopwords()
    warmup_embedding_model()
    logger.info("Starting up - initializing database")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation backend powered by Groq, sentence-transformers, and pgvector.",
    version="1.0.0",
    lifespan=lifespan,
)

_settings = get_settings()
_cors_origins = [origin.strip() for origin in _settings.cors_origins.split(",") if origin.strip()]
logger.info("CORS allow_origins: %s", _cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    logger.info(
        "%s %s -> %d (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(scrape.router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
