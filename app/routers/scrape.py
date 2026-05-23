import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import get_settings
from app.schemas import ScrapeRequest, ScrapeResponse
from app.services.web_scraper import scrape_url
from app.services.embedding import generate_embeddings_batch
from app.services.vector_store import insert_chunks, source_exists, delete_by_source
from app.services.bm25_index import bm25_index
from app.utils.chunker import chunk_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "/url",
    response_model=ScrapeResponse,
    summary="Ingest content from a URL",
    description="Scrape a web page, extract readable text, chunk it, generate embeddings, and store in the vector database.",
)
async def ingest_url(
    request: ScrapeRequest,
    db: AsyncSession = Depends(get_db),
):
    source = str(request.url)

    exists = await source_exists(db, source)
    if exists and not request.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Source '{source}' already ingested. Set overwrite=true to replace.",
        )
    if exists and request.overwrite:
        await delete_by_source(db, source)
        bm25_index.mark_dirty()

    try:
        scraped = await scrape_url(source)
    except Exception as e:
        logger.error("Scraping failed for %s: %s", source, e)
        raise HTTPException(status_code=422, detail=f"Failed to scrape URL: {e}")

    text = scraped["text"]
    title = scraped["title"]

    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="No readable text extracted from the URL.")

    settings = get_settings()
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)

    if not chunks:
        raise HTTPException(status_code=422, detail="Text extraction produced no usable chunks.")

    try:
        embeddings = await generate_embeddings_batch(chunks)
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to generate embeddings.")

    metadata_list = [{"title": title, "url": source} for _ in chunks]

    try:
        stored = await insert_chunks(db, source, "url", chunks, embeddings, metadata_list)
        bm25_index.mark_dirty()
    except Exception as e:
        logger.error("DB insert failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store chunks in the database.")

    return ScrapeResponse(
        message="URL content ingested successfully.",
        chunks_stored=stored,
        source=source,
        title=title,
    )
