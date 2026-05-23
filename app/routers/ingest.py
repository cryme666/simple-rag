import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import get_settings
from app.schemas import IngestFileResponse
from app.services.pdf_parser import extract_text_from_pdf
from app.services.embedding import generate_embeddings_batch
from app.services.vector_store import insert_chunks, source_exists, delete_by_source, clear_all_documents
from app.services.bm25_index import bm25_index
from app.utils.chunker import chunk_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}
ALLOWED_EXTENSIONS = {".pdf"}


def _validate_pdf(file: UploadFile):
    filename = file.filename or ""
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed extensions: {ALLOWED_EXTENSIONS}",
        )
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported MIME type '{file.content_type}'. Expected application/pdf.",
        )


@router.post(
    "/file",
    response_model=IngestFileResponse,
    summary="Ingest a PDF file",
    description="Upload a PDF file to extract text, chunk it, generate embeddings, and store in the vector database.",
)
async def ingest_file(
    file: UploadFile = File(..., description="PDF file to ingest"),
    overwrite: bool = Form(False, description="If true, replace existing chunks for this source"),
    db: AsyncSession = Depends(get_db),
):
    _validate_pdf(file)
    source = file.filename or "unknown.pdf"

    exists = await source_exists(db, source)
    if exists and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Source '{source}' already ingested. Set overwrite=true to replace.",
        )
    if exists and overwrite:
        await delete_by_source(db, source)
        bm25_index.mark_dirty()

    try:
        file_bytes = await file.read()
        pages = extract_text_from_pdf(file_bytes)
    except Exception as e:
        logger.error("PDF parsing failed: %s", e)
        raise HTTPException(status_code=422, detail="Failed to parse the PDF file.")

    if not pages:
        raise HTTPException(status_code=422, detail="No extractable text found in the PDF.")

    settings = get_settings()
    all_chunks: list[str] = []
    metadata_list: list[dict] = []

    for page_info in pages:
        page_chunks = chunk_text(page_info["text"], settings.chunk_size, settings.chunk_overlap)
        for chunk in page_chunks:
            all_chunks.append(chunk)
            metadata_list.append({"page": page_info["page"]})

    if not all_chunks:
        raise HTTPException(status_code=422, detail="Text extraction produced no usable chunks.")

    try:
        embeddings = await generate_embeddings_batch(all_chunks)
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to generate embeddings.")

    try:
        stored = await insert_chunks(db, source, "file", all_chunks, embeddings, metadata_list)
        bm25_index.mark_dirty()
    except Exception as e:
        logger.error("DB insert failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store chunks in the database.")

    return IngestFileResponse(
        message="File ingested successfully.",
        chunks_stored=stored,
        source=source,
    )


@router.delete(
    "/clear",
    summary="Clear the vector database",
    description="Deletes all document chunks from the vector database.",
)
async def clear_database(db: AsyncSession = Depends(get_db)):
    try:
        deleted_count = await clear_all_documents(db)
        bm25_index.mark_dirty()
        return {
            "message": "Database cleared successfully.",
            "deleted_count": deleted_count,
        }
    except Exception as e:
        logger.error("Failed to clear database: %s", e)
        raise HTTPException(status_code=500, detail="Failed to clear the database.")
