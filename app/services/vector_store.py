import uuid

import logging

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk

logger = logging.getLogger(__name__)

async def similarity_candidates(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    embedding_literal = f"[{','.join(str(x) for x in query_embedding)}]"

    stmt = text(
        """
        SELECT
          id,
          source,
          source_type,
          content,
          metadata,
          created_at,
          (embedding <=> cast(:embedding AS vector)) AS distance
        FROM document_chunks
        ORDER BY distance
        LIMIT :top_k
        """
    ).bindparams(embedding=embedding_literal, top_k=top_k)

    result = await session.execute(stmt)
    rows = result.fetchall()
    return [
        {
            "id": str(row.id),
            "source": row.source,
            "source_type": row.source_type,
            "content": row.content,
            "metadata": row.metadata,
            "created_at": row.created_at,
            "distance": float(row.distance),
        }
        for row in rows
    ]


async def insert_chunks(
    session: AsyncSession,
    source: str,
    source_type: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata_list: list[dict] | None = None,
) -> int:
    if metadata_list is None:
        metadata_list = [{} for _ in chunks]

    objects = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        meta = metadata_list[i] if i < len(metadata_list) else {}
        meta["chunk_index"] = i
        obj = DocumentChunk(
            id=uuid.uuid4(),
            source=source,
            source_type=source_type,
            content=chunk,
            embedding=embedding,
            metadata_=meta,
        )
        objects.append(obj)

    session.add_all(objects)
    await session.commit()
    return len(objects)


async def delete_by_source(session: AsyncSession, source: str) -> int:
    result = await session.execute(
        delete(DocumentChunk).where(DocumentChunk.source == source)
    )
    await session.commit()
    return result.rowcount


async def clear_all_documents(session: AsyncSession) -> int:
    result = await session.execute(delete(DocumentChunk))
    await session.commit()
    return result.rowcount


async def source_exists(session: AsyncSession, source: str) -> bool:
    result = await session.execute(
        select(DocumentChunk.id).where(DocumentChunk.source == source).limit(1)
    )
    return result.scalar_one_or_none() is not None


async def similarity_search(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int = 5,
    distance_threshold: float = 0.5,
) -> list[DocumentChunk]:
    candidates = await similarity_candidates(session, query_embedding, top_k=top_k)

    if candidates:
        logger.info(
            "[VECTOR] top_k=%d threshold=%.4f -> candidates=%d | best_distance=%.4f",
            top_k,
            float(distance_threshold),
            len(candidates),
            float(candidates[0]["distance"]),
        )
        for i, c in enumerate(candidates):
            preview = c["content"][:150] + "..." if len(c["content"]) > 150 else c["content"]
            logger.info(
                "[VECTOR] Candidate[%d] dist=%.4f source=%s type=%s | %r",
                i,
                float(c["distance"]),
                c["source"],
                c["source_type"],
                preview,
            )
    else:
        logger.info("[VECTOR] top_k=%d threshold=%.4f -> no candidates (empty table?)", top_k, float(distance_threshold))

    filtered = [c for c in candidates if float(c["distance"]) < float(distance_threshold)]
    if not filtered and candidates:
        logger.info(
            "[VECTOR] 0 chunks passed threshold=%.4f (best_distance=%.4f).",
            float(distance_threshold),
            float(candidates[0]["distance"]),
        )

    chunks = []
    for c in filtered:
        chunk = DocumentChunk(
            id=uuid.UUID(c["id"]),
            source=c["source"],
            source_type=c["source_type"],
            content=c["content"],
            metadata_=c["metadata"],
            created_at=c["created_at"],
        )
        chunks.append(chunk)

    return chunks
