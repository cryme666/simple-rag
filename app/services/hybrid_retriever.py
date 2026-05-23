import logging
import uuid
from collections.abc import Iterable

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk
from app.services.bm25_index import bm25_index
from app.services.vector_store import similarity_candidates

logger = logging.getLogger(__name__)
RRF_K = 60


def _preview(text_value: str, limit: int = 160) -> str:
    t = (text_value or "").strip()
    return t if len(t) <= limit else t[:limit] + "..."


def _to_document_chunk(candidate: dict) -> DocumentChunk:
    return DocumentChunk(
        id=uuid.UUID(candidate["id"]),
        source=candidate["source"],
        source_type=candidate["source_type"],
        content=candidate["content"],
        metadata_=candidate["metadata"],
        created_at=candidate["created_at"],
    )


def _score_rrf(
    scores: dict[uuid.UUID, float],
    ranked_ids: Iterable[uuid.UUID],
    *,
    label: str,
) -> None:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (RRF_K + rank))
        logger.debug(
            "[HYBRID][RRF] source=%s rank=%d id=%s partial_score=%.6f",
            label,
            rank,
            str(chunk_id),
            scores[chunk_id],
        )


async def hybrid_search_rrf(
    session: AsyncSession,
    *,
    query_text: str,
    query_embedding: list[float],
    top_k: int,
    distance_threshold: float,
    bm25_top_k: int | None = None,
) -> list[DocumentChunk]:
    """
    Hybrid retrieval for normal_chat using Reciprocal Rank Fusion (RRF).
    Vector and BM25 each contribute rank-based scores; final ordering is by fused score.
    """

    bm25_k = int(bm25_top_k) if bm25_top_k is not None else int(top_k)
    bm25_k = max(bm25_k, 0)
    top_k = max(int(top_k), 0)

    logger.info(
        "[HYBRID] start. query=%r top_k=%d bm25_top_k=%d distance_threshold=%.4f rrf_k=%d",
        _preview(query_text),
        top_k,
        bm25_k,
        float(distance_threshold),
        RRF_K,
    )

    vector_candidates = await similarity_candidates(session, query_embedding, top_k=top_k)
    vector_filtered = [
        candidate
        for candidate in vector_candidates
        if float(candidate["distance"]) < float(distance_threshold)
    ]
    vector_results = [_to_document_chunk(candidate) for candidate in vector_filtered]

    bm25_rows = await bm25_index.search(session, query_text, top_k=bm25_k)

    vector_by_id = {chunk.id: chunk for chunk in vector_results}
    bm25_by_id: dict[uuid.UUID, DocumentChunk] = {}
    for row in bm25_rows:
        bm25_by_id[row.id] = (
            DocumentChunk(
                id=row.id,
                source=row.source,
                source_type=row.source_type,
                content=row.content,
                metadata_=row.metadata,
                created_at=row.created_at,
            )
        )

    fused_scores: dict[uuid.UUID, float] = {}
    _score_rrf(fused_scores, [chunk.id for chunk in vector_results], label="vector")
    _score_rrf(fused_scores, [row.id for row in bm25_rows], label="bm25")

    dedup_overlap = len(set(vector_by_id) & set(bm25_by_id))
    merged_by_id = {**bm25_by_id, **vector_by_id}
    ranked_items = sorted(
        merged_by_id.items(),
        key=lambda item: (
            fused_scores.get(item[0], 0.0),
            1 if item[0] in vector_by_id else 0,
        ),
        reverse=True,
    )
    results = [chunk for _, chunk in ranked_items[:top_k]]

    logger.info(
        "[HYBRID] vector=%d bm25=%d overlap=%d -> fused=%d (top_k=%d)",
        len(vector_results),
        len(bm25_rows),
        dedup_overlap,
        len(results),
        top_k,
    )
    for rank, chunk in enumerate(results[:5], start=1):
        logger.info(
            "[HYBRID] Final[%d] rrf=%.6f id=%s source=%s type=%s | %r",
            rank,
            fused_scores.get(chunk.id, 0.0),
            str(chunk.id),
            chunk.source,
            chunk.source_type,
            _preview(chunk.content),
        )

    return results

