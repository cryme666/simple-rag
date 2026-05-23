import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ']+")

@lru_cache
def _english_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        # Best-effort: if stopwords corpus isn't available (common in Docker),
        # fall back to no stopwords rather than failing retrieval.
        logger.warning("[BM25] NLTK stopwords not available; using empty stopwords set.")
        return set()


def _tokenize(text_value: str) -> list[str]:
    if not text_value:
        return []
    sw = _english_stopwords()
    tokens = [t.lower() for t in _TOKEN_RE.findall(text_value)]
    # Drop stopwords and very short tokens to reduce lexical noise.
    return [t for t in tokens if t not in sw and len(t) > 2]


def _preview(text_value: str, limit: int = 160) -> str:
    t = (text_value or "").strip()
    return t if len(t) <= limit else t[:limit] + "..."


@dataclass(frozen=True)
class _ChunkRow:
    id: uuid.UUID
    source: str
    source_type: str
    content: str
    metadata: dict[str, Any] | None
    created_at: datetime | None


class BM25Index:

    def __init__(self) -> None:
        self._dirty = True
        self._lock = asyncio.Lock()
        self._bm25 = None
        self._rows: list[_ChunkRow] = []
        self._corpus_tokens: list[list[str]] = []

    def mark_dirty(self) -> None:
        self._dirty = True
        logger.info("[BM25] mark_dirty set. next_search_will_rebuild=%s", True)

    async def ensure_ready(self, session: AsyncSession) -> None:
        if not self._dirty and self._bm25 is not None:
            return

        async with self._lock:
            if not self._dirty and self._bm25 is not None:
                return
            logger.info("[BM25] rebuilding index (dirty=%s)", self._dirty)

            try:
                from rank_bm25 import BM25Okapi  # imported lazily
            except Exception as e:  # pragma: no cover
                logger.error("rank_bm25 is not installed; BM25 disabled. err=%s", e)
                self._bm25 = None
                self._rows = []
                self._corpus_tokens = []
                self._dirty = False
                return

            stmt = text(
                """
                SELECT id, source, source_type, content, metadata, created_at
                FROM document_chunks
                """
            )
            result = await session.execute(stmt)
            fetched = result.fetchall()

            rows: list[_ChunkRow] = []
            corpus_tokens: list[list[str]] = []
            for r in fetched:
                content = r.content or ""
                rows.append(
                    _ChunkRow(
                        id=r.id,
                        source=r.source,
                        source_type=r.source_type,
                        content=content,
                        metadata=r.metadata,
                        created_at=r.created_at,
                    )
                )
                corpus_tokens.append(_tokenize(content))

            self._rows = rows
            self._corpus_tokens = corpus_tokens
            self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
            self._dirty = False

            logger.info("[BM25] index rebuilt. chunks=%d", len(self._rows))

    async def search(self, session: AsyncSession, query: str, top_k: int) -> list[_ChunkRow]:
        query = (query or "").strip()
        if not query or top_k <= 0:
            return []

        await self.ensure_ready(session)
        if self._bm25 is None or not self._rows:
            logger.info("[BM25] search skipped (index empty). query=%r", _preview(query))
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            logger.info("[BM25] search skipped (no query tokens). query=%r", _preview(query))
            return []

        logger.info(
            "[BM25] search start. query=%r tokens=%d top_k=%d corpus=%d",
            _preview(query),
            len(q_tokens),
            int(top_k),
            len(self._rows),
        )

        scores = self._bm25.get_scores(q_tokens)
        if scores is None:
            return []

        # scores is a numpy array or list-like; avoid dependency on numpy directly.
        # Take top_k indices by score (descending).
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: float(x[1]), reverse=True)
        top = indexed[:top_k]
        if top:
            best_score = float(top[0][1])
            best_id = str(self._rows[top[0][0]].id)

            # Gate: if BM25 found no positive lexical signal, don't return noisy "top-k".
            if best_score <= 0.0:
                logger.info(
                    "[BM25] gate triggered (best_score=%.6f). returning 0 results. query=%r",
                    best_score,
                    _preview(query),
                )
                return []

            logger.info(
                "[BM25] search done. best_score=%.6f best_id=%s",
                best_score,
                best_id,
            )
            for rank, (i, s) in enumerate(top[:3], start=1):
                row = self._rows[i]
                logger.debug(
                    "[BM25] Top[%d] score=%.6f id=%s source=%s type=%s content=%r",
                    rank,
                    float(s),
                    str(row.id),
                    row.source,
                    row.source_type,
                    _preview(row.content),
                )

        return [self._rows[i] for i, _ in top]


bm25_index = BM25Index()

