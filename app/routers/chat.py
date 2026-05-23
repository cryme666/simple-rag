import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_groq_client
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, SourceInfo
from app.services.embedding import generate_embedding
from app.services.hybrid_retriever import hybrid_search_rrf
from app.services.llm import chat_completion
from app.services.query_transform import split_and_clarify_query, transform_query_with_context
from app.services.vector_store import similarity_candidates

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the AI assistant",
    description="Send a message and get a response grounded in the ingested knowledge base (RAG).",
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    llm_client=Depends(get_groq_client),
):
    conversation_history = [msg.model_dump() for msg in request.conversation_history]

    logger.info(
        "[CHAT] Incoming request: message=%r | history_len=%d",
        request.message,
        len(conversation_history),
    )

    return await _handle_normal_chat(llm_client, db, request.message, conversation_history)


async def _handle_normal_chat(
    llm_client,
    db: AsyncSession,
    user_message: str,
    conversation_history: list[dict],
) -> ChatResponse:
    settings = get_settings()

    retrieval_message = user_message
    if (
        settings.query_transform_enabled
        and settings.query_transform_split_clarify_enabled
        and len((user_message or "").strip()) >= int(settings.query_transform_min_query_len)
    ):
        try:
            clarified = await split_and_clarify_query(llm_client, user_message, settings=settings)
            retrieval_message = clarified[0] if clarified else user_message
            logger.info(
                "[QTRANSFORM] using transformed retrieval query. original=%r transformed=%r",
                (user_message or "")[:200] + "..." if len(user_message or "") > 200 else (user_message or ""),
                (retrieval_message or "")[:200] + "..." if len(retrieval_message or "") > 200 else (retrieval_message or ""),
            )
        except Exception as e:
            logger.warning("[QTRANSFORM] error; continuing with original query. err=%s", e)
            retrieval_message = user_message
    else:
        logger.info(
            "[QTRANSFORM] skipped. enabled=%s split_clarify=%s len=%d min_len=%d",
            bool(settings.query_transform_enabled),
            bool(settings.query_transform_split_clarify_enabled),
            len((user_message or "").strip()),
            int(settings.query_transform_min_query_len),
        )

    try:
        query_embedding = await generate_embedding(retrieval_message)
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to generate embedding for the query.")

    try:
        results = await hybrid_search_rrf(
            db,
            query_text=user_message,
            query_embedding=query_embedding,
            top_k=settings.top_k,
            distance_threshold=settings.distance_threshold,
            bm25_top_k=getattr(settings, "bm25_top_k", settings.top_k),
        )
    except Exception as e:
        logger.error("Hybrid search failed: %s", e)
        results = []

    # If query transformation made retrieval worse, retry once with the original user message embedding.
    if not results and (retrieval_message or "").strip() and (retrieval_message or "").strip() != (user_message or "").strip():
        try:
            logger.info("[QTRANSFORM] no results with transformed query; retrying retrieval with original user message embedding.")
            query_embedding = await generate_embedding(user_message)
            results = await hybrid_search_rrf(
                db,
                query_text=user_message,
                query_embedding=query_embedding,
                top_k=settings.top_k,
                distance_threshold=settings.distance_threshold,
                bm25_top_k=getattr(settings, "bm25_top_k", settings.top_k),
            )
        except Exception as e:
            logger.warning("[QTRANSFORM] retry with original embedding failed; continuing. err=%s", e)
            results = []

    # Fallback: if retrieval got no chunks, try rewriting using previous user messages and rerun retrieval once.
    if (
        not results
        and settings.query_transform_enabled
        and settings.query_transform_fallback_with_context_enabled
        and conversation_history
    ):
        prev_user_msgs = [
            (m.get("content") or "").strip()
            for m in conversation_history
            if (m.get("role") == "user" and (m.get("content") or "").strip())
        ]
        if prev_user_msgs:
            try:
                rewritten = await transform_query_with_context(
                    llm_client,
                    user_message,
                    prev_user_msgs,
                    settings=settings,
                )
                if rewritten and rewritten.strip() and rewritten.strip() != (retrieval_message or "").strip():
                    logger.info(
                        "[QTRANSFORM] fallback rerun retrieval. prev_user=%d rewritten=%r",
                        min(len(prev_user_msgs), int(settings.query_transform_fallback_max_prev_user_messages)),
                        (rewritten or "")[:200] + "..." if len(rewritten or "") > 200 else (rewritten or ""),
                    )
                    query_embedding = await generate_embedding(rewritten)
                    try:
                        results = await hybrid_search_rrf(
                            db,
                            query_text=user_message,
                            query_embedding=query_embedding,
                            top_k=settings.top_k,
                            distance_threshold=settings.distance_threshold,
                            bm25_top_k=getattr(settings, "bm25_top_k", settings.top_k),
                        )
                    except Exception as e:
                        logger.error("Hybrid search failed (fallback rerun): %s", e)
                        results = []
                else:
                    logger.info("[QTRANSFORM] fallback rewrite produced no change; skip rerun.")
            except Exception as e:
                logger.warning("[QTRANSFORM] fallback rewrite failed; skip rerun. err=%s", e)
        else:
            logger.info("[QTRANSFORM] fallback skipped (no previous user messages).")

    logger.info("[CHAT] Retrieved %d chunks for query", len(results))
    for i, r in enumerate(results):
        preview = r.content[:150] + "..." if len(r.content) > 150 else r.content
        logger.info("[CHAT] Chunk[%d] source=%s type=%s | %r", i, r.source, r.source_type, preview)
    logger.debug("[CHAT] Full chunks content: %s", [r.content for r in results])

    context_chunks = [r.content for r in results]
    seen = set()
    sources = []
    for r in results:
        key = (r.source, r.source_type)
        if key not in seen:
            seen.add(key)
            sources.append(SourceInfo(source=r.source, source_type=r.source_type))

    try:
        answer = await chat_completion(llm_client, user_message, conversation_history, context_chunks)
    except Exception as e:
        logger.error("LLM completion failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to get response from the language model.")

    return ChatResponse(response=answer, sources=sources)


@router.post(
    "/debug/search",
    summary="Debug vector search for a query",
    description="Returns the top-K vector candidates with distances for the given message. Useful to tune distance_threshold.",
)
async def debug_search(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    llm_client=Depends(get_groq_client),
):
    query_embedding = await generate_embedding(request.message)
    settings = get_settings()
    candidates = await similarity_candidates(db, query_embedding, top_k=settings.top_k)
    return {
        "message": request.message,
        "top_k": settings.top_k,
        "distance_threshold": settings.distance_threshold,
        "candidates": [
            {
                "rank": i,
                "distance": c["distance"],
                "passed_threshold": c["distance"] < settings.distance_threshold,
                "source": c["source"],
                "source_type": c["source_type"],
                "metadata": c["metadata"],
                "content_preview": (c["content"][:200] + "...") if len(c["content"]) > 200 else c["content"],
            }
            for i, c in enumerate(candidates)
        ],
    }
