import json
import logging

from groq import AsyncGroq

from app.config import Settings

logger = logging.getLogger(__name__)


_SPLIT_PROMPT = """<role>
You are a query transformation engine for a RAG system. Your task is to split the user's text into one or more standalone questions.
</role>

<rules>
- Keep the SAME language as the user input.
- Do NOT add facts or constraints that are not in the input.
- If the input contains multiple different questions, split them into separate questions.
- If the input is already one clear question, return a single-item list containing it.
- If the input is not a question but still a request, convert it into a single clear question without changing meaning.
- Return ONLY a valid JSON object, no markdown.
</rules>

<output_schema>
{
  "questions": ["string", "..."]
}
</output_schema>"""


_CLARIFY_PROMPT = """<role>
You rewrite user questions for better retrieval in a RAG system.
</role>

<rules>
- Keep the SAME language as the input question.
- Make it clear and specific for search, but do NOT change meaning.
- Do NOT add new facts, names, numbers, dates, or constraints.
- Return ONLY a valid JSON object, no markdown.
</rules>

<output_schema>
{
  "question": "string"
}
</output_schema>"""

_CONTEXT_TRANSFORM_PROMPT = """<role>
You rewrite the user's current message into a standalone search query for a RAG system, using previous user messages only for disambiguation.
</role>

<rules>
- Keep the SAME language as the current message.
- Only use the provided previous user messages to resolve references (e.g., "it", "this", "there").
- Do NOT add new facts, names, numbers, dates, or constraints that are not implied by the messages.
- Output should be a single concise question suitable for semantic search.
- Return ONLY a valid JSON object, no markdown.
</rules>

<output_schema>
{
  "query": "string"
}
</output_schema>"""


def _safe_preview(text: str, limit: int = 200) -> str:
    t = (text or "").strip()
    return t if len(t) <= limit else t[:limit] + "..."


async def split_and_clarify_query(
    client: AsyncGroq,
    user_message: str,
    *,
    settings: Settings,
) -> list[str]:
    """
    Returns a list of clarified questions. Best-effort:
    - On any LLM/parsing error, returns [user_message].
    """
    user_message = (user_message or "").strip()
    if not user_message:
        return [""]

    max_q = int(settings.query_transform_max_questions)

    # 1) Split
    try:
        resp = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": _SPLIT_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=512,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        questions = data.get("questions", [])
        if not isinstance(questions, list):
            questions = []
        questions = [str(q).strip() for q in questions if str(q).strip()]
    except Exception as e:
        logger.warning("[QTRANSFORM] split failed; fallback to original. err=%s", e)
        questions = [user_message]

    if not questions:
        questions = [user_message]

    # de-dup & cap
    seen: set[str] = set()
    deduped: list[str] = []
    for q in questions:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(q)
        if len(deduped) >= max_q:
            break
    questions = deduped

    logger.info(
        "[QTRANSFORM] split -> %d question(s). original=%r first=%r",
        len(questions),
        _safe_preview(user_message),
        _safe_preview(questions[0]) if questions else "",
    )

    # 2) Clarify each question
    clarified: list[str] = []
    for i, q in enumerate(questions):
        try:
            resp = await client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": _CLARIFY_PROMPT},
                    {"role": "user", "content": q},
                ],
                temperature=0,
                max_tokens=256,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            cq = str(data.get("question", "")).strip()
            clarified.append(cq if cq else q)
        except Exception as e:
            logger.warning("[QTRANSFORM] clarify failed; keep split question. i=%d err=%s", i, e)
            clarified.append(q)

    if not clarified:
        clarified = [user_message]

    logger.info(
        "[QTRANSFORM] clarify -> %d question(s). first=%r",
        len(clarified),
        _safe_preview(clarified[0]),
    )

    return clarified


async def transform_query_with_context(
    client: AsyncGroq,
    current_message: str,
    previous_user_messages: list[str],
    *,
    settings: Settings,
) -> str:
    """
    Best-effort rewrite of current_message into a standalone retrieval query using previous_user_messages.
    On error, returns current_message.
    """
    current_message = (current_message or "").strip()
    if not current_message:
        return ""

    prev = [str(m).strip() for m in (previous_user_messages or []) if str(m).strip()]
    if not prev:
        return current_message

    # keep only last N
    max_prev = int(getattr(settings, "query_transform_fallback_max_prev_user_messages", 5))
    prev = prev[-max_prev:]

    try:
        resp = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": _CONTEXT_TRANSFORM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_message": current_message,
                            "previous_user_messages": prev,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0,
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        q = str(data.get("query", "")).strip()
        rewritten = q if q else current_message
    except Exception as e:
        logger.warning("[QTRANSFORM] context-rewrite failed; keep original. err=%s", e)
        rewritten = current_message

    logger.info(
        "[QTRANSFORM] context-rewrite used prev=%d. current=%r rewritten=%r",
        len(prev),
        _safe_preview(current_message),
        _safe_preview(rewritten),
    )
    return rewritten

