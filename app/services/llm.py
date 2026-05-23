import json
import logging

from groq import AsyncGroq

from app.config import get_settings

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """<role>
You are a helpful assistant in a Retrieval-Augmented Generation (RAG) system.
</role>

<core_objectives>
1. Answer the user's question using ONLY the information in the provided <context>.
2. If the <context> does not contain enough information, say so plainly and ask the user to ingest more documents or provide more details.
</core_objectives>

<context>
{context}
</context>"""

async def chat_completion(
    client: AsyncGroq,
    user_message: str,
    conversation_history: list[dict],
    context_chunks: list[str],
) -> str:
    settings = get_settings()

    if context_chunks:
        context = "\n\n---\n\n".join(context_chunks)
        logger.info("[CHAT] Chunks in LLM prompt (%d total):", len(context_chunks))
        for i, chunk in enumerate(context_chunks):
            logger.info("[CHAT] --- Chunk[%d] ---\n%s", i, chunk)
    else:
        context = "No relevant context found in the knowledge base."
        logger.info("[CHAT] No chunks in LLM prompt (empty context)")

    system_message = {
        "role": "system",
        "content": RAG_SYSTEM_PROMPT.format(context=context),
    }

    messages = [system_message]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    system_len = len(messages[0]["content"]) if messages else 0
    user_preview = user_message[:200] + "..." if len(user_message) > 200 else user_message
    logger.info(
        "[CHAT] Prompt to LLM: %d messages | system_len=%d chars | user_msg=%r",
        len(messages),
        system_len,
        user_preview,
    )
    logger.debug(
        "[CHAT] Full prompt (messages): %s",
        json.dumps(
            [
                {"role": m["role"], "content_len": len(m.get("content", ""))}
                for m in messages
            ],
            indent=2,
        ),
    )
    logger.debug("[CHAT] Full system+context (first message): %s", messages[0]["content"][:2000] if messages else "")

    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )

    answer = response.choices[0].message.content
    logger.info(
        "[CHAT] LLM response: len=%d chars | preview=%r",
        len(answer),
        answer[:300] + "..." if len(answer) > 300 else answer,
    )
    logger.debug("[CHAT] Full LLM response: %s", answer)

    return answer
