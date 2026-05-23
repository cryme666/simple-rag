import asyncio

from sentence_transformers import SentenceTransformer

from app.config import get_settings

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        settings = get_settings()
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def warmup_embedding_model() -> None:
    """
    Trigger model download/load at startup. Safe to call multiple times.
    """
    _ = get_embedding_model()


async def generate_embedding(text: str) -> list[float]:
    model = get_embedding_model()
    vec = await asyncio.to_thread(
        model.encode,
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.tolist()


async def generate_embeddings_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    vectors = await asyncio.to_thread(
        model.encode,
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]
