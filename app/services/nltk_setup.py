import logging
import os
from pathlib import Path

import nltk

logger = logging.getLogger(__name__)


def ensure_nltk_stopwords() -> None:
    """
    Ensure NLTK 'stopwords' corpus is available.
    Downloads once into a local data directory (default: project-root/.nltk_data)
    unless NLTK_DATA env var is set.
    """
    base_dir = Path(__file__).resolve().parents[2]  # project root (folder containing 'app/')
    data_dir = Path(os.getenv("NLTK_DATA", str(base_dir / ".nltk_data")))
    os.environ["NLTK_DATA"] = str(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        nltk.data.find("corpora/stopwords")
        logger.info("[NLTK] stopwords corpus found. NLTK_DATA=%s", str(data_dir))
        return
    except LookupError:
        logger.info("[NLTK] stopwords corpus missing; downloading. NLTK_DATA=%s", str(data_dir))

    try:
        nltk.download("stopwords", download_dir=str(data_dir), quiet=True)
        nltk.data.find("corpora/stopwords")
        logger.info("[NLTK] stopwords corpus ready. NLTK_DATA=%s", str(data_dir))
    except Exception as e:
        # Best-effort: BM25/stopwords are an optimization; do not fail app startup.
        logger.warning(
            "[NLTK] stopwords unavailable; continuing without it. err=%s NLTK_DATA=%s",
            e,
            str(data_dir),
        )
        return

