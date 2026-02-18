"""Gemini embedding wrapper (singleton).

Uses the Google GenAI SDK to produce 768-dimensional embeddings via
the ``gemini-embedding-001`` model (configurable through ``settings``).

Paid tier: 1,500+ RPM. We batch up to 100 texts per API call with
a 1-second pause between calls and retry on 429.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Sequence

from google import genai
from google.genai import types

from src.config import settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100  # max texts per single API call
_MIN_INTERVAL = 1.0  # seconds between API calls
_MAX_RETRIES = 8
_API_TIMEOUT = 60  # seconds per API call before timeout


class Embedder:
    """Singleton embedding wrapper around Gemini embedding model."""

    _instance: Embedder | None = None

    def __new__(cls) -> Embedder:
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._client = genai.Client(api_key=settings.gemini_api_key)
            instance._model = settings.gemini_embedding_model
            instance._last_call_time = 0.0
            logger.info(
                "Embedder initialised with model=%s (dim=%d)",
                instance._model,
                settings.embedding_dim,
            )
            cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pace(self) -> None:
        """Ensure minimum interval between API calls."""
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_call_time = time.monotonic()

    def _call_embed_api(self, contents) -> list[list[float]]:
        """Call embed_content with retry on 429 and timeout protection."""
        for attempt in range(_MAX_RETRIES):
            self._pace()
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(
                    self._client.models.embed_content,
                    model=self._model,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        output_dimensionality=settings.embedding_dim,
                    ),
                )
                result = future.result(timeout=_API_TIMEOUT)
                executor.shutdown(wait=False)
                return [list(e.values) for e in result.embeddings]
            except FuturesTimeoutError:
                logger.warning(
                    "Embedding API timeout (attempt %d/%d, %ds)",
                    attempt + 1, _MAX_RETRIES, _API_TIMEOUT,
                )
                executor.shutdown(wait=False)
                time.sleep(5)
            except Exception as e:
                executor.shutdown(wait=False)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = min(2 ** attempt * 5, 65)
                    logger.warning(
                        "Rate limited (attempt %d/%d), waiting %ds",
                        attempt + 1, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Embedding failed after max retries")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for a single text string."""
        if not text or not text.strip():
            text = " "
        return self._call_embed_api(text)[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts in batches of 100 per API call."""
        cleaned = [t if t and t.strip() else " " for t in texts]
        all_vectors: list[list[float]] = []

        for start in range(0, len(cleaned), _BATCH_SIZE):
            batch = cleaned[start : start + _BATCH_SIZE]
            vectors = self._call_embed_api(batch)
            all_vectors.extend(vectors)
            if len(cleaned) > _BATCH_SIZE:
                logger.info(
                    "Embedded %d/%d chunks", len(all_vectors), len(cleaned)
                )

        return all_vectors
