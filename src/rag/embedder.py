"""Gemini text-embedding-004 wrapper (singleton).

Uses the Google GenAI SDK to produce 768-dimensional embeddings via
the ``text-embedding-004`` model (configurable through ``settings``).
The class is a singleton so the underlying client is only created once.
"""

from __future__ import annotations

import logging
from typing import Sequence

from google import genai

from src.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """Singleton embedding wrapper around Gemini text-embedding-004."""

    _instance: Embedder | None = None

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def __new__(cls) -> Embedder:
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._client = genai.Client(api_key=settings.gemini_api_key)
            instance._model = settings.gemini_embedding_model
            logger.info(
                "Embedder initialised with model=%s (dim=%d)",
                instance._model,
                settings.embedding_dim,
            )
            cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for a single text string.

        Parameters
        ----------
        text:
            The input text to embed.  Empty / whitespace-only strings are
            normalised to a single space to avoid API errors.

        Returns
        -------
        list[float]
            A list of floats with length ``settings.embedding_dim`` (768).
        """
        if not text or not text.strip():
            text = " "

        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return list(result.embeddings[0].values)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts, returning one vector per input.

        Currently calls :meth:`embed` in a loop.  The Gemini embedding
        endpoint does not yet support true batching, so this is a
        convenience wrapper that keeps the interface consistent.

        Parameters
        ----------
        texts:
            An iterable of strings to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text.
        """
        return [self.embed(t) for t in texts]
