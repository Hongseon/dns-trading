"""RAG orchestration chain: retrieve, generate, format.

Provides :class:`RAGChain` which ties together the retriever and generator
into a single ``run()`` call, and a module-level singleton accessor
:func:`get_chain` for use throughout the application.
"""

from __future__ import annotations

import asyncio
import logging

from src.rag.generator import Generator
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# KakaoTalk simple text response limit
_KAKAO_MAX_CHARS = 1000


class RAGChain:
    """End-to-end RAG pipeline: retrieve relevant chunks then generate an answer."""

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.generator = Generator()

    # ------------------------------------------------------------------
    # Full RAG run
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        source_type: str | None = None,
        after_date: str | None = None,
        top_k: int = 5,
    ) -> str:
        """Execute the full RAG pipeline.

        1. Retrieve the top-*k* relevant document chunks.
        2. Format them into LLM context.
        3. Extract source citations.
        4. Generate an answer via the LLM.
        5. Truncate to the KakaoTalk 1 000-character limit.

        Parameters
        ----------
        query:
            User question in natural language.
        source_type:
            Optional filter -- ``"dropbox"`` or ``"email"``.
        after_date:
            Optional ISO-8601 date filter.
        top_k:
            Number of chunks to retrieve (default 5).

        Returns
        -------
        str
            The final answer, guaranteed to be at most 1 000 characters.
        """
        # Step 1 -- Retrieve (synchronous, run in thread to avoid blocking)
        results, context, sources = await asyncio.to_thread(
            self.retriever.search_and_prepare,
            query,
            source_type,
            after_date,
            top_k,
        )

        # Guard: no documents found
        if not results:
            logger.info("No results found for query: %s", query[:60])
            return "관련 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보세요."

        # Step 2 -- Generate
        answer = await self.generator.generate(query, context, sources)

        # Step 3 -- Truncate to KakaoTalk limit
        answer = _truncate(answer, _KAKAO_MAX_CHARS)

        logger.info(
            "RAG run complete (%d results, %d char answer) for: %s",
            len(results),
            len(answer),
            query[:60],
        )
        return answer

    # ------------------------------------------------------------------
    # Quick RAG (non-callback mode, tighter budget)
    # ------------------------------------------------------------------

    async def quick_run(self, query: str) -> str:
        """Quick RAG for the non-callback code path.

        Uses fewer retrieved chunks (``top_k=3``) to reduce latency so
        the response has a better chance of fitting within the 5-second
        KakaoTalk skill timeout.
        """
        return await self.run(query, top_k=3)


# ======================================================================
# Module-level singleton
# ======================================================================

_chain: RAGChain | None = None


def get_chain() -> RAGChain:
    """Return (and lazily create) the shared :class:`RAGChain` instance."""
    global _chain
    if _chain is None:
        _chain = RAGChain()
        logger.info("RAGChain singleton created")
    return _chain


# ======================================================================
# Internal helpers
# ======================================================================


def _truncate(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars*, appending ``...`` if trimmed."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
