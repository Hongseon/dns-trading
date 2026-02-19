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

        Uses fewer retrieved chunks (``top_k=2``) and a shorter max token
        limit to fit within the 5-second KakaoTalk skill timeout.
        """
        # Step 1 -- Retrieve with minimal chunks
        results, context, sources = await asyncio.to_thread(
            self.retriever.search_and_prepare, query, None, None, 2
        )
        if not results:
            return "관련 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보세요."

        # Step 2 -- Generate with shorter output
        answer = await self.generator.generate_quick(query, context, sources)

        # Step 3 -- Truncate
        return _truncate(answer, _KAKAO_MAX_CHARS)


# ======================================================================
# Module-level singleton
# ======================================================================

    def search_only(self, query: str, top_k: int = 3) -> str:
        """Return formatted search results without LLM generation.

        Used as a fast fallback when the full RAG pipeline times out.
        """
        results = self.retriever.search(query, top_k=top_k)
        if not results:
            return "관련 문서를 찾을 수 없습니다."

        parts: list[str] = []
        for idx, doc in enumerate(results, start=1):
            src_type = doc.get("source_type", "")
            if src_type == "dropbox":
                label = doc.get("filename") or "파일"
            else:
                label = doc.get("email_subject") or "이메일"
            content = (doc.get("content") or "")[:150].strip()
            parts.append(f"{idx}. [{label}]\n{content}")

        header = f"검색 결과 ({len(results)}건):\n\n"
        return header + "\n\n".join(parts)


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
