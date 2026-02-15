"""Vector search retriever backed by Supabase pgvector.

Embeds the user query via :class:`Embedder`, calls the ``match_documents``
RPC function in Supabase, and provides helpers for formatting results into
LLM context and extracting source citations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.db.supabase_client import get_client
from src.rag.embedder import Embedder

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant document chunks from Supabase via vector similarity."""

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.client = get_client()

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        source_type: str | None = None,
        after_date: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Embed *query* and perform vector similarity search.

        Parameters
        ----------
        query:
            Natural-language user question.
        source_type:
            Optional filter -- ``"dropbox"`` or ``"email"``.  ``None`` searches
            all sources.
        after_date:
            Optional ISO-8601 date string (e.g. ``"2025-01-01"``).  Only
            documents with ``created_date >= after_date`` are returned.
        top_k:
            Maximum number of results to return (default 5).

        Returns
        -------
        list[dict]
            Each dict mirrors the columns returned by the
            ``match_documents`` RPC: ``id``, ``source_type``, ``content``,
            ``filename``, ``email_subject``, ``email_from``,
            ``created_date``, ``similarity``.
        """
        query_embedding = self.embedder.embed(query)

        rpc_params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "filter_source_type": source_type,
            "filter_after_date": after_date,
        }

        try:
            response = self.client.rpc("match_documents", rpc_params).execute()
            results: list[dict[str, Any]] = response.data or []
        except Exception:
            logger.exception("match_documents RPC failed for query: %s", query[:80])
            return []

        logger.info(
            "Retrieved %d results for query (top_k=%d): %s",
            len(results),
            top_k,
            query[:60],
        )
        return results

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def format_context(self, results: list[dict[str, Any]]) -> str:
        """Format retrieved documents into a numbered context string for the LLM.

        Each entry includes a sequential number, a source label, and the
        chunk content.  The label is ``[파일: filename]`` for Dropbox
        documents or ``[이메일: subject - sender]`` for emails.

        Parameters
        ----------
        results:
            List of result dicts as returned by :meth:`search`.

        Returns
        -------
        str
            A multi-line string ready to be inserted into the LLM prompt as
            context.  Returns an empty string when *results* is empty.
        """
        if not results:
            return ""

        parts: list[str] = []
        for idx, doc in enumerate(results, start=1):
            label = self._make_source_label(doc)
            similarity = doc.get("similarity")
            sim_str = f" (유사도: {similarity:.2f})" if similarity is not None else ""

            parts.append(
                f"[문서 {idx}] {label}{sim_str}\n{doc.get('content', '').strip()}"
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Source extraction
    # ------------------------------------------------------------------

    def extract_sources(self, results: list[dict[str, Any]]) -> list[str]:
        """Extract deduplicated human-readable source citations.

        Parameters
        ----------
        results:
            List of result dicts as returned by :meth:`search`.

        Returns
        -------
        list[str]
            Unique citation strings, e.g.
            ``"Dropbox/계약서/A사_계약.pdf"`` or
            ``'이메일 - "회의 안건" (2025-02-10)'``.
        """
        seen: set[str] = set()
        sources: list[str] = []

        for doc in results:
            citation = self._make_citation(doc)
            if citation and citation not in seen:
                seen.add(citation)
                sources.append(citation)

        return sources

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_source_label(doc: dict[str, Any]) -> str:
        """Build the inline source label shown in context."""
        source_type = doc.get("source_type", "")

        if source_type == "dropbox":
            filename = doc.get("filename") or "알 수 없는 파일"
            return f"[파일: {filename}]"

        if source_type == "email":
            subject = doc.get("email_subject") or "제목 없음"
            sender = doc.get("email_from") or "알 수 없음"
            return f"[이메일: {subject} - {sender}]"

        # Fallback for unknown source types
        return f"[{source_type}]"

    @staticmethod
    def _make_citation(doc: dict[str, Any]) -> str:
        """Build a human-readable citation string for the sources list."""
        source_type = doc.get("source_type", "")

        if source_type == "dropbox":
            filename = doc.get("filename") or "알 수 없는 파일"
            folder = doc.get("folder_path") or ""
            # Build a path like "Dropbox/계약서/A사.pdf"
            if folder:
                folder = folder.strip("/")
                return f"Dropbox/{folder}/{filename}"
            return f"Dropbox/{filename}"

        if source_type == "email":
            subject = doc.get("email_subject") or "제목 없음"
            created = doc.get("created_date")
            date_str = _format_date(created)
            return f'이메일 - "{subject}" ({date_str})'

        return ""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def search_and_prepare(
        self,
        query: str,
        source_type: str | None = None,
        after_date: str | None = None,
        top_k: int = 5,
    ) -> tuple[list[dict[str, Any]], str, list[str]]:
        """Run search, format context, and extract sources in one call.

        Returns
        -------
        tuple
            ``(results, context_str, sources_list)``
        """
        results = self.search(
            query, source_type=source_type, after_date=after_date, top_k=top_k
        )
        context = self.format_context(results)
        sources = self.extract_sources(results)
        return results, context, sources


# ======================================================================
# Module-level helpers
# ======================================================================


def _format_date(value: Any) -> str:
    """Best-effort date formatting to ``YYYY-MM-DD``."""
    if value is None:
        return "날짜 없음"

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    # value is likely an ISO string from Supabase
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(value)[:10]
