"""Vector search retriever backed by Zilliz Cloud (Milvus).

Embeds the user query via :class:`Embedder`, performs ANN search on the
``documents`` collection in Zilliz, and provides helpers for formatting
results into LLM context and extracting source citations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.db.zilliz_client import get_client
from src.rag.embedder import Embedder

logger = logging.getLogger(__name__)

_OUTPUT_FIELDS = [
    "source_type",
    "source_id",
    "content",
    "filename",
    "folder_path",
    "email_subject",
    "email_from",
    "created_date",
]


class Retriever:
    """Retrieve relevant document chunks from Zilliz via vector similarity."""

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
            Each dict has keys: ``id``, ``source_type``, ``content``,
            ``filename``, ``email_subject``, ``email_from``,
            ``created_date``, ``similarity``.
        """
        query_embedding = self.embedder.embed(query)

        # Build filter expression
        filter_parts: list[str] = []
        if source_type:
            filter_parts.append(f'source_type == "{source_type}"')
        if after_date:
            filter_parts.append(f'created_date >= "{after_date}"')

        filter_expr = " and ".join(filter_parts) if filter_parts else ""

        try:
            raw_results = self.client.search(
                collection_name="documents",
                data=[query_embedding],
                filter=filter_expr if filter_expr else None,
                limit=top_k,
                output_fields=_OUTPUT_FIELDS,
                search_params={"metric_type": "COSINE"},
            )

            # Milvus returns list of list of hits; we use the first query's results
            results: list[dict[str, Any]] = []
            if raw_results and len(raw_results) > 0:
                for hit in raw_results[0]:
                    doc = {
                        "id": hit["id"],
                        "similarity": hit["distance"],
                    }
                    entity = hit.get("entity", hit)
                    for field in _OUTPUT_FIELDS:
                        doc[field] = entity.get(field, None)
                    results.append(doc)

        except Exception:
            logger.exception("Zilliz search failed for query: %s", query[:80])
            return []

        logger.info(
            "Retrieved %d results for query (top_k=%d): %s",
            len(results),
            top_k,
            query[:60],
        )
        return results

    # ------------------------------------------------------------------
    # Date-range query (no vector search)
    # ------------------------------------------------------------------

    def search_by_date_range(
        self,
        date_field: str,
        start_date: str,
        end_date: str,
        source_type: str | None = None,
        limit: int = 30,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Query documents by date range without vector similarity.

        Parameters
        ----------
        date_field:
            Field to filter on -- ``"created_date"`` or ``"updated_date"``.
        start_date:
            ISO-8601 start date (inclusive).
        end_date:
            ISO-8601 end date (inclusive).
        source_type:
            Optional ``"dropbox"`` or ``"email"`` filter.
        limit:
            Maximum results (default 30).
        output_fields:
            Fields to return.  Defaults to :data:`_OUTPUT_FIELDS` plus
            ``updated_date`` and ``email_date``.

        Returns
        -------
        list[dict]
            Documents matching the date range, sorted by *date_field*
            descending.  Only ``chunk_index == 0`` rows are returned so
            each document appears at most once.
        """
        if output_fields is None:
            output_fields = _OUTPUT_FIELDS + ["updated_date", "email_date", "folder_path"]

        parts = [
            f'{date_field} >= "{start_date}"',
            f'{date_field} <= "{end_date}"',
            "chunk_index == 0",
        ]
        if source_type:
            parts.append(f'source_type == "{source_type}"')

        filter_expr = " and ".join(parts)

        try:
            results = self.client.query(
                collection_name="documents",
                filter=filter_expr,
                output_fields=output_fields,
                limit=limit,
            )
            results.sort(
                key=lambda x: x.get(date_field, ""),
                reverse=True,
            )
            logger.info(
                "Date-range query (%s %s~%s, type=%s): %d results",
                date_field,
                start_date[:10],
                end_date[:10],
                source_type or "all",
                len(results),
            )
            return results
        except Exception:
            logger.exception(
                "Date-range query failed (%s %s~%s)",
                date_field, start_date[:10], end_date[:10],
            )
            return []

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
                f"[문서 {idx}] {label}{sim_str}\n{(doc.get('content') or '').strip()}"
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
    if value is None or value == "":
        return "날짜 없음"

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    # value is likely an ISO string
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(value)[:10]
