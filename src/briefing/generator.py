"""Briefing generator for daily, weekly, and monthly summaries.

Queries Zilliz Cloud for recent documents within the target date range,
searches for schedule-related keywords, deduplicates results, and
uses the Gemini LLM to produce a structured Korean briefing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.db.zilliz_client import get_client
from src.rag.retriever import Retriever
from src.rag.generator import Generator

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_TYPE_LABELS: dict[str, str] = {
    "daily": "일간",
    "weekly": "주간",
    "monthly": "월간",
}

_SCHEDULE_KEYWORDS: list[str] = [
    "일정",
    "마감",
    "deadline",
    "회의",
    "미팅",
    "예정",
]

_KST = timezone(timedelta(hours=9))

_BRIEFING_PROMPT_TEMPLATE = """\
다음은 {start}~{end} 기간의 업무 문서/이메일 목록입니다.

{documents}

위 내용을 바탕으로 다음 형식의 업무 브리핑을 작성하세요:

📋 {type_label} 브리핑 ({date})

[지난 기간 업무 요약]
• 주요 활동 3~5개

[향후 할 일]
⚠️ 마감 임박 항목
• 예정된 업무 목록

[기타 참고사항]
• 중요 공지나 변경 사항

800자 이내로 작성하세요."""


# ------------------------------------------------------------------
# BriefingGenerator
# ------------------------------------------------------------------


class BriefingGenerator:
    """Generate periodic business briefings from indexed documents."""

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.generator = Generator()
        self.client = get_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, briefing_type: str = "daily") -> str:
        """Generate a briefing for the given period type.

        Parameters
        ----------
        briefing_type:
            One of ``"daily"``, ``"weekly"``, or ``"monthly"``.

        Returns
        -------
        str
            The generated briefing text, or a notice when no documents
            were found for the period.
        """
        if briefing_type not in _TYPE_LABELS:
            raise ValueError(
                f"Invalid briefing_type '{briefing_type}'. "
                f"Must be one of {list(_TYPE_LABELS.keys())}."
            )

        start, end = self._get_date_range(briefing_type)
        logger.info(
            "Generating %s briefing for %s ~ %s",
            briefing_type,
            start.isoformat(),
            end.isoformat(),
        )

        # 1. Fetch documents within the date range
        date_docs = self._get_documents_by_date(start, end)

        # 2. Search for schedule-related documents via vector similarity
        schedule_docs = self._search_schedule_keywords(start)

        # 3. Merge and deduplicate
        merged = self._deduplicate(date_docs + schedule_docs)

        if not merged:
            msg = "해당 기간에 새로운 문서/메일이 없습니다."
            logger.info(msg)
            self._save_briefing(briefing_type, msg)
            return msg

        # 4. Limit to avoid exceeding LLM context window
        merged = merged[:20]

        # 5. Build the LLM prompt
        type_label = _TYPE_LABELS[briefing_type]
        now_kst = datetime.now(_KST)
        documents_text = self._format_documents(merged)

        prompt = _BRIEFING_PROMPT_TEMPLATE.format(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            documents=documents_text,
            type_label=type_label,
            date=now_kst.strftime("%Y-%m-%d %a"),
        )

        # 6. Call the LLM
        try:
            content = await self.generator._call_with_fallback(prompt)
        except Exception:
            logger.exception("Failed to generate briefing via LLM")
            content = (
                "브리핑 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
            )

        # 7. Persist
        self._save_briefing(briefing_type, content)

        logger.info(
            "%s briefing generated (%d chars)", briefing_type, len(content)
        )
        return content

    # ------------------------------------------------------------------
    # Date range
    # ------------------------------------------------------------------

    @staticmethod
    def _get_date_range(
        briefing_type: str,
    ) -> tuple[datetime, datetime]:
        """Return ``(start, end)`` datetimes for the given briefing type.

        All datetimes are in KST (UTC+9).
        """
        now = datetime.now(_KST)

        if briefing_type == "daily":
            start = now - timedelta(days=1)
        elif briefing_type == "weekly":
            start = now - timedelta(days=7)
        elif briefing_type == "monthly":
            start = now - timedelta(days=30)
        else:
            start = now - timedelta(days=1)

        return start, now

    # ------------------------------------------------------------------
    # Document fetching
    # ------------------------------------------------------------------

    def _get_documents_by_date(
        self,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Query the documents collection filtered by ``created_date`` range.

        Returns up to 50 documents ordered by recency.
        """
        try:
            start_iso = start.isoformat()
            end_iso = end.isoformat()
            results = self.client.query(
                collection_name="documents",
                filter=f'created_date >= "{start_iso}" and created_date <= "{end_iso}"',
                output_fields=[
                    "source_type", "content", "filename",
                    "email_subject", "email_from", "created_date",
                ],
                limit=50,
            )
            # Sort by created_date descending (Milvus query doesn't support ORDER BY)
            results.sort(
                key=lambda x: x.get("created_date", ""),
                reverse=True,
            )
            logger.info(
                "Fetched %d documents by date range (%s ~ %s)",
                len(results),
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            return results
        except Exception:
            logger.exception("Failed to fetch documents by date range")
            return []

    def _search_schedule_keywords(
        self,
        after_date: datetime,
    ) -> list[dict[str, Any]]:
        """Search for schedule-related documents via vector similarity.

        Queries each keyword in :data:`_SCHEDULE_KEYWORDS` and collects
        up to 3 results per keyword, filtered to documents after
        *after_date*.
        """
        all_results: list[dict[str, Any]] = []
        after_iso = after_date.isoformat()

        for keyword in _SCHEDULE_KEYWORDS:
            try:
                results = self.retriever.search(
                    query=keyword,
                    after_date=after_iso,
                    top_k=3,
                )
                all_results.extend(results)
            except Exception:
                logger.warning(
                    "Schedule keyword search failed for '%s'",
                    keyword,
                    exc_info=True,
                )

        logger.info(
            "Schedule keyword search returned %d total results",
            len(all_results),
        )
        return all_results

    # ------------------------------------------------------------------
    # Deduplication & formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate documents based on ``id`` or content prefix.

        When an ``id`` field is present it is used as the dedup key;
        otherwise the first 50 characters of ``content`` are used as a
        fallback fingerprint.
        """
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []

        for doc in docs:
            # Prefer id, fall back to content prefix
            doc_id = doc.get("id")
            if doc_id is not None:
                key = f"id:{doc_id}"
            else:
                content = doc.get("content", "")
                key = f"content:{content[:50]}"

            if key in seen:
                continue
            seen.add(key)
            unique.append(doc)

        logger.debug(
            "Deduplicated %d -> %d documents", len(docs), len(unique)
        )
        return unique

    @staticmethod
    def _format_documents(docs: list[dict[str, Any]]) -> str:
        """Format a list of document dicts into numbered text for the prompt."""
        if not docs:
            return "(문서 없음)"

        parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            source_type = doc.get("source_type", "")
            content = (doc.get("content") or "").strip()

            if source_type == "dropbox":
                label = doc.get("filename") or "파일"
                source_label = f"[파일: {label}]"
            elif source_type == "email":
                subject = doc.get("email_subject") or "제목 없음"
                sender = doc.get("email_from") or ""
                source_label = f"[이메일: {subject} - {sender}]"
            else:
                source_label = f"[{source_type}]"

            created = doc.get("created_date", "")
            date_str = ""
            if created:
                date_str = f" ({str(created)[:10]})"

            # Truncate overly long content to keep prompt manageable
            if len(content) > 300:
                content = content[:300] + "..."

            parts.append(f"{idx}. {source_label}{date_str}\n{content}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_briefing(self, briefing_type: str, content: str) -> None:
        """Insert the generated briefing into the ``briefings`` collection."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            self.client.insert(
                collection_name="briefings",
                data=[{
                    "briefing_type": briefing_type,
                    "content": content[:10000],  # VARCHAR limit
                    "generated_at": now,
                    "sent": False,
                    "_dummy_vec": [0.0, 0.0],
                }],
            )
            logger.info("Briefing saved to database (type=%s)", briefing_type)
        except Exception:
            logger.exception(
                "Failed to save briefing to database (type=%s)", briefing_type
            )
