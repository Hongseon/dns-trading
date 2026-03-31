"""Briefing generator for daily, weekly, and monthly summaries.

Queries Zilliz Cloud for recent Dropbox file changes and email activity
within a calendar-based date range, and uses the Gemini LLM to produce
a structured Korean briefing with separate sections for files and emails.
"""

from __future__ import annotations

import asyncio
import logging
import time
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from typing import Any

from src.db.zilliz_client import get_client
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.server import chat_logger

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_TYPE_LABELS: dict[str, str] = {
    "daily": "오늘",
    "yesterday": "어제",
    "weekly": "이번 주",
    "last_week": "지난 주",
    "monthly": "이번 달",
    "last_month": "지난 달",
}

# DnS staff email addresses (used to distinguish sent vs received)
_DNS_STAFF_EMAILS: set[str] = {
    "theking57@naver.com",
    "ruthkim2015@naver.com",
}

_KST = timezone(timedelta(hours=9))


def _format_datetime(raw: str) -> str:
    """Format an ISO datetime string as 'MM/DD HH:MM'."""
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(str(raw))
        return dt.strftime("%m/%d %H:%M")
    except (ValueError, TypeError):
        return str(raw)[:16]

# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_DAILY_PROMPT = """\
다음은 {date_label}({date_range})의 업무 활동 데이터입니다.

== 변동된 파일 ({file_count}건) ==
{files_section}

== 받은 메일 ({received_count}건) ==
{received_section}

== 보낸 메일 ({sent_count}건) ==
{sent_section}

위 데이터를 분석하여 다음 형식으로 업무 브리핑을 작성하세요:

📋 {date_label} 업무 브리핑 ({date})

[파일 변동 사항]
• 새로 추가/수정된 파일과 주요 내용 요약

[받은 메일 요약]
• 외부에서 수신한 주요 메일의 핵심 내용

[보낸 메일 요약]
• DnS 직원이 발신한 주요 메일의 핵심 내용

[할 일 / 주요 일정]
• 파일과 메일 내용에서 파악되는 마감일, 일정, 할 일 항목

[참고사항]
• 기타 중요 사항

규칙:
- 데이터가 없는 섹션은 "해당 없음"으로 표시
- 할 일/일정은 파일과 메일 내용에서만 추출 (외부 데이터 사용 금지)
- 900자 이내로 작성
- 한국어로 작성"""

_WEEKLY_PROMPT = """\
다음은 {date_label}({date_range})의 업무 활동 데이터입니다.

== 변동된 파일 ({file_count}건) ==
{files_section}

== 받은 메일 ({received_count}건) ==
{received_section}

== 보낸 메일 ({sent_count}건) ==
{sent_section}

위 데이터를 분석하여 다음 형식으로 업무 브리핑을 작성하세요:

📋 {date_label} 업무 브리핑 ({date})

[주요 활동]
• 주요 파일 작업 및 메일 활동 요약 (3~5개)

[받은 메일 요약]
• 외부에서 수신한 주요 메일의 핵심 내용

[보낸 메일 요약]
• DnS 직원이 발신한 주요 메일의 핵심 내용

[프로젝트별 진행 상황]
• 프로젝트/계약 단위로 진행 상황 정리

[할 일 / 주요 일정]
• 파일과 메일 내용에서 파악되는 마감일, 일정, 할 일 항목

[참고사항]
• 기타 중요 사항

규칙:
- 데이터가 없는 섹션은 "해당 없음"으로 표시
- 할 일/일정은 파일과 메일 내용에서만 추출 (외부 데이터 사용 금지)
- 900자 이내로 작성
- 한국어로 작성"""

_MONTHLY_PROMPT = """\
다음은 {date_label}({date_range})의 업무 활동 데이터입니다.

== 변동된 파일 ({file_count}건) ==
{files_section}

== 받은 메일 ({received_count}건) ==
{received_section}

== 보낸 메일 ({sent_count}건) ==
{sent_section}

위 데이터를 분석하여 다음 형식으로 업무 브리핑을 작성하세요:

📋 {date_label} 업무 브리핑 ({date})

[주요 성과]
• 완료된 주요 업무 (3~5개)

[받은 메일 요약]
• 외부에서 수신한 주요 메일의 핵심 내용

[보낸 메일 요약]
• DnS 직원이 발신한 주요 메일의 핵심 내용

[프로젝트별 진행 현황]
• 프로젝트/계약 단위 현황 정리

[할 일 / 주요 일정]
• 파일과 메일 내용에서 파악되는 향후 마감일, 일정 항목

[참고사항]
• 기타 중요 사항

규칙:
- 데이터가 없는 섹션은 "해당 없음"으로 표시
- 할 일/일정은 파일과 메일 내용에서만 추출 (외부 데이터 사용 금지)
- 900자 이내로 작성
- 한국어로 작성"""

# Map each briefing type to its prompt template
_PROMPTS: dict[str, str] = {
    "daily": _DAILY_PROMPT,
    "yesterday": _DAILY_PROMPT,
    "weekly": _WEEKLY_PROMPT,
    "last_week": _WEEKLY_PROMPT,
    "monthly": _MONTHLY_PROMPT,
    "last_month": _MONTHLY_PROMPT,
}


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
            One of ``"daily"``, ``"yesterday"``, ``"weekly"``,
            ``"last_week"``, ``"monthly"``, or ``"last_month"``.

        Returns
        -------
        str
            The generated briefing text.
        """
        t_start = time.monotonic()

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

        # Collect data in categories
        data = self._collect_briefing_data(briefing_type, start, end)

        has_data = (
            data["recent_files"]
            or data["received_emails"]
            or data["sent_emails"]
        )
        if not has_data:
            label = _TYPE_LABELS[briefing_type]
            msg = f"{label} 기간에 새로운 문서/메일이 없습니다."
            logger.info(msg)
            self._save_briefing(briefing_type, msg)
            return msg

        # Build the LLM prompt
        now_kst = datetime.now(_KST)
        prompt = self._build_prompt(briefing_type, data, now_kst, start, end)

        # Call the LLM with briefing-specific settings
        briefing_system = (
            "당신은 업무 브리핑을 작성하는 AI 어시스턴트입니다. "
            "제공된 파일 변동 사항과 이메일 데이터를 분석하여 "
            "구조화된 한국어 업무 브리핑을 작성하세요. "
            "모든 섹션을 빠짐없이 작성하고, 900자 이내로 완성하세요. "
            "할 일/일정 항목은 제공된 데이터 내에서만 추출하세요."
        )
        usage: dict = {}
        try:
            content, usage = await self.generator._call_with_fallback(
                prompt,
                system_instruction=briefing_system,
                max_output_tokens=2048,
            )
        except Exception:
            logger.exception("Failed to generate briefing via LLM")
            content = "브리핑 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

        # Append source citations
        sources = self._format_sources(data)
        if sources:
            content = content.rstrip() + "\n\n" + sources

        # Persist
        self._save_briefing(briefing_type, content)

        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "%s briefing generated (%d chars)", briefing_type, len(content)
        )

        # Fire-and-forget logging
        asyncio.create_task(chat_logger.log_chat(
            query_type="briefing",
            user_query=briefing_type,
            response=content,
            usage=usage,
            response_time_ms=elapsed_ms,
        ))

        return content

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_briefing_data(
        self,
        briefing_type: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, list[dict[str, Any]]]:
        """Collect documents and emails for the briefing period."""
        start_iso = start.isoformat()
        end_iso = end.isoformat()

        file_limit = 15 if briefing_type in ("daily", "yesterday") else 30
        email_limit = 30 if briefing_type in ("daily", "yesterday") else 60

        # 1. Recently changed Dropbox files (by created_date = server_modified)
        recent_files = self.retriever.search_by_date_range(
            date_field="created_date",
            start_date=start_iso,
            end_date=end_iso,
            source_type="dropbox",
            limit=file_limit,
        )

        # 2. Recent emails (by created_date = email date)
        all_emails = self.retriever.search_by_date_range(
            date_field="created_date",
            start_date=start_iso,
            end_date=end_iso,
            source_type="email",
            limit=email_limit,
        )

        # Split into received vs sent based on DnS staff addresses
        received_emails: list[dict[str, Any]] = []
        sent_emails: list[dict[str, Any]] = []
        for email in all_emails:
            sender = (email.get("email_from") or "").lower().strip()
            if sender in _DNS_STAFF_EMAILS:
                sent_emails.append(email)
            else:
                received_emails.append(email)

        return {
            "recent_files": recent_files,
            "received_emails": received_emails,
            "sent_emails": sent_emails,
        }

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        briefing_type: str,
        data: dict[str, list[dict[str, Any]]],
        now_kst: datetime,
        start: datetime,
        end: datetime,
    ) -> str:
        """Build the LLM prompt with separated file/email sections."""
        files_section = self._format_files(data["recent_files"])
        received_section = self._format_emails(data["received_emails"], label="받은")
        sent_section = self._format_emails(data["sent_emails"], label="보낸")

        date_label = _TYPE_LABELS[briefing_type]
        date_range = f"{start.strftime('%m/%d')}~{end.strftime('%m/%d')}"

        template = _PROMPTS[briefing_type]
        return template.format(
            date_label=date_label,
            date_range=date_range,
            file_count=len(data["recent_files"]),
            files_section=files_section,
            received_count=len(data["received_emails"]),
            received_section=received_section,
            sent_count=len(data["sent_emails"]),
            sent_section=sent_section,
            date=now_kst.strftime("%Y-%m-%d %a"),
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_files(docs: list[dict[str, Any]]) -> str:
        """Format Dropbox files as 'filename (folder) - date'."""
        if not docs:
            return "(변동된 파일 없음)"

        parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            filename = doc.get("filename") or "알 수 없는 파일"
            folder = doc.get("folder_path") or ""
            if folder:
                folder = folder.strip("/")
            created = str(doc.get("created_date", ""))[:10]
            content = (doc.get("content") or "").strip()
            if len(content) > 150:
                content = content[:150] + "..."

            line = f"{idx}. [{filename}]"
            if folder:
                line += f" ({folder})"
            if created:
                line += f" - {created}"
            if content:
                line += f"\n   {content}"
            parts.append(line)

        return "\n\n".join(parts)

    @staticmethod
    def _format_emails(docs: list[dict[str, Any]], label: str = "") -> str:
        """Format emails as '[subject] from sender - date'."""
        if not docs:
            return f"({label} 메일 없음)" if label else "(메일 없음)"

        parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            subject = doc.get("email_subject") or "제목 없음"
            sender = doc.get("email_from") or ""
            email_date = str(doc.get("email_date") or doc.get("created_date") or "")[:10]
            content = (doc.get("content") or "").strip()
            if len(content) > 150:
                content = content[:150] + "..."

            line = f"{idx}. [{subject}]"
            if sender:
                line += f" 발신: {sender}"
            if email_date:
                line += f" ({email_date})"
            if content:
                line += f"\n   {content}"
            parts.append(line)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Source citations
    # ------------------------------------------------------------------

    @staticmethod
    def _format_sources(data: dict[str, list[dict[str, Any]]]) -> str:
        """Build a compact source citation block from briefing data."""
        seen: set[str] = set()
        lines: list[str] = []

        for doc in data.get("recent_files", []):
            filename = doc.get("filename") or ""
            folder = (doc.get("folder_path") or "").strip("/")
            if not filename:
                continue
            name_key = f"{folder}/{filename}" if folder else filename
            date_str = _format_datetime(doc.get("created_date") or "")
            label = f"{name_key} ({date_str})" if date_str else name_key
            if name_key not in seen:
                seen.add(name_key)
                lines.append(f"- {label}")

        for doc in data.get("received_emails", []) + data.get("sent_emails", []):
            subject = doc.get("email_subject") or ""
            if not subject:
                continue
            date_str = _format_datetime(
                doc.get("email_date") or doc.get("created_date") or ""
            )
            label = f"{subject} ({date_str})" if date_str else subject
            if subject not in seen:
                seen.add(subject)
                lines.append(f"- {label}")

        if not lines:
            return ""

        return "[출처]\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate documents based on ``id`` or content prefix."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []

        for doc in docs:
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

        return unique

    # ------------------------------------------------------------------
    # Date range (calendar-based)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_date_range(briefing_type: str) -> tuple[datetime, datetime]:
        """Return ``(start, end)`` datetimes in KST for the given type.

        Uses calendar-based boundaries instead of simple timedelta offsets.
        """
        now = datetime.now(_KST)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if briefing_type == "daily":
            # Today 00:00 ~ now
            return today_start, now

        if briefing_type == "yesterday":
            # Yesterday 00:00 ~ today 00:00
            yesterday_start = today_start - timedelta(days=1)
            return yesterday_start, today_start

        if briefing_type == "weekly":
            # This week Monday 00:00 ~ now
            monday = today_start - timedelta(days=now.weekday())
            return monday, now

        if briefing_type == "last_week":
            # Last week Monday 00:00 ~ this week Monday 00:00
            this_monday = today_start - timedelta(days=now.weekday())
            last_monday = this_monday - timedelta(days=7)
            return last_monday, this_monday

        if briefing_type == "monthly":
            # This month 1st 00:00 ~ now
            month_start = today_start.replace(day=1)
            return month_start, now

        if briefing_type == "last_month":
            # Last month 1st ~ this month 1st
            this_month_start = today_start.replace(day=1)
            # Go to the last day of previous month, then to 1st
            prev_month_end = this_month_start - timedelta(days=1)
            last_month_start = prev_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return last_month_start, this_month_start

        # Fallback (should not reach here if validation is correct)
        return today_start, now

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
                    "content": content[:10000],
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
