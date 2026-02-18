"""Briefing generator for daily, weekly, and monthly summaries.

Queries Zilliz Cloud for recent Dropbox file changes and email activity,
searches for schedule/task-related keywords, and uses the Gemini LLM to
produce a structured Korean briefing with separate sections for files,
emails, and upcoming tasks.
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

# DnS staff email addresses (used to distinguish sent vs received)
_DNS_STAFF_EMAILS: set[str] = {
    "theking57@naver.com",
    "ruthkim2015@naver.com",
}

_TASK_KEYWORDS: list[str] = [
    "일정",
    "마감",
    "deadline",
    "회의",
    "미팅",
    "예정",
    "납기",
    "납품",
    "검수",
    "계약",
    "입찰",
    "제출",
    "보고",
    "완료 예정",
    "진행 중",
    "pending",
]

_KST = timezone(timedelta(hours=9))

# ------------------------------------------------------------------
# Prompt templates (daily vs weekly/monthly)
# ------------------------------------------------------------------

_DAILY_PROMPT = """\
다음은 최근 업무 활동 데이터입니다.

== 최근 변동된 파일 ({file_count}건) ==
{files_section}

== 받은 메일 ({received_count}건) ==
{received_section}

== 보낸 메일 ({sent_count}건) ==
{sent_section}

== 업무/일정 관련 문서 ({task_count}건) ==
{tasks_section}

위 데이터를 분석하여 다음 형식으로 일간 업무 브리핑을 작성하세요:

📋 일간 업무 브리핑 ({date})

[파일 변동 사항]
• 새로 추가/수정된 파일과 주요 내용 요약

[받은 메일 요약]
• 외부에서 수신한 주요 메일의 핵심 내용

[보낸 메일 요약]
• DnS 직원이 발신한 주요 메일의 핵심 내용

[오늘의 할 일]
⚠️ 마감 임박 항목
• 예정된 업무 목록

[참고사항]
• 기타 중요 사항

규칙:
- 데이터가 없는 섹션은 "해당 없음"으로 표시
- 900자 이내로 작성
- 한국어로 작성"""

_WEEKLY_PROMPT = """\
다음은 지난 한 주간의 업무 활동 데이터입니다.

== 이번 주 변동된 파일 ({file_count}건) ==
{files_section}

== 이번 주 받은 메일 ({received_count}건) ==
{received_section}

== 이번 주 보낸 메일 ({sent_count}건) ==
{sent_section}

== 업무/일정 관련 문서 ({task_count}건) ==
{tasks_section}

위 데이터를 분석하여 다음 형식으로 주간 업무 브리핑을 작성하세요:

📋 주간 업무 브리핑 ({date})

[이번 주 주요 활동]
• 주요 파일 작업 및 메일 활동 요약 (3~5개)

[받은 메일 요약]
• 외부에서 수신한 주요 메일의 핵심 내용

[보낸 메일 요약]
• DnS 직원이 발신한 주요 메일의 핵심 내용

[프로젝트별 진행 상황]
• 프로젝트/계약 단위로 진행 상황 정리

[다음 주 예정 업무]
⚠️ 마감 임박 항목
• 예정된 업무 목록

[참고사항]
• 기타 중요 사항

규칙:
- 데이터가 없는 섹션은 "해당 없음"으로 표시
- 900자 이내로 작성
- 한국어로 작성"""

_MONTHLY_PROMPT = """\
다음은 지난 한 달간의 업무 활동 데이터입니다.

== 이번 달 변동된 파일 ({file_count}건) ==
{files_section}

== 이번 달 받은 메일 ({received_count}건) ==
{received_section}

== 이번 달 보낸 메일 ({sent_count}건) ==
{sent_section}

== 업무/일정 관련 문서 ({task_count}건) ==
{tasks_section}

위 데이터를 분석하여 다음 형식으로 월간 업무 브리핑을 작성하세요:

📋 월간 업무 브리핑 ({date})

[이번 달 주요 성과]
• 완료된 주요 업무 (3~5개)

[받은 메일 요약]
• 외부에서 수신한 주요 메일의 핵심 내용

[보낸 메일 요약]
• DnS 직원이 발신한 주요 메일의 핵심 내용

[프로젝트별 진행 현황]
• 프로젝트/계약 단위 현황 정리

[다음 달 주요 일정]
⚠️ 마감 임박 항목
• 예정된 업무 및 마감 일정

[참고사항]
• 기타 중요 사항

규칙:
- 데이터가 없는 섹션은 "해당 없음"으로 표시
- 900자 이내로 작성
- 한국어로 작성"""

_PROMPTS: dict[str, str] = {
    "daily": _DAILY_PROMPT,
    "weekly": _WEEKLY_PROMPT,
    "monthly": _MONTHLY_PROMPT,
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
            One of ``"daily"``, ``"weekly"``, or ``"monthly"``.

        Returns
        -------
        str
            The generated briefing text.
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

        # Collect data in categories
        data = self._collect_briefing_data(briefing_type, start, end)

        has_data = (
            data["recent_files"]
            or data["received_emails"]
            or data["sent_emails"]
            or data["upcoming_tasks"]
        )
        if not has_data:
            msg = "해당 기간에 새로운 문서/메일이 없습니다."
            logger.info(msg)
            self._save_briefing(briefing_type, msg)
            return msg

        # Build the LLM prompt
        now_kst = datetime.now(_KST)
        prompt = self._build_prompt(briefing_type, data, now_kst)

        # Call the LLM with briefing-specific settings
        briefing_system = (
            "당신은 업무 브리핑을 작성하는 AI 어시스턴트입니다. "
            "제공된 파일 변동 사항과 이메일 데이터를 분석하여 "
            "구조화된 한국어 업무 브리핑을 작성하세요. "
            "모든 섹션을 빠짐없이 작성하고, 900자 이내로 완성하세요."
        )
        try:
            content = await self.generator._call_with_fallback(
                prompt,
                system_instruction=briefing_system,
                max_output_tokens=2048,
            )
        except Exception:
            logger.exception("Failed to generate briefing via LLM")
            content = "브리핑 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

        # Persist
        self._save_briefing(briefing_type, content)

        logger.info(
            "%s briefing generated (%d chars)", briefing_type, len(content)
        )
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
        """Collect documents, emails, and task-related items for the briefing."""
        start_iso = start.isoformat()
        end_iso = end.isoformat()

        file_limit = 15 if briefing_type == "daily" else 30
        email_limit = 30 if briefing_type == "daily" else 60

        # 1. Recently changed Dropbox files (by updated_date = indexing time)
        recent_files = self.retriever.search_by_date_range(
            date_field="updated_date",
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

        # 3. Upcoming tasks / schedule-related (vector search)
        upcoming_tasks = self._search_upcoming_tasks(start_iso)

        return {
            "recent_files": recent_files,
            "received_emails": received_emails,
            "sent_emails": sent_emails,
            "upcoming_tasks": upcoming_tasks,
        }

    def _search_upcoming_tasks(
        self,
        after_date: str,
    ) -> list[dict[str, Any]]:
        """Search for schedule/task-related documents via vector similarity.

        Uses an expanded set of keywords and deduplicates results.
        """
        all_results: list[dict[str, Any]] = []

        for keyword in _TASK_KEYWORDS:
            try:
                results = self.retriever.search(
                    query=keyword,
                    after_date=after_date,
                    top_k=3,
                )
                all_results.extend(results)
            except Exception:
                logger.warning(
                    "Task keyword search failed for '%s'",
                    keyword,
                    exc_info=True,
                )

        # Deduplicate
        unique = self._deduplicate(all_results)

        logger.info(
            "Task keyword search: %d raw -> %d unique results",
            len(all_results),
            len(unique),
        )
        return unique[:15]

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        briefing_type: str,
        data: dict[str, list[dict[str, Any]]],
        now_kst: datetime,
    ) -> str:
        """Build the LLM prompt with separated file/email/task sections."""
        files_section = self._format_files(data["recent_files"])
        received_section = self._format_emails(data["received_emails"], label="받은")
        sent_section = self._format_emails(data["sent_emails"], label="보낸")
        tasks_section = self._format_tasks(data["upcoming_tasks"])

        template = _PROMPTS[briefing_type]
        return template.format(
            file_count=len(data["recent_files"]),
            files_section=files_section,
            received_count=len(data["received_emails"]),
            received_section=received_section,
            sent_count=len(data["sent_emails"]),
            sent_section=sent_section,
            task_count=len(data["upcoming_tasks"]),
            tasks_section=tasks_section,
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

    @staticmethod
    def _format_tasks(docs: list[dict[str, Any]]) -> str:
        """Format task/schedule-related documents with content excerpts."""
        if not docs:
            return "(관련 문서 없음)"

        parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            source_type = doc.get("source_type", "")
            content = (doc.get("content") or "").strip()
            if len(content) > 200:
                content = content[:200] + "..."

            if source_type == "dropbox":
                label = doc.get("filename") or "파일"
                source_label = f"[파일: {label}]"
            elif source_type == "email":
                subject = doc.get("email_subject") or "제목 없음"
                sender = doc.get("email_from") or ""
                source_label = f"[이메일: {subject} - {sender}]"
            else:
                source_label = f"[{source_type}]"

            created = str(doc.get("created_date", ""))[:10]
            date_part = f" ({created})" if created else ""

            parts.append(f"{idx}. {source_label}{date_part}\n   {content}")

        return "\n\n".join(parts)

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
    # Date range
    # ------------------------------------------------------------------

    @staticmethod
    def _get_date_range(briefing_type: str) -> tuple[datetime, datetime]:
        """Return ``(start, end)`` datetimes in KST for the given type."""
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
