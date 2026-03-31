"""Tests for FastAPI skill endpoints (src.server.main + src.server.skill_handler).

Uses FastAPI TestClient with mocked RAG chain and briefing modules so no
external services (Supabase, Gemini, Dropbox, KakaoTalk) are needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.server.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_skill_request(utterance: str, callback_url: str | None = None) -> dict:
    """Build a minimal KakaoTalk OpenBuilder skill request body."""
    body = {
        "intent": {"name": "test"},
        "userRequest": {"utterance": utterance},
        "bot": {"id": "test"},
        "action": {"params": {}},
    }
    if callback_url:
        body["userRequest"]["callbackUrl"] = callback_url
    return body


# ---------------------------------------------------------------------------
# Tests -- Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    """GET /health should return 200 and {"status": "ok"}."""

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data == {"status": "ok"}


# ---------------------------------------------------------------------------
# Tests -- /skill/query
# ---------------------------------------------------------------------------


class TestQueryWithCallback:
    """POST /skill/query with a callbackUrl should return useCallback=True."""

    @patch("src.server.skill_handler.process_and_callback", new_callable=AsyncMock)
    @patch("src.server.skill_handler.get_chain")
    def test_query_with_callback(self, mock_get_chain, mock_callback):
        body = make_skill_request(
            utterance="A사 계약서 납품 기한 알려줘",
            callback_url="https://bot-api.kakao.com/callback/test123",
        )
        response = client.post("/skill/query", json=body)

        assert response.status_code == 200
        data = response.json()

        # Must include useCallback: true
        assert data.get("useCallback") is True
        assert data["version"] == "2.0"

        # The immediate placeholder text should be present
        text = data["template"]["outputs"][0]["simpleText"]["text"]
        assert len(text) > 0


class TestQueryWithoutCallback:
    """POST /skill/query without callbackUrl should run quick_run and return the answer."""

    @patch("src.server.skill_handler.get_chain")
    def test_query_without_callback(self, mock_get_chain):
        # Set up the mock chain
        mock_chain = MagicMock()
        mock_chain.quick_run = AsyncMock(return_value="납품 기한은 2025년 3월 15일입니다.")
        mock_get_chain.return_value = mock_chain

        body = make_skill_request(utterance="납품 기한 알려줘")
        response = client.post("/skill/query", json=body)

        assert response.status_code == 200
        data = response.json()

        # Should NOT include useCallback
        assert "useCallback" not in data

        text = data["template"]["outputs"][0]["simpleText"]["text"]
        assert "납품 기한은 2025년 3월 15일입니다." in text


class TestQueryTruncatesLongAnswer:
    """Answers longer than 1000 chars should be truncated."""

    @patch("src.server.skill_handler.get_chain")
    def test_query_truncates_long_answer(self, mock_get_chain):
        # Generate a 1500-char answer
        long_answer = "A" * 1500
        mock_chain = MagicMock()
        mock_chain.quick_run = AsyncMock(return_value=long_answer)
        mock_get_chain.return_value = mock_chain

        body = make_skill_request(utterance="긴 답변 테스트")
        response = client.post("/skill/query", json=body)

        assert response.status_code == 200
        data = response.json()

        text = data["template"]["outputs"][0]["simpleText"]["text"]
        assert len(text) <= 1000


class TestQueryEmptyUtterance:
    """POST /skill/query with an empty utterance should return a prompt message."""

    def test_query_empty_utterance(self):
        body = make_skill_request(utterance="")
        response = client.post("/skill/query", json=body)

        assert response.status_code == 200
        data = response.json()
        text = data["template"]["outputs"][0]["simpleText"]["text"]
        assert "질문을 입력해 주세요" in text


# ---------------------------------------------------------------------------
# Tests -- /skill/briefing
# ---------------------------------------------------------------------------


class TestBriefingDetectsDaily:
    """Utterance '오늘 브리핑' should be classified as briefing_type 'daily'."""

    @patch("src.server.skill_handler.process_briefing_and_callback", new_callable=AsyncMock)
    def test_briefing_detects_daily(self, mock_briefing_callback):
        body = make_skill_request(
            utterance="오늘 브리핑",
            callback_url="https://bot-api.kakao.com/callback/test_daily",
        )
        response = client.post("/skill/briefing", json=body)

        assert response.status_code == 200

        # Verify the briefing callback was called with 'daily'
        mock_briefing_callback.assert_called_once()
        call_args = mock_briefing_callback.call_args
        assert call_args[0][0] == "daily"  # first positional arg = briefing_type


class TestBriefingDetectsWeekly:
    """Utterance '주간 브리핑' should be classified as briefing_type 'weekly'."""

    @patch("src.server.skill_handler.process_briefing_and_callback", new_callable=AsyncMock)
    def test_briefing_detects_weekly(self, mock_briefing_callback):
        body = make_skill_request(
            utterance="주간 브리핑",
            callback_url="https://bot-api.kakao.com/callback/test_weekly",
        )
        response = client.post("/skill/briefing", json=body)

        assert response.status_code == 200

        mock_briefing_callback.assert_called_once()
        call_args = mock_briefing_callback.call_args
        assert call_args[0][0] == "weekly"


class TestBriefingDetectsMonthly:
    """Utterance '월간 브리핑' should be classified as briefing_type 'monthly'."""

    @patch("src.server.skill_handler.process_briefing_and_callback", new_callable=AsyncMock)
    def test_briefing_detects_monthly(self, mock_briefing_callback):
        body = make_skill_request(
            utterance="이번 달 업무 브리핑 보여줘",
            callback_url="https://bot-api.kakao.com/callback/test_monthly",
        )
        response = client.post("/skill/briefing", json=body)

        assert response.status_code == 200

        mock_briefing_callback.assert_called_once()
        call_args = mock_briefing_callback.call_args
        assert call_args[0][0] == "monthly"


class TestBriefingWithoutCallback:
    """Briefing without callbackUrl should return a notice (too slow for 5s)."""

    def test_briefing_without_callback(self):
        body = make_skill_request(utterance="오늘 브리핑")
        response = client.post("/skill/briefing", json=body)

        assert response.status_code == 200
        data = response.json()
        text = data["template"]["outputs"][0]["simpleText"]["text"]
        # Should mention that callback is not active
        assert "콜백" in text or "시간" in text
