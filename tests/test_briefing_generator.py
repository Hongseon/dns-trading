"""Tests for briefing generation safeguards."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


class TestBriefingGeneratorTimeout:
    """Briefing generation should fail fast on slow LLM calls."""

    @patch("src.briefing.generator.chat_logger.log_chat", new_callable=AsyncMock)
    @patch("src.briefing.generator.get_client")
    @patch("src.briefing.generator.Generator")
    @patch("src.briefing.generator.Retriever")
    def test_generate_returns_timeout_message(
        self,
        mock_retriever_cls,
        mock_generator_cls,
        mock_get_client,
        mock_log_chat,
        monkeypatch,
    ):
        from src.briefing.generator import BriefingGenerator

        mock_retriever_cls.return_value = MagicMock()
        mock_get_client.return_value = MagicMock()

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(0.05)
            return ("should not complete", {"model": "test"})

        mock_generator = MagicMock()
        mock_generator._call_with_fallback = slow_call
        mock_generator_cls.return_value = mock_generator

        monkeypatch.setattr(
            "src.briefing.generator._BRIEFING_LLM_TIMEOUT_SECONDS",
            0.01,
        )

        generator = BriefingGenerator()
        sample_data = {
            "recent_files": [{"filename": "report.pdf", "content": "summary"}],
            "received_emails": [],
            "sent_emails": [],
        }

        def fake_create_task(coro):
            coro.close()
            return MagicMock()

        with patch("src.briefing.generator.asyncio.create_task", side_effect=fake_create_task):
            with patch.object(generator, "_collect_briefing_data", return_value=sample_data):
                with patch.object(generator, "_save_briefing") as mock_save:
                    result = asyncio.run(generator.generate("daily"))

        assert "시간이 초과" in result
        mock_save.assert_called_once()
