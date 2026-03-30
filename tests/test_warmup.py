"""Tests for startup warmup helpers in src.server.warmup."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from src.server import warmup


class TestWarmRagDependencies:
    """Warmup should be safe when config is incomplete or dependencies fail."""

    def test_skips_when_required_settings_are_missing(self, monkeypatch):
        monkeypatch.setattr(warmup.settings, "gemini_api_key", "")
        monkeypatch.setattr(warmup.settings, "zilliz_uri", "")
        monkeypatch.setattr(warmup.settings, "zilliz_token", "")

        with patch("src.server.warmup.get_chain") as mock_get_chain:
            result = asyncio.run(warmup.warm_rag_dependencies())

        assert result is False
        mock_get_chain.assert_not_called()

    def test_warms_chain_when_required_settings_exist(self, monkeypatch):
        monkeypatch.setattr(warmup.settings, "gemini_api_key", "test-key")
        monkeypatch.setattr(warmup.settings, "zilliz_uri", "https://zilliz.example")
        monkeypatch.setattr(warmup.settings, "zilliz_token", "test-token")

        with patch("src.server.warmup.get_chain") as mock_get_chain:
            result = asyncio.run(warmup.warm_rag_dependencies())

        assert result is True
        mock_get_chain.assert_called_once_with()

    def test_returns_false_when_chain_initialisation_fails(self, monkeypatch):
        monkeypatch.setattr(warmup.settings, "gemini_api_key", "test-key")
        monkeypatch.setattr(warmup.settings, "zilliz_uri", "https://zilliz.example")
        monkeypatch.setattr(warmup.settings, "zilliz_token", "test-token")

        with patch("src.server.warmup.get_chain", side_effect=RuntimeError("boom")):
            result = asyncio.run(warmup.warm_rag_dependencies())

        assert result is False


class TestWarmupTaskHelpers:
    """Warmup task helpers should deduplicate work and expose state."""

    def test_ensure_rag_warmup_reuses_running_task(self):
        class AppState:
            pass

        app_state = AppState()

        async def runner():
            task = asyncio.create_task(asyncio.sleep(0.01, result=True))
            app_state.rag_warmup_task = task
            result = await warmup.ensure_rag_warmup(app_state)
            assert result is True
            assert app_state.rag_warmup_task is task

        asyncio.run(runner())

    def test_get_rag_warmup_status_reports_ready(self):
        class DoneTask:
            def cancelled(self) -> bool:
                return False

            def done(self) -> bool:
                return True

            def result(self) -> bool:
                return True

        class AppState:
            rag_warmup_task = DoneTask()

        assert warmup.get_rag_warmup_status(AppState()) == "ready"
