"""Best-effort startup warmup helpers for latency-sensitive endpoints.

The KakaoTalk skill path is sensitive to the very first request after boot.
These helpers pre-create the shared RAG chain in the background once the app
starts so the first user request does not pay all lazy-initialisation costs.
"""

from __future__ import annotations

import asyncio
import logging

from src.config import settings

logger = logging.getLogger(__name__)


def get_chain():
    """Load the shared RAG chain lazily to keep server startup lightweight."""
    from src.rag.chain import get_chain as _get_chain

    return _get_chain()


def _has_required_settings() -> bool:
    """Return whether the RAG chain can be safely warmed."""
    return bool(
        settings.gemini_api_key
        and settings.zilliz_uri
        and settings.zilliz_token
    )


async def warm_rag_dependencies() -> bool:
    """Initialise the shared RAG chain in the background.

    Returns ``True`` when warmup completed successfully and ``False`` when it
    was skipped or failed. Failures are logged but never raised so application
    startup remains resilient.
    """
    if not _has_required_settings():
        logger.info(
            "Skipping RAG warmup; missing required env vars for Gemini or Zilliz"
        )
        return False

    try:
        await asyncio.to_thread(get_chain)
        logger.info("RAG warmup completed")
        return True
    except Exception:
        logger.exception("RAG warmup failed")
        return False


def start_rag_warmup() -> asyncio.Task[bool]:
    """Start background warmup for the shared RAG chain."""
    return asyncio.create_task(
        warm_rag_dependencies(),
        name="rag-startup-warmup",
    )


def ensure_rag_warmup_started(app_state) -> asyncio.Task[bool]:
    """Ensure a background warmup task exists without waiting for it."""
    task: asyncio.Task[bool] | None = getattr(app_state, "rag_warmup_task", None)
    if task is None or task.cancelled():
        task = start_rag_warmup()
        app_state.rag_warmup_task = task
    return task


async def ensure_rag_warmup(app_state) -> bool:
    """Ensure there is a single shared warmup task and await its result."""
    task: asyncio.Task[bool] | None = getattr(app_state, "rag_warmup_task", None)

    if task is None or task.cancelled():
        task = start_rag_warmup()
        app_state.rag_warmup_task = task
    elif task.done():
        try:
            if task.result():
                return True
        except Exception:
            logger.exception("Stored RAG warmup task failed")

        task = start_rag_warmup()
        app_state.rag_warmup_task = task

    try:
        return await task
    except Exception:
        logger.exception("RAG warmup task crashed")
        return False


def get_rag_warmup_status(app_state) -> str:
    """Return a human-readable status for the current warmup task."""
    task: asyncio.Task[bool] | None = getattr(app_state, "rag_warmup_task", None)
    if task is None:
        return "idle"
    if task.cancelled():
        return "cancelled"
    if not task.done():
        return "warming"

    try:
        return "ready" if task.result() else _failed_warmup_status()
    except Exception:
        return "failed"


def _failed_warmup_status() -> str:
    """Classify a failed warmup for operators."""
    return "skipped" if not _has_required_settings() else "failed"
