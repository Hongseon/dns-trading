"""Best-effort startup warmup helpers for latency-sensitive endpoints.

The KakaoTalk skill path is sensitive to the very first request after boot.
These helpers pre-create the shared RAG chain in the background once the app
starts so the first user request does not pay all lazy-initialisation costs.
"""

from __future__ import annotations

import asyncio
import logging

from src.config import settings
from src.rag.chain import get_chain

logger = logging.getLogger(__name__)


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
