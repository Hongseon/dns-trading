"""Singleton Supabase client wrapper.

Provides a lazily-initialized, module-level Supabase client so that
every module in the project shares a single connection instance.
"""

from __future__ import annotations

import logging

from supabase import Client, create_client

from src.config import settings

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_client() -> Client:
    """Return the shared Supabase client, creating it on first call.

    The client is cached at module level so subsequent calls are free.

    Raises:
        ValueError: If ``SUPABASE_URL`` or ``SUPABASE_SERVICE_KEY`` are not
            configured in the environment / ``.env`` file.
    """
    global _client

    if _client is not None:
        return _client

    if not settings.supabase_url or not settings.supabase_service_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set. "
            "Check your .env file or environment variables."
        )

    logger.info("Initializing Supabase client for %s", settings.supabase_url)
    _client = create_client(settings.supabase_url, settings.supabase_service_key)
    return _client
