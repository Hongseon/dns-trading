"""Admin API endpoints for chat log and usage monitoring.

Provides read-only access to conversation history and LLM cost tracking.
Protected by a simple API key (``ADMIN_API_KEY`` env var).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from src.config import settings
from src.server.chat_logger import get_recent_logs, get_usage_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ------------------------------------------------------------------
# Auth helper
# ------------------------------------------------------------------


def _check_key(key: str) -> None:
    """Raise 401 if the key does not match the configured admin key."""
    if not settings.admin_api_key:
        raise HTTPException(503, "ADMIN_API_KEY is not configured on the server.")
    if key != settings.admin_api_key:
        raise HTTPException(401, "Invalid admin API key.")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.get("/logs")
async def admin_logs(
    key: str = Query("", description="Admin API key"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return recent chat logs ordered by newest first."""
    _check_key(key)

    logs, total = get_recent_logs(limit=limit, offset=offset)
    return {
        "logs": logs,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/usage")
async def admin_usage(
    key: str = Query("", description="Admin API key"),
    period: str = Query("daily", description="daily | weekly | monthly | all"),
):
    """Return aggregated usage/cost summary for the given period."""
    _check_key(key)

    if period not in ("daily", "weekly", "monthly", "all"):
        raise HTTPException(400, "period must be one of: daily, weekly, monthly, all")

    summary = get_usage_summary(period=period)
    return summary
