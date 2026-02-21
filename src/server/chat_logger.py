"""Chat logging and usage tracking for admin monitoring.

Persists every RAG query and briefing request to the ``chat_logs``
Zilliz collection, including token counts and estimated cost.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.db.zilliz_client import get_client

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))
_COLLECTION = "chat_logs"

# ------------------------------------------------------------------
# Model pricing (USD per 1 million tokens)
# ------------------------------------------------------------------

_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3-flash": {"input": 0.50, "output": 3.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
}

_DEFAULT_PRICING = {"input": 0.50, "output": 3.00}


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD."""
    pricing = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------


async def log_chat(
    *,
    query_type: str,
    user_query: str,
    response: str,
    usage: dict,
    response_time_ms: int,
) -> None:
    """Persist a chat interaction to the ``chat_logs`` collection.

    This function is designed to be called via ``asyncio.create_task()``
    so that logging never delays the user-facing response.  Any exception
    is caught and logged rather than propagated.
    """
    try:
        model = usage.get("model", "")
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        cost = _calculate_cost(model, input_tokens, output_tokens)

        now = datetime.now(_KST).isoformat()

        client = get_client()
        client.insert(
            collection_name=_COLLECTION,
            data=[{
                "query_type": query_type[:20],
                "user_query": user_query[:2000],
                "response": response[:60000],
                "model_used": model[:50],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost,
                "response_time_ms": response_time_ms,
                "created_at": now,
                "_dummy_vec": [0.0, 0.0],
            }],
        )
        logger.info(
            "Chat logged: type=%s model=%s tokens=%d cost=$%.5f",
            query_type, model, total_tokens, cost,
        )
    except Exception:
        logger.exception("Failed to log chat interaction")


# ------------------------------------------------------------------
# Admin query helpers
# ------------------------------------------------------------------


def get_recent_logs(limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """Return recent chat logs ordered by ``created_at`` descending.

    Returns ``(logs, total_count)``.
    """
    client = get_client()

    # Get total count
    total = client.query(
        collection_name=_COLLECTION,
        filter="",
        output_fields=["count(*)"],
    )
    total_count = total[0].get("count(*)", 0) if total else 0

    if total_count == 0:
        return [], 0

    results = client.query(
        collection_name=_COLLECTION,
        filter="",
        output_fields=[
            "query_type", "user_query", "response", "model_used",
            "input_tokens", "output_tokens", "total_tokens",
            "cost_usd", "response_time_ms", "created_at",
        ],
        limit=limit + offset,
    )

    # Sort by created_at descending and apply offset
    results.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    page = results[offset : offset + limit]

    return page, total_count


def get_usage_summary(period: str = "daily") -> dict:
    """Aggregate usage statistics for the given period.

    Parameters
    ----------
    period:
        One of ``"daily"``, ``"weekly"``, ``"monthly"``, ``"all"``.
    """
    now = datetime.now(_KST)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if period == "daily":
        start = today_start
        date_range = now.strftime("%Y-%m-%d")
    elif period == "weekly":
        start = today_start - timedelta(days=now.weekday())
        date_range = f"{start.strftime('%m/%d')}~{now.strftime('%m/%d')}"
    elif period == "monthly":
        start = today_start.replace(day=1)
        date_range = now.strftime("%Y-%m")
    else:  # "all"
        start = datetime(2020, 1, 1, tzinfo=_KST)
        date_range = "all"

    start_iso = start.isoformat()

    client = get_client()
    results = client.query(
        collection_name=_COLLECTION,
        filter=f'created_at >= "{start_iso}"',
        output_fields=[
            "query_type", "input_tokens", "output_tokens",
            "total_tokens", "cost_usd", "response_time_ms",
        ],
        limit=10000,
    )

    # Aggregate
    by_type: dict[str, int] = {}
    total_input = 0
    total_output = 0
    total_cost = 0.0
    total_time = 0

    for r in results:
        qtype = r.get("query_type", "unknown")
        by_type[qtype] = by_type.get(qtype, 0) + 1
        total_input += r.get("input_tokens", 0)
        total_output += r.get("output_tokens", 0)
        total_cost += r.get("cost_usd", 0.0)
        total_time += r.get("response_time_ms", 0)

    count = len(results)
    return {
        "period": period,
        "date_range": date_range,
        "total_requests": count,
        "by_type": by_type,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": round(total_cost, 5),
        "avg_response_time_ms": int(total_time / count) if count else 0,
    }
