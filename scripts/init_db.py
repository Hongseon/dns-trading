#!/usr/bin/env python3
"""Verify that all required Supabase tables exist.

Run this script after applying ``src/db/schema.sql`` to your Supabase
project to confirm that the tables are reachable.

Usage::

    python -m scripts.init_db
    # or
    python scripts/init_db.py
"""

from __future__ import annotations

import sys
import logging

# Ensure the project root is importable when executed directly.
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.db.supabase_client import get_client  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_REQUIRED_TABLES = ["documents", "sync_state", "briefings"]


def main() -> None:
    """Query each required table to verify it exists and is accessible."""
    client = get_client()
    errors: list[str] = []

    for table in _REQUIRED_TABLES:
        try:
            client.table(table).select("id").limit(1).execute()
            logger.info("Table '%s' -- OK", table)
        except Exception as exc:
            logger.error("Table '%s' -- FAILED: %s", table, exc)
            errors.append(table)

    if errors:
        logger.error(
            "Verification failed for table(s): %s. "
            "Please run src/db/schema.sql against your Supabase project.",
            ", ".join(errors),
        )
        sys.exit(1)

    print("\nAll tables verified successfully.")


if __name__ == "__main__":
    main()
