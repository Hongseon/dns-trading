#!/usr/bin/env python3
"""Verify that all required Zilliz Cloud collections exist.

Creates collections if they don't exist, then verifies they are reachable.

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

from src.db.zilliz_client import get_client, init_collections  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_REQUIRED_COLLECTIONS = ["documents", "sync_state", "briefings"]


def main() -> None:
    """Create collections if needed, then verify they exist."""
    # Initialize collections (creates if not exists)
    init_collections()

    # Verify each collection is accessible
    client = get_client()
    errors: list[str] = []

    for collection in _REQUIRED_COLLECTIONS:
        try:
            if client.has_collection(collection):
                stats = client.get_collection_stats(collection)
                row_count = stats.get("row_count", 0)
                logger.info(
                    "Collection '%s' -- OK (%s rows)", collection, row_count
                )
            else:
                logger.error("Collection '%s' -- NOT FOUND", collection)
                errors.append(collection)
        except Exception as exc:
            logger.error("Collection '%s' -- FAILED: %s", collection, exc)
            errors.append(collection)

    if errors:
        logger.error(
            "Verification failed for collection(s): %s.",
            ", ".join(errors),
        )
        sys.exit(1)

    print("\nAll collections verified successfully.")


if __name__ == "__main__":
    main()
