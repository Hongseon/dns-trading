"""Run full data sync (Dropbox + Naver Mail).

Orchestrates both data-source syncers sequentially and reports results.
Designed to be called from the command line or from a GitHub Actions
workflow.

Usage::

    python scripts/full_sync.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so that ``src`` is importable
# regardless of the working directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ingestion.dropbox_sync import DropboxSync  # noqa: E402
from src.ingestion.naver_mail_sync import NaverMailSync  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run Dropbox and Naver Mail syncs, then log a combined summary."""
    overall_start = time.monotonic()
    has_error = False

    # ---- Dropbox sync ----
    logger.info("=" * 60)
    logger.info("Starting Dropbox sync")
    logger.info("=" * 60)

    dropbox_result: dict[str, int] = {
        "added": 0,
        "deleted": 0,
        "skipped": 0,
        "errors": 0,
    }
    try:
        t0 = time.monotonic()
        dropbox_syncer = DropboxSync()
        dropbox_result = dropbox_syncer.sync()
        elapsed = time.monotonic() - t0
        logger.info("Dropbox sync finished in %.1f s: %s", elapsed, dropbox_result)
    except Exception:
        logger.exception("Dropbox sync raised an unhandled exception")
        dropbox_result["errors"] += 1
        has_error = True

    # ---- Naver Mail sync ----
    logger.info("=" * 60)
    logger.info("Starting Naver Mail sync")
    logger.info("=" * 60)

    mail_result: dict[str, int] = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
    }
    try:
        t0 = time.monotonic()
        mail_syncer = NaverMailSync()
        mail_result = mail_syncer.sync()
        elapsed = time.monotonic() - t0
        logger.info("Naver Mail sync finished in %.1f s: %s", elapsed, mail_result)
    except Exception:
        logger.exception("Naver Mail sync raised an unhandled exception")
        mail_result["errors"] += 1
        has_error = True

    # ---- Summary ----
    total_elapsed = time.monotonic() - overall_start
    total_errors = dropbox_result.get("errors", 0) + mail_result.get("errors", 0)

    logger.info("=" * 60)
    logger.info("Full sync complete in %.1f s", total_elapsed)
    logger.info("  Dropbox : %s", dropbox_result)
    logger.info("  Mail    : %s", mail_result)
    logger.info("  Total errors: %d", total_errors)
    logger.info("=" * 60)

    if has_error or total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
