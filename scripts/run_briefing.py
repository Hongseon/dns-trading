#!/usr/bin/env python3
"""Generate a business briefing manually.

Usage::

    python scripts/run_briefing.py              # daily (default)
    python scripts/run_briefing.py daily
    python scripts/run_briefing.py weekly
    python scripts/run_briefing.py monthly
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure the project root is importable when executed directly.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.briefing.generator import BriefingGenerator  # noqa: E402
from src.briefing.sender import BriefingSender  # noqa: E402

logger = logging.getLogger(__name__)

_VALID_TYPES = ("daily", "yesterday", "weekly", "last_week", "monthly", "last_month")


async def main() -> None:
    """Parse arguments, generate the briefing, and send it."""
    briefing_type = sys.argv[1] if len(sys.argv) > 1 else "daily"

    if briefing_type not in _VALID_TYPES:
        print(
            f"Invalid type: {briefing_type}. "
            f"Use one of: {', '.join(_VALID_TYPES)}."
        )
        sys.exit(1)

    logger.info("Starting %s briefing generation...", briefing_type)

    generator = BriefingGenerator()
    content = await generator.generate(briefing_type)

    print("=" * 60)
    print(content)
    print("=" * 60)

    sender = BriefingSender()
    success = await sender.send_to_channel(content)

    if success:
        logger.info("Briefing delivered successfully.")
    else:
        logger.error("Briefing delivery failed.")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
