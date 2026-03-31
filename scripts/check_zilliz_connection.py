"""Fail-fast diagnostic probe for Zilliz/Milvus connectivity.

This script is intended for CI preflight checks. It logs which stage is
currently running so workflow logs can distinguish between a hang during
client creation and a hang during the first Milvus API call.
"""

from __future__ import annotations

import logging
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# Ensure the project root is importable when this script is run directly.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pymilvus import MilvusClient  # noqa: E402

from src.config import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _extract_host(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.hostname:
        return parsed.hostname
    return uri.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]


def main() -> None:
    """Probe DNS, client construction, and one cheap Milvus API call."""
    if not settings.zilliz_uri or not settings.zilliz_token:
        raise ValueError("ZILLIZ_URI and ZILLIZ_TOKEN must be set for diagnostics.")

    host = _extract_host(settings.zilliz_uri)
    logger.info("Starting Zilliz connectivity probe for host=%s", host)

    t0 = time.monotonic()
    addresses = sorted({result[4][0] for result in socket.getaddrinfo(host, 443)})
    logger.info(
        "DNS resolved in %.2f s: %s",
        time.monotonic() - t0,
        ", ".join(addresses),
    )

    t0 = time.monotonic()
    logger.info("Constructing MilvusClient")
    client = MilvusClient(uri=settings.zilliz_uri, token=settings.zilliz_token)
    logger.info("MilvusClient constructed in %.2f s", time.monotonic() - t0)

    t0 = time.monotonic()
    logger.info("Running has_collection('documents')")
    exists = client.has_collection("documents")
    logger.info(
        "has_collection('documents') returned %s in %.2f s",
        exists,
        time.monotonic() - t0,
    )


if __name__ == "__main__":
    main()
