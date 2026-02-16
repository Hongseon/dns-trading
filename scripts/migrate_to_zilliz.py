#!/usr/bin/env python3
"""Migrate data from Supabase pgvector to Zilliz Cloud.

Reads all documents, sync_state, and briefings from Supabase and
inserts them into the corresponding Zilliz collections.

Usage::

    python scripts/migrate_to_zilliz.py

Requires both SUPABASE_* and ZILLIZ_* env vars to be set.
The supabase package must be pip-installed (not in requirements.txt).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Page size for Supabase queries
_PAGE_SIZE = 1000
# Batch size for Zilliz inserts
_BATCH_SIZE = 1000


def get_supabase_client():
    """Create a Supabase client using env vars."""
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)


def migrate_documents(sb_client, zilliz_client) -> int:
    """Migrate all documents from Supabase to Zilliz."""
    logger.info("Starting documents migration...")

    total = 0
    offset = 0

    while True:
        # Fetch a page from Supabase
        response = (
            sb_client.table("documents")
            .select(
                "source_type, source_id, content, embedding, chunk_index, "
                "created_date, updated_date, filename, folder_path, file_type, "
                "email_from, email_to, email_subject, email_date"
            )
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )

        rows = response.data
        if not rows:
            break

        # Transform rows for Zilliz
        zilliz_rows = []
        skipped = 0
        for row in rows:
            embedding = row.get("embedding")

            # Supabase returns pgvector as a string like "[0.1,0.2,...]"
            if isinstance(embedding, str):
                try:
                    import json
                    embedding = json.loads(embedding)
                except (json.JSONDecodeError, ValueError):
                    pass

            if not embedding or not isinstance(embedding, list) or len(embedding) != 768:
                skipped += 1
                continue

            zilliz_row = {
                "source_type": row.get("source_type") or "",
                "source_id": row.get("source_id") or "",
                "content": (row.get("content") or "")[:10000],
                "embedding": embedding,
                "chunk_index": row.get("chunk_index", 0),
                "created_date": row.get("created_date") or "",
                "updated_date": row.get("updated_date") or "",
                "filename": row.get("filename") or "",
                "folder_path": row.get("folder_path") or "",
                "file_type": row.get("file_type") or "",
                "email_from": row.get("email_from") or "",
                "email_to": row.get("email_to") or "",
                "email_subject": row.get("email_subject") or "",
                "email_date": row.get("email_date") or "",
            }
            zilliz_rows.append(zilliz_row)

        # Batch insert to Zilliz
        if zilliz_rows:
            for start in range(0, len(zilliz_rows), _BATCH_SIZE):
                batch = zilliz_rows[start : start + _BATCH_SIZE]
                try:
                    zilliz_client.insert(
                        collection_name="documents",
                        data=batch,
                    )
                except Exception:
                    logger.exception(
                        "Failed to insert batch at offset %d-%d",
                        offset + start,
                        offset + start + len(batch),
                    )

        total += len(zilliz_rows)
        logger.info(
            "Migrated %d documents so far (fetched %d this page)",
            total,
            len(rows),
        )

        if len(rows) < _PAGE_SIZE:
            break

        offset += _PAGE_SIZE
        time.sleep(0.5)  # Be gentle on Supabase

    return total


def migrate_sync_state(sb_client, zilliz_client) -> int:
    """Migrate sync_state rows from Supabase to Zilliz."""
    logger.info("Starting sync_state migration...")

    response = sb_client.table("sync_state").select("*").execute()
    rows = response.data or []

    for row in rows:
        zilliz_row = {
            "sync_type": row.get("sync_type", ""),
            "last_cursor": row.get("last_cursor") or "",
            "last_sync_time": row.get("last_sync_time") or "",
            "updated_at": row.get("updated_at") or "",
            "_dummy_vec": [0.0, 0.0],
        }
        try:
            zilliz_client.upsert(
                collection_name="sync_state",
                data=[zilliz_row],
            )
        except Exception:
            logger.exception(
                "Failed to migrate sync_state row: %s", row.get("sync_type")
            )

    logger.info("Migrated %d sync_state rows", len(rows))
    return len(rows)


def migrate_briefings(sb_client, zilliz_client) -> int:
    """Migrate briefings rows from Supabase to Zilliz."""
    logger.info("Starting briefings migration...")

    response = (
        sb_client.table("briefings")
        .select("briefing_type, content, generated_at, sent")
        .execute()
    )
    rows = response.data or []

    for row in rows:
        zilliz_row = {
            "briefing_type": row.get("briefing_type", ""),
            "content": (row.get("content") or "")[:10000],
            "generated_at": row.get("generated_at") or "",
            "sent": row.get("sent", False),
            "_dummy_vec": [0.0, 0.0],
        }
        try:
            zilliz_client.insert(
                collection_name="briefings",
                data=[zilliz_row],
            )
        except Exception:
            logger.exception("Failed to migrate briefing row")

    logger.info("Migrated %d briefings rows", len(rows))
    return len(rows)


def verify_counts(sb_client, zilliz_client) -> None:
    """Compare row counts between Supabase and Zilliz."""
    logger.info("Verifying row counts...")

    # Supabase counts
    sb_docs = sb_client.table("documents").select("id", count="exact").execute()
    sb_doc_count = sb_docs.count or 0

    sb_sync = sb_client.table("sync_state").select("id", count="exact").execute()
    sb_sync_count = len(sb_sync.data) if sb_sync.data else 0

    sb_brief = sb_client.table("briefings").select("id", count="exact").execute()
    sb_brief_count = sb_brief.count or 0

    # Zilliz counts
    z_doc_stats = zilliz_client.get_collection_stats("documents")
    z_doc_count = z_doc_stats.get("row_count", 0)

    z_sync_stats = zilliz_client.get_collection_stats("sync_state")
    z_sync_count = z_sync_stats.get("row_count", 0)

    z_brief_stats = zilliz_client.get_collection_stats("briefings")
    z_brief_count = z_brief_stats.get("row_count", 0)

    print("\n=== Migration Verification ===")
    print(f"documents:  Supabase={sb_doc_count:>8}  Zilliz={z_doc_count:>8}  {'OK' if sb_doc_count == z_doc_count else 'MISMATCH'}")
    print(f"sync_state: Supabase={sb_sync_count:>8}  Zilliz={z_sync_count:>8}  {'OK' if sb_sync_count == z_sync_count else 'MISMATCH'}")
    print(f"briefings:  Supabase={sb_brief_count:>8}  Zilliz={z_brief_count:>8}  {'OK' if sb_brief_count == z_brief_count else 'MISMATCH'}")


def main() -> None:
    from src.db.zilliz_client import get_client, init_collections

    # Setup Zilliz
    zilliz_client = get_client()
    init_collections()

    # Setup Supabase
    sb_client = get_supabase_client()

    # Migrate
    doc_count = migrate_documents(sb_client, zilliz_client)
    sync_count = migrate_sync_state(sb_client, zilliz_client)
    brief_count = migrate_briefings(sb_client, zilliz_client)

    print(f"\n=== Migration Complete ===")
    print(f"Documents: {doc_count}")
    print(f"Sync state: {sync_count}")
    print(f"Briefings: {brief_count}")

    # Wait for Zilliz to update stats
    time.sleep(3)

    # Verify
    verify_counts(sb_client, zilliz_client)


if __name__ == "__main__":
    main()
