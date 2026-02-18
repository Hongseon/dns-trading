"""Re-process scanned image PDFs that were skipped during initial Dropbox sync.

Lists all PDF files in Dropbox, checks which ones are missing from Zilliz
(i.e. were skipped because text extraction returned empty), downloads them,
runs OCR via PaddleOCR, and indexes the extracted text.

Usage::

    # Dry run — list skipped PDFs without processing
    python scripts/ocr_reprocess.py --dry-run

    # Process all skipped PDFs
    python scripts/ocr_reprocess.py

    # Limit to N files (useful for testing)
    python scripts/ocr_reprocess.py --limit 10

    # Resume from a specific file index (skip first N)
    python scripts/ocr_reprocess.py --offset 100
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dropbox import Dropbox
from dropbox.files import FileMetadata, FolderMetadata

from src.config import settings
from src.db.zilliz_client import get_client
from src.ingestion.chunker import TextChunker
from src.ingestion.indexer import DocumentMetadata, Indexer
from src.ingestion.text_extractor import extract_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_dropbox_client() -> Dropbox:
    if settings.dropbox_refresh_token:
        return Dropbox(
            app_key=settings.dropbox_app_key,
            app_secret=settings.dropbox_app_secret,
            oauth2_refresh_token=settings.dropbox_refresh_token,
        )
    return Dropbox(settings.dropbox_access_token)


def _list_all_pdfs(dbx: Dropbox, folder_path: str) -> list[FileMetadata]:
    """List all PDF files in Dropbox recursively."""
    pdfs: list[FileMetadata] = []
    max_size = settings.max_file_size_mb * 1024 * 1024

    result = dbx.files_list_folder(folder_path, recursive=True)
    while True:
        for entry in result.entries:
            if isinstance(entry, FolderMetadata):
                continue
            if not isinstance(entry, FileMetadata):
                continue
            if not entry.name.lower().endswith(".pdf"):
                continue
            if entry.size > max_size:
                continue
            pdfs.append(entry)

        if not result.has_more:
            break
        result = dbx.files_list_folder_continue(result.cursor)

    logger.info("Found %d PDF files in Dropbox", len(pdfs))
    return pdfs


def _get_indexed_source_ids(client) -> set[str]:
    """Query Zilliz for all indexed Dropbox PDF source_ids (standalone only)."""
    indexed: set[str] = set()
    try:
        results = client.query(
            collection_name="documents",
            filter='source_type == "dropbox" and file_type == "pdf" and chunk_index == 0',
            output_fields=["source_id"],
            limit=16384,
        )
        for row in results:
            sid = row.get("source_id")
            if sid:
                indexed.add(sid)
    except Exception:
        logger.exception("Failed to query indexed PDF source_ids")
    logger.info("Found %d already-indexed PDF source_ids in Zilliz", len(indexed))
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-process skipped scanned PDFs with OCR")
    parser.add_argument("--dry-run", action="store_true", help="List skipped PDFs without processing")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N files")
    args = parser.parse_args()

    dbx = _get_dropbox_client()
    client = get_client()
    chunker = TextChunker()
    indexer = Indexer()

    # 1. List all PDFs in Dropbox
    all_pdfs = _list_all_pdfs(dbx, settings.dropbox_folder_path)

    # 2. Find which are already indexed
    indexed_ids = _get_indexed_source_ids(client)

    # 3. Filter to unindexed (skipped) PDFs
    skipped = [pdf for pdf in all_pdfs if pdf.id not in indexed_ids]
    skipped.sort(key=lambda e: e.path_display or "")

    logger.info(
        "Skipped (unindexed) PDFs: %d / %d total",
        len(skipped), len(all_pdfs),
    )

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — {len(skipped)} skipped PDFs found:")
        print(f"{'='*60}")
        for i, pdf in enumerate(skipped):
            print(f"  [{i+1:4d}] {pdf.path_display} ({pdf.size:,} bytes)")
        print(f"{'='*60}")
        return

    # Apply offset/limit
    target = skipped[args.offset:]
    if args.limit > 0:
        target = target[:args.limit]

    logger.info(
        "Processing %d files (offset=%d, limit=%s)",
        len(target), args.offset, args.limit or "all",
    )

    stats = {"processed": 0, "ocr_success": 0, "still_empty": 0, "errors": 0}
    start_time = time.monotonic()

    for i, entry in enumerate(target):
        file_num = args.offset + i + 1
        logger.info(
            "[%d/%d] Processing: %s (%s bytes)",
            file_num, args.offset + len(target),
            entry.path_display, f"{entry.size:,}",
        )

        ext = Path(entry.name).suffix.lower()
        tmp_path: str | None = None

        try:
            # Download
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
            os.close(tmp_fd)
            dbx.files_download_to_file(tmp_path, entry.id)

            # Extract text (with OCR fallback)
            text = extract_text(Path(tmp_path), file_extension=ext)

            if not text.strip():
                logger.warning("  Still no text after OCR: %s", entry.path_display)
                stats["still_empty"] += 1
                continue

            # Chunk and index
            created_date = (
                entry.server_modified.isoformat()
                if entry.server_modified
                else datetime.now(timezone.utc).isoformat()
            )
            folder_path = str(Path(entry.path_display).parent)

            meta = DocumentMetadata(
                source_type="dropbox",
                source_id=entry.id,
                created_date=created_date,
                filename=entry.name,
                folder_path=folder_path,
                file_type=ext.lstrip("."),
            )
            chunks = chunker.split(text)
            inserted = indexer.index_document(chunks, meta)

            if inserted > 0:
                stats["ocr_success"] += 1
                logger.info(
                    "  Indexed %d chunks (%d chars) for %s",
                    inserted, len(text), entry.name,
                )
            else:
                stats["errors"] += 1

        except Exception:
            logger.exception("  Error processing: %s", entry.path_display)
            stats["errors"] += 1
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        stats["processed"] += 1

        # Progress report every 50 files
        if stats["processed"] % 50 == 0:
            elapsed = time.monotonic() - start_time
            rate = stats["processed"] / elapsed if elapsed > 0 else 0
            remaining = (len(target) - stats["processed"]) / rate if rate > 0 else 0
            logger.info(
                "  Progress: %d/%d (%.1f files/min, ~%.0f min remaining)",
                stats["processed"], len(target),
                rate * 60, remaining / 60,
            )

    elapsed = time.monotonic() - start_time
    logger.info("=" * 60)
    logger.info("OCR re-processing complete in %.1f min", elapsed / 60)
    logger.info("  Processed : %d", stats["processed"])
    logger.info("  OCR success: %d", stats["ocr_success"])
    logger.info("  Still empty: %d", stats["still_empty"])
    logger.info("  Errors     : %d", stats["errors"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
