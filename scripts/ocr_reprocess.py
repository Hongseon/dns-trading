"""Re-process scanned image PDFs that were skipped during initial Dropbox sync.

Lists all PDF files in Dropbox, checks which ones are missing from Zilliz
(i.e. were skipped because text extraction returned empty), downloads them,
runs OCR via PaddleOCR, and indexes the extracted text.

Supports parallel processing with multiple worker processes for faster OCR.

Usage::

    # Dry run — list skipped PDFs without processing
    python scripts/ocr_reprocess.py --dry-run

    # Process all skipped PDFs (auto-detects worker count)
    python scripts/ocr_reprocess.py

    # Use 4 parallel workers
    python scripts/ocr_reprocess.py --workers 4

    # Limit to N files (useful for testing)
    python scripts/ocr_reprocess.py --limit 10

    # Resume from a specific file index (skip first N)
    python scripts/ocr_reprocess.py --offset 100

    # Override OCR DPI (default 150, higher = slower but more detail)
    python scripts/ocr_reprocess.py --dpi 200
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
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


# ======================================================================
# Dropbox & Zilliz helpers
# ======================================================================


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
    """Query Zilliz for all indexed Dropbox PDF source_ids."""
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


# ======================================================================
# Worker process globals & initializer
# ======================================================================

_w_dbx: Dropbox | None = None
_w_chunker: TextChunker | None = None
_w_indexer: Indexer | None = None


def _init_worker(dpi: int) -> None:
    """Called once per worker process to set up shared resources."""
    global _w_dbx, _w_chunker, _w_indexer

    # Override DPI before OCR engine is lazily loaded
    settings.ocr_dpi = dpi

    _w_dbx = _get_dropbox_client()
    _w_chunker = TextChunker()
    _w_indexer = Indexer()


def _process_one(entry_tuple: tuple) -> dict:
    """Process a single PDF file. Runs inside a worker process.

    Args:
        entry_tuple: (entry_id, entry_name, path_display, size, server_modified_iso)

    Returns:
        dict with keys: status ('indexed', 'empty', 'error'), name, chars, chunks
    """
    entry_id, entry_name, path_display, size, server_modified_iso = entry_tuple

    result = {
        "status": "error",
        "name": entry_name,
        "path": path_display,
        "chars": 0,
        "chunks": 0,
    }

    ext = Path(entry_name).suffix.lower()
    tmp_path: str | None = None

    try:
        # Download
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(tmp_fd)
        _w_dbx.files_download_to_file(tmp_path, entry_id)

        # Extract text (with OCR fallback)
        text = extract_text(Path(tmp_path), file_extension=ext)

        if not text.strip():
            result["status"] = "empty"
            return result

        # Chunk and index
        folder_path = str(Path(path_display).parent)
        meta = DocumentMetadata(
            source_type="dropbox",
            source_id=entry_id,
            created_date=server_modified_iso,
            filename=entry_name,
            folder_path=folder_path,
            file_type=ext.lstrip("."),
        )
        chunks = _w_chunker.split(text)
        inserted = _w_indexer.index_document(chunks, meta)

        if inserted > 0:
            result["status"] = "indexed"
            result["chars"] = len(text)
            result["chunks"] = inserted
        else:
            result["status"] = "error"

    except Exception as exc:
        logger.error("Worker error for %s: %s", path_display, exc)
        result["status"] = "error"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return result


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-process skipped scanned PDFs with OCR (parallel)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List skipped PDFs without processing"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max files to process (0 = all)"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip first N files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Number of parallel workers (default: half of CPU cores)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=settings.ocr_dpi,
        help=f"OCR rendering DPI (default: {settings.ocr_dpi})",
    )
    args = parser.parse_args()

    dbx = _get_dropbox_client()
    client = get_client()

    # 1. List all PDFs in Dropbox
    all_pdfs = _list_all_pdfs(dbx, settings.dropbox_folder_path)

    # 2. Find which are already indexed
    indexed_ids = _get_indexed_source_ids(client)

    # 3. Filter to unindexed (skipped) PDFs
    skipped = [pdf for pdf in all_pdfs if pdf.id not in indexed_ids]
    skipped.sort(key=lambda e: e.path_display or "")

    logger.info(
        "Skipped (unindexed) PDFs: %d / %d total",
        len(skipped),
        len(all_pdfs),
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
    target = skipped[args.offset :]
    if args.limit > 0:
        target = target[: args.limit]

    total = len(target)
    logger.info(
        "Processing %d files (offset=%d, limit=%s, workers=%d, dpi=%d)",
        total,
        args.offset,
        args.limit or "all",
        args.workers,
        args.dpi,
    )

    # Prepare serializable tuples for workers
    work_items: list[tuple] = []
    for entry in target:
        mod_iso = (
            entry.server_modified.isoformat()
            if entry.server_modified
            else datetime.now(timezone.utc).isoformat()
        )
        work_items.append(
            (entry.id, entry.name, entry.path_display, entry.size, mod_iso)
        )

    stats = {"indexed": 0, "empty": 0, "errors": 0}
    start_time = time.monotonic()

    # Process with multiprocessing pool
    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.dpi,),
    ) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_process_one, work_items), start=1
        ):
            status = result["status"]
            stats[status if status in stats else "errors"] += 1

            if status == "indexed":
                logger.info(
                    "[%d/%d] Indexed %d chunks (%d chars): %s",
                    i, total,
                    result["chunks"],
                    result["chars"],
                    result["name"],
                )
            elif status == "empty":
                logger.info("[%d/%d] No text (even with OCR): %s", i, total, result["name"])
            else:
                logger.warning("[%d/%d] Error: %s", i, total, result["path"])

            # Progress every 50 files
            if i % 50 == 0:
                elapsed = time.monotonic() - start_time
                rate = i / elapsed
                remaining = (total - i) / rate if rate > 0 else 0
                logger.info(
                    "  Progress: %d/%d (%.1f files/min, ~%.0f min remaining)",
                    i, total, rate * 60, remaining / 60,
                )

    elapsed = time.monotonic() - start_time
    logger.info("=" * 60)
    logger.info("OCR re-processing complete in %.1f min", elapsed / 60)
    logger.info("  Indexed   : %d", stats["indexed"])
    logger.info("  Still empty: %d", stats["empty"])
    logger.info("  Errors     : %d", stats["errors"])
    logger.info("  Total      : %d files in %.1f min", total, elapsed / 60)
    if total > 0:
        logger.info("  Avg speed  : %.1f files/min", total / (elapsed / 60))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
