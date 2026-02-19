"""Re-process all supported Dropbox files that are missing from Zilliz.

Lists all supported files in Dropbox, checks which ones are not yet indexed
in Zilliz, downloads them, extracts text, and indexes.

Usage::

    # Dry run — list missing files without processing
    python scripts/reprocess_missing.py --dry-run

    # Process all missing files with 2 workers
    python scripts/reprocess_missing.py --workers 2

    # Limit to N files (useful for testing)
    python scripts/reprocess_missing.py --limit 10
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dropbox import Dropbox
from dropbox.files import FileMetadata, FolderMetadata

from src.config import settings
from src.db.zilliz_client import get_client
from src.ingestion.chunker import TextChunker
from src.ingestion.indexer import DocumentMetadata, Indexer
from src.ingestion.text_extractor import extract_files_from_archive, extract_text

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


def _list_all_supported_files(dbx: Dropbox) -> list[FileMetadata]:
    """List all supported files in Dropbox recursively."""
    supported = set(settings.supported_extensions)
    max_size = settings.max_file_size_mb * 1024 * 1024
    files: list[FileMetadata] = []

    result = dbx.files_list_folder(settings.dropbox_folder_path, recursive=True)
    while True:
        for entry in result.entries:
            if not isinstance(entry, FileMetadata):
                continue
            ext = "." + entry.name.rsplit(".", 1)[-1].lower() if "." in entry.name else ""
            if ext not in supported:
                continue
            if entry.size > max_size:
                continue
            files.append(entry)
        if not result.has_more:
            break
        result = dbx.files_list_folder_continue(result.cursor)

    logger.info("Found %d supported files in Dropbox", len(files))
    return files


def _get_indexed_source_ids(client) -> set[str]:
    """Query Zilliz for all indexed Dropbox source_ids."""
    indexed: set[str] = set()
    try:
        results = client.query(
            collection_name="documents",
            filter='source_type == "dropbox" and chunk_index == 0',
            output_fields=["source_id"],
            limit=16384,
        )
        for row in results:
            sid = row.get("source_id")
            if sid:
                indexed.add(sid)
    except Exception:
        logger.exception("Failed to query indexed source_ids")
    logger.info("Found %d already-indexed source_ids in Zilliz", len(indexed))
    return indexed


# Worker process globals
_w_dbx: Dropbox | None = None
_w_chunker: TextChunker | None = None
_w_indexer: Indexer | None = None


def _init_worker() -> None:
    global _w_dbx, _w_chunker, _w_indexer
    _w_dbx = _get_dropbox_client()
    _w_chunker = TextChunker()
    _w_indexer = Indexer()


def _process_one(entry_tuple: tuple) -> dict:
    """Process a single file inside a worker process."""
    entry_id, entry_name, path_display, size, server_modified_iso, is_zip = entry_tuple

    result = {
        "status": "error",
        "name": entry_name,
        "path": path_display,
        "chars": 0,
        "chunks": 0,
    }

    ext = "." + entry_name.rsplit(".", 1)[-1].lower() if "." in entry_name else ""
    tmp_path: str | None = None

    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(tmp_fd)
        _w_dbx.files_download_to_file(tmp_path, entry_id)

        if is_zip:
            # ZIP: extract internal files
            archive_results = extract_files_from_archive(Path(tmp_path))
            if not archive_results:
                result["status"] = "empty"
                return result

            total_chunks = 0
            total_chars = 0
            folder_path = str(Path(path_display).parent)

            for internal_path, text in archive_results:
                if not text.strip():
                    continue
                inner_ext = "." + internal_path.rsplit(".", 1)[-1].lower() if "." in internal_path else ""
                meta = DocumentMetadata(
                    source_type="dropbox",
                    source_id=f"{entry_id}:{internal_path}",
                    created_date=server_modified_iso,
                    filename=Path(internal_path).name,
                    folder_path=f"{folder_path}/{entry_name}",
                    file_type=inner_ext.lstrip(".") or "zip",
                )
                chunks = _w_chunker.split(text)
                inserted = _w_indexer.index_document(chunks, meta)
                total_chunks += inserted
                total_chars += len(text)

            if total_chunks > 0:
                result["status"] = "indexed"
                result["chars"] = total_chars
                result["chunks"] = total_chunks
            else:
                result["status"] = "empty"
        else:
            # Regular file
            text = extract_text(Path(tmp_path), file_extension=ext)
            if not text.strip():
                result["status"] = "empty"
                return result

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
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-process missing Dropbox files")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    dbx = _get_dropbox_client()
    client = get_client()

    all_files = _list_all_supported_files(dbx)
    indexed_ids = _get_indexed_source_ids(client)

    # Filter to unindexed files (exclude PDFs - already handled by ocr_reprocess)
    missing = [f for f in all_files if f.id not in indexed_ids
               and not f.name.lower().endswith(".pdf")]
    missing.sort(key=lambda e: e.path_display or "")

    logger.info("Missing (unindexed) non-PDF files: %d", len(missing))

    if args.dry_run:
        from collections import Counter
        ext_counts = Counter()
        print(f"\n{'='*60}")
        print(f"DRY RUN — {len(missing)} missing files found:")
        print(f"{'='*60}")
        for i, f in enumerate(missing):
            ext = "." + f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
            ext_counts[ext] += 1
            print(f"  [{i+1:4d}] {f.path_display} ({f.size:,} bytes)")
        print(f"{'='*60}")
        print("By type:")
        for ext, cnt in ext_counts.most_common():
            print(f"  {ext}: {cnt}")
        return

    target = missing[args.offset:]
    if args.limit > 0:
        target = target[:args.limit]

    total = len(target)
    logger.info("Processing %d files (offset=%d, limit=%s, workers=%d)",
                total, args.offset, args.limit or "all", args.workers)

    work_items = []
    for entry in target:
        mod_iso = (
            entry.server_modified.isoformat()
            if entry.server_modified
            else datetime.now(timezone.utc).isoformat()
        )
        is_zip = entry.name.lower().endswith(".zip")
        work_items.append(
            (entry.id, entry.name, entry.path_display, entry.size, mod_iso, is_zip)
        )

    stats = {"indexed": 0, "empty": 0, "errors": 0}
    start_time = time.monotonic()

    with mp.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_one, work_items), start=1):
            status = result["status"]
            stats[status if status in stats else "errors"] += 1

            if status == "indexed":
                logger.info("[%d/%d] Indexed %d chunks (%d chars): %s",
                            i, total, result["chunks"], result["chars"], result["name"])
            elif status == "empty":
                logger.info("[%d/%d] No text: %s", i, total, result["name"])
            else:
                logger.warning("[%d/%d] Error: %s", i, total, result["path"])

            if i % 50 == 0:
                elapsed = time.monotonic() - start_time
                rate = i / elapsed
                remaining = (total - i) / rate if rate > 0 else 0
                logger.info("  Progress: %d/%d (%.1f files/min, ~%.0f min remaining)",
                            i, total, rate * 60, remaining / 60)

    elapsed = time.monotonic() - start_time
    logger.info("=" * 60)
    logger.info("Re-processing complete in %.1f min", elapsed / 60)
    logger.info("  Indexed   : %d", stats["indexed"])
    logger.info("  Empty     : %d", stats["empty"])
    logger.info("  Errors    : %d", stats["errors"])
    logger.info("  Total     : %d files in %.1f min", total, elapsed / 60)
    if total > 0:
        logger.info("  Avg speed : %.1f files/min", total / (elapsed / 60))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
