"""Dropbox cursor-based incremental sync.

Connects to the Dropbox API, detects file changes via a stored cursor,
downloads new/modified files, extracts text, chunks, embeds, and indexes
into Zilliz Cloud.  Deleted files are removed from the index.

Usage::

    syncer = DropboxSync()
    result = syncer.sync()
    # result == {"added": 3, "deleted": 1, "skipped": 2, "errors": 0}
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dropbox import Dropbox
from dropbox.files import (
    DeletedMetadata,
    FileMetadata,
    FolderMetadata,
    ListFolderResult,
)

from src.config import settings
from src.db.zilliz_client import get_client
from src.ingestion.chunker import TextChunker
from src.ingestion.indexer import DocumentMetadata, Indexer
from src.ingestion.text_extractor import extract_files_from_archive, extract_text

logger = logging.getLogger(__name__)


class DropboxSync:
    """Dropbox incremental file sync with cursor persistence.

    On the first run the full folder listing is fetched and every supported
    file is processed.  Subsequent runs use the saved cursor so that only
    changed files are downloaded and indexed.
    """

    SYNC_TYPE = "dropbox"

    def __init__(self) -> None:
        if settings.dropbox_refresh_token:
            self._dbx = Dropbox(
                app_key=settings.dropbox_app_key,
                app_secret=settings.dropbox_app_secret,
                oauth2_refresh_token=settings.dropbox_refresh_token,
            )
        else:
            self._dbx = Dropbox(settings.dropbox_access_token)
        self._client = get_client()
        self._chunker = TextChunker()
        self._indexer = Indexer()

        self._folder_path = settings.dropbox_folder_path
        self._supported_extensions = set(settings.supported_extensions)
        self._max_file_size_bytes = settings.max_file_size_mb * 1024 * 1024

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self) -> dict[str, int]:
        """Run an incremental (or initial) sync and return counters.

        Returns
        -------
        dict
            ``{"added": N, "deleted": N, "skipped": N, "errors": N}``
        """
        stats = {"added": 0, "deleted": 0, "skipped": 0, "errors": 0}

        cursor = self._load_cursor()

        try:
            if cursor:
                logger.info("Resuming Dropbox sync with existing cursor")
                result: ListFolderResult = self._dbx.files_list_folder_continue(cursor)
            else:
                logger.info(
                    "Starting initial Dropbox sync for folder: %s",
                    self._folder_path,
                )
                result = self._dbx.files_list_folder(
                    self._folder_path, recursive=True
                )

            # Process all pages (has_more pagination loop)
            while True:
                for entry in result.entries:
                    self._process_entry(entry, stats)

                if not result.has_more:
                    break
                result = self._dbx.files_list_folder_continue(result.cursor)

            # Persist the latest cursor on success
            self._save_cursor(result.cursor)
            logger.info(
                "Dropbox sync complete: added=%d, deleted=%d, skipped=%d, errors=%d",
                stats["added"],
                stats["deleted"],
                stats["skipped"],
                stats["errors"],
            )
        except Exception:
            logger.exception("Dropbox sync failed")
            stats["errors"] += 1

        return stats

    # ------------------------------------------------------------------
    # Entry processing
    # ------------------------------------------------------------------

    def _process_entry(
        self,
        entry: FileMetadata | DeletedMetadata | FolderMetadata,
        stats: dict[str, int],
    ) -> None:
        """Dispatch a single Dropbox list-folder entry."""
        if isinstance(entry, FolderMetadata):
            # Folders don't need indexing
            return

        if isinstance(entry, DeletedMetadata):
            self._handle_deleted(entry, stats)
            return

        if isinstance(entry, FileMetadata):
            self._handle_file(entry, stats)
            return

        logger.debug("Unknown entry type: %s", type(entry).__name__)

    def _handle_deleted(
        self, entry: DeletedMetadata, stats: dict[str, int]
    ) -> None:
        """Remove all indexed chunks whose filename + folder_path match the
        deleted Dropbox path."""
        path_lower = entry.path_lower or entry.path_display or ""
        if not path_lower:
            return

        filename = Path(path_lower).name
        folder_path = str(Path(path_lower).parent)

        try:
            escaped_filename = filename.replace('"', '\\"')
            escaped_folder = folder_path.replace('"', '\\"')
            self._client.delete(
                collection_name="documents",
                filter=(
                    f'source_type == "dropbox" '
                    f'and filename == "{escaped_filename}" '
                    f'and folder_path == "{escaped_folder}"'
                ),
            )
            logger.info("Deleted index entries for: %s", path_lower)
            stats["deleted"] += 1
        except Exception:
            logger.exception("Failed to delete index for: %s", path_lower)
            stats["errors"] += 1

    def _handle_file(
        self, entry: FileMetadata, stats: dict[str, int]
    ) -> None:
        """Download, extract, chunk, and index a single Dropbox file."""
        if not self._should_process(entry):
            stats["skipped"] += 1
            return

        ext = Path(entry.name).suffix.lower()
        tmp_path: str | None = None

        try:
            # Download to a temporary file
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
            os.close(tmp_fd)

            logger.info(
                "Downloading %s (%d bytes)", entry.path_display, entry.size
            )
            self._dbx.files_download_to_file(tmp_path, entry.id)

            tmp_file = Path(tmp_path)
            created_date = (
                entry.server_modified.isoformat()
                if entry.server_modified
                else datetime.now(timezone.utc).isoformat()
            )
            folder_path = str(Path(entry.path_display).parent)

            if ext == ".zip":
                self._process_zip(entry, tmp_file, folder_path, created_date, stats)
            else:
                self._process_regular_file(
                    entry, tmp_file, ext, folder_path, created_date, stats
                )
        except Exception:
            logger.exception(
                "Error processing Dropbox file: %s", entry.path_display
            )
            stats["errors"] += 1
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # File-type handlers
    # ------------------------------------------------------------------

    def _process_regular_file(
        self,
        entry: FileMetadata,
        tmp_file: Path,
        ext: str,
        folder_path: str,
        created_date: str,
        stats: dict[str, int],
    ) -> None:
        """Extract, chunk, and index a single non-archive file."""
        text = extract_text(tmp_file, file_extension=ext)
        if not text.strip():
            logger.warning("No text extracted from: %s", entry.path_display)
            stats["skipped"] += 1
            return

        meta = DocumentMetadata(
            source_type="dropbox",
            source_id=entry.id,
            created_date=created_date,
            filename=entry.name,
            folder_path=folder_path,
            file_type=ext.lstrip("."),
        )

        chunks = self._chunker.split(text)
        inserted = self._indexer.index_document(chunks, meta)
        if inserted > 0:
            stats["added"] += 1
            logger.info(
                "Indexed %d chunks for %s", inserted, entry.path_display
            )
        else:
            stats["errors"] += 1

    def _process_zip(
        self,
        entry: FileMetadata,
        tmp_file: Path,
        folder_path: str,
        created_date: str,
        stats: dict[str, int],
    ) -> None:
        """Extract supported files from a ZIP and index each separately."""
        extracted_files = extract_files_from_archive(tmp_file)
        if not extracted_files:
            logger.warning("No extractable content in ZIP: %s", entry.path_display)
            stats["skipped"] += 1
            return

        for internal_path, text in extracted_files:
            if not text.strip():
                continue

            source_id = f"{entry.id}:{internal_path}"
            internal_ext = Path(internal_path).suffix.lower()

            meta = DocumentMetadata(
                source_type="dropbox",
                source_id=source_id,
                created_date=created_date,
                filename=entry.name,
                folder_path=folder_path,
                file_type=internal_ext.lstrip("."),
            )

            chunks = self._chunker.split(text)
            inserted = self._indexer.index_document(chunks, meta)
            if inserted > 0:
                stats["added"] += 1
                logger.info(
                    "Indexed %d chunks for %s [%s]",
                    inserted,
                    entry.path_display,
                    internal_path,
                )
            else:
                stats["errors"] += 1

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _should_process(self, entry: FileMetadata) -> bool:
        """Decide whether a FileMetadata entry should be downloaded and indexed.

        Skipped when:
        - Extension is not in the supported list.
        - File size exceeds the configured maximum.
        """
        ext = Path(entry.name).suffix.lower()
        if ext not in self._supported_extensions:
            logger.debug("Skipping unsupported extension: %s", entry.name)
            return False

        if entry.size > self._max_file_size_bytes:
            logger.warning(
                "Skipping oversized file (%d bytes > %d max): %s",
                entry.size,
                self._max_file_size_bytes,
                entry.path_display,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Cursor persistence (sync_state collection)
    # ------------------------------------------------------------------

    def _load_cursor(self) -> str | None:
        """Load the Dropbox sync cursor from the ``sync_state`` collection."""
        try:
            results = self._client.query(
                collection_name="sync_state",
                filter=f'sync_type == "{self.SYNC_TYPE}"',
                output_fields=["last_cursor"],
            )
            if results:
                cursor = results[0].get("last_cursor")
                if cursor:
                    logger.debug("Loaded Dropbox cursor: %s...", cursor[:20])
                return cursor
        except Exception:
            logger.exception("Failed to load Dropbox cursor from sync_state")
        return None

    def _save_cursor(self, cursor: str) -> None:
        """Persist the Dropbox sync cursor to the ``sync_state`` collection."""
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "sync_type": self.SYNC_TYPE,
            "last_cursor": cursor,
            "last_sync_time": now,
            "updated_at": now,
            "_dummy_vec": [0.0, 0.0],
        }
        try:
            self._client.upsert(
                collection_name="sync_state",
                data=[row],
            )
            logger.debug("Saved Dropbox cursor")
        except Exception:
            logger.exception("Failed to save Dropbox cursor to sync_state")
