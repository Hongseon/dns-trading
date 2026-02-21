"""Naver Mail IMAP incremental sync.

Connects to Naver's IMAP server, fetches new emails since the last sync,
extracts body text and attachment content, then chunks, embeds, and indexes
everything into Zilliz Cloud.

Usage::

    syncer = NaverMailSync()
    result = syncer.sync()
    # result == {"processed": 5, "skipped": 1, "errors": 0}
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from imap_tools import AND, MailBox, MailMessage, MailMessageFlags

from src.config import settings
from src.db.zilliz_client import get_client
from src.ingestion.chunker import TextChunker
from src.ingestion.indexer import DocumentMetadata, Indexer
from src.ingestion.text_extractor import extract_text

logger = logging.getLogger(__name__)

# Folders to sync from Naver Mail
_FOLDERS_TO_SYNC: list[str] = [
    "INBOX",
    "Sent Messages",
    "CAS",
    "SOI",
    "Nanotech",
    "DAPA",
    "RKCC",
    "기타 재포장",
    "내게쓴메일함",
    "군사항공연구원",
    "주요편지",
    "구메일",
    "RedSun",
    "선사(선진로지스틱스)",
    "선사(선진로지스틱스)/하자 포워더(AGL)",
    "선사(선진로지스틱스)/PNL",
    "선사(선진로지스틱스)/MNL(FAST ROPE)",
    "선사(선진로지스틱스)/SKY RODE",
    "선사(선진로지스틱스)/CJ 대한통운",
]


class NaverMailSync:
    """Naver Mail IMAP incremental sync with date-based progression.

    On each run the syncer fetches all messages received on or after the
    last sync date (stored in ``sync_state``).  Each message body and its
    attachments are indexed as separate documents in Zilliz Cloud.
    """

    SYNC_TYPE = "email"

    def __init__(self) -> None:
        self._client = get_client()
        self._chunker = TextChunker()
        self._indexer = Indexer()

        self._imap_server = settings.naver_imap_server
        self._email = settings.naver_email
        self._password = settings.naver_password
        self._supported_extensions = set(settings.supported_extensions)
        self._max_file_size_bytes = settings.max_file_size_mb * 1024 * 1024


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self) -> dict[str, int]:
        """Run an incremental mail sync and return counters.

        Returns
        -------
        dict
            ``{"processed": N, "skipped": N, "errors": N}``
        """
        stats = {"processed": 0, "skipped": 0, "errors": 0}
        since_date = self._load_last_sync_date()

        logger.info(
            "Starting Naver Mail sync (since=%s, folders=%s)",
            since_date.isoformat() if since_date else "ALL",
            _FOLDERS_TO_SYNC,
        )

        try:
            with MailBox(self._imap_server).login(
                self._email,
                self._password,
            ) as mailbox:
                for folder in _FOLDERS_TO_SYNC:
                    self._sync_folder(mailbox, folder, since_date, stats)

            # Update last_sync_time on success
            self._save_last_sync_time()

            logger.info(
                "Naver Mail sync complete: processed=%d, skipped=%d, errors=%d",
                stats["processed"],
                stats["skipped"],
                stats["errors"],
            )
        except Exception:
            logger.exception("Naver Mail sync failed")
            stats["errors"] += 1

        return stats

    # ------------------------------------------------------------------
    # Folder / message processing
    # ------------------------------------------------------------------

    def _sync_folder(
        self,
        mailbox: MailBox,
        folder: str,
        since_date: date | None,
        stats: dict[str, int],
    ) -> None:
        """Fetch and process all messages in *folder* since *since_date*."""
        try:
            mailbox.folder.set(folder)
        except Exception:
            logger.warning("Could not select folder '%s', skipping", folder)
            return

        criteria = AND(date_gte=since_date) if since_date else "ALL"

        logger.info("Fetching messages from '%s'", folder)

        for msg in mailbox.fetch(
            criteria,
            mark_seen=False,
            bulk=True,
        ):
            try:
                self._process_message(msg, stats)
            except Exception:
                logger.exception(
                    "Error processing message uid=%s subject='%s'",
                    msg.uid,
                    msg.subject,
                )
                stats["errors"] += 1

    def _process_message(
        self,
        msg: MailMessage,
        stats: dict[str, int],
    ) -> None:
        """Index the body and attachments of a single email message."""
        # Quick check: skip if this message body is already indexed
        body_source_id = f"email:{msg.uid}:body"
        if self._is_indexed(body_source_id):
            logger.debug("Already indexed, skipping uid=%s", msg.uid)
            stats["skipped"] += 1
            return

        email_date = (
            msg.date.isoformat() if msg.date else datetime.now(timezone.utc).isoformat()
        )
        email_to = ", ".join(msg.to) if msg.to else ""

        # ---- Body ----
        body_text = self._extract_body(msg)
        if body_text and body_text.strip():

            meta = DocumentMetadata(
                source_type="email",
                source_id=body_source_id,
                created_date=email_date,
                email_from=msg.from_,
                email_to=email_to,
                email_subject=msg.subject,
                email_date=email_date,
            )

            chunks = self._chunker.split(body_text)
            if chunks:
                self._indexer.index_document(chunks, meta)
                logger.info(
                    "Indexed %d body chunks for uid=%s subject='%s'",
                    len(chunks),
                    msg.uid,
                    msg.subject,
                )
        else:
            logger.debug(
                "Empty body for uid=%s subject='%s'", msg.uid, msg.subject
            )

        # ---- Attachments ----
        for att in msg.attachments:
            self._process_attachment(msg, att, email_date, email_to, stats)

        stats["processed"] += 1

    # ------------------------------------------------------------------
    # Body extraction
    # ------------------------------------------------------------------

    def _extract_body(self, msg: MailMessage) -> str:
        """Extract clean text from an email message body.

        Prefers the HTML version (for richer structure) and falls back to
        the plain-text version.  HTML is cleaned via the text_extractor's
        HTML pipeline (write to temp .html file and call extract_text).
        """
        html_body: str = msg.html or ""
        text_body: str = msg.text or ""

        if html_body.strip():
            return self._html_to_text(html_body)

        return text_body.strip()

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Convert an HTML string to plain text using the text extractor.

        Writes the HTML to a temporary ``.html`` file and delegates to
        :func:`extract_text` so that signature/disclaimer stripping and
        encoding logic are reused.
        """
        tmp_path: str | None = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".html")
            os.close(tmp_fd)
            Path(tmp_path).write_text(html, encoding="utf-8")
            return extract_text(Path(tmp_path), file_extension=".html")
        except Exception:
            logger.exception("HTML body extraction failed")
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Attachment processing
    # ------------------------------------------------------------------

    def _process_attachment(
        self,
        msg: MailMessage,
        att,
        email_date: str,
        email_to: str,
        stats: dict[str, int],
    ) -> None:
        """Save an attachment to a temp file, extract text, and index it."""
        filename: str = att.filename or ""
        if not filename:
            return

        ext = Path(filename).suffix.lower()
        if ext not in self._supported_extensions:
            logger.debug(
                "Skipping unsupported attachment extension '%s': %s",
                ext,
                filename,
            )
            stats["skipped"] += 1
            return

        payload: bytes = att.payload
        if not payload:
            return

        if len(payload) > self._max_file_size_bytes:
            logger.warning(
                "Skipping oversized attachment (%d bytes): %s",
                len(payload),
                filename,
            )
            stats["skipped"] += 1
            return

        tmp_path: str | None = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
            os.close(tmp_fd)
            Path(tmp_path).write_bytes(payload)

            text = extract_text(Path(tmp_path), file_extension=ext)
            if not text.strip():
                logger.debug(
                    "No text from attachment %s in uid=%s", filename, msg.uid
                )
                return

            att_source_id = f"email:{msg.uid}:att:{filename}"

            meta = DocumentMetadata(
                source_type="email",
                source_id=att_source_id,
                created_date=email_date,
                filename=filename,
                file_type=ext.lstrip("."),
                email_from=msg.from_,
                email_to=email_to,
                email_subject=msg.subject,
                email_date=email_date,
            )

            chunks = self._chunker.split(text)
            if chunks:
                self._indexer.index_document(chunks, meta)
                logger.info(
                    "Indexed %d chunks for attachment '%s' (uid=%s)",
                    len(chunks),
                    filename,
                    msg.uid,
                )
        except Exception:
            logger.exception(
                "Error processing attachment '%s' for uid=%s",
                filename,
                msg.uid,
            )
            stats["errors"] += 1
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    def _is_indexed(self, source_id: str) -> bool:
        """Check whether *source_id* already has rows in the documents collection."""
        try:
            escaped_id = source_id.replace('"', '\\"')
            results = self._client.query(
                collection_name="documents",
                filter=f'source_id == "{escaped_id}"',
                output_fields=["source_id"],
                limit=1,
            )
            return bool(results)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Sync-state persistence
    # ------------------------------------------------------------------

    def _load_last_sync_date(self) -> date | None:
        """Load the last sync timestamp from ``sync_state`` and return it
        as a :class:`date` suitable for IMAP SINCE queries.

        Returns ``None`` if no previous sync has been recorded (triggers a
        full initial sync).
        """
        try:
            results = self._client.query(
                collection_name="sync_state",
                filter=f'sync_type == "{self.SYNC_TYPE}"',
                output_fields=["last_sync_time"],
            )
            if results:
                raw = results[0].get("last_sync_time")
                if raw:
                    dt = datetime.fromisoformat(raw)
                    logger.debug("Last email sync time: %s", dt.isoformat())
                    return dt.date()
        except Exception:
            logger.exception("Failed to load email sync state")
        return None

    def _save_last_sync_time(self) -> None:
        """Persist the current UTC time as the last email sync time."""
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "sync_type": self.SYNC_TYPE,
            "last_sync_time": now,
            "updated_at": now,
            "_dummy_vec": [0.0, 0.0],
        }
        try:
            self._client.upsert(
                collection_name="sync_state",
                data=[row],
            )
            logger.debug("Saved email sync state")
        except Exception:
            logger.exception("Failed to save email sync state")
