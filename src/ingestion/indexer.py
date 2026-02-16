"""Document chunk indexer for Zilliz Cloud (Milvus).

Responsible for embedding text chunks and upserting them into the
``documents`` collection.  On update the old rows for a given ``source_id``
are deleted first so stale chunks never linger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from src.db.zilliz_client import get_client
from src.ingestion.chunker import Chunk
from src.rag.embedder import Embedder

logger = logging.getLogger(__name__)

# Maximum rows per Milvus insert call.
_BATCH_SIZE = 100


@dataclass
class DocumentMetadata:
    """Metadata attached to every chunk of a single source document."""

    source_type: str  # 'dropbox' or 'email'
    source_id: str  # Dropbox file_id or email message_id

    # Common
    created_date: str | None = None

    # Dropbox-specific
    filename: str | None = None
    folder_path: str | None = None
    file_type: str | None = None

    # Email-specific
    email_from: str | None = None
    email_to: str | None = None
    email_subject: str | None = None
    email_date: str | None = None


class Indexer:
    """Embeds and indexes document chunks into Zilliz Cloud."""

    def __init__(self) -> None:
        self._embedder = Embedder()
        self._client = get_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_document(
        self,
        chunks: list[Chunk],
        metadata: DocumentMetadata,
    ) -> int:
        """Delete existing rows for *source_id*, then embed and insert *chunks*.

        Parameters
        ----------
        chunks:
            Ordered list of :class:`Chunk` objects produced by the chunker.
        metadata:
            Shared metadata for every chunk (source info, dates, etc.).

        Returns
        -------
        int
            The number of chunk rows successfully inserted.
        """
        if not chunks:
            logger.warning(
                "index_document called with empty chunks for source_id=%s",
                metadata.source_id,
            )
            return 0

        # 1. Remove previous version of this document.
        self.delete_document(metadata.source_id)

        # 2. Embed all chunk texts.
        texts = [c.text for c in chunks]
        logger.info(
            "Embedding %d chunks for source_id=%s",
            len(texts),
            metadata.source_id,
        )
        embeddings = self._embedder.embed_batch(texts)

        # 3. Build row dicts.
        rows = self._build_rows(chunks, embeddings, metadata)

        # 4. Batch insert.
        inserted = self._batch_insert(rows)
        logger.info(
            "Indexed %d/%d chunks for source_id=%s",
            inserted,
            len(rows),
            metadata.source_id,
        )
        return inserted

    def delete_document(self, source_id: str) -> None:
        """Remove all chunk rows for a given *source_id*."""
        try:
            # Escape double quotes in source_id for filter expression
            escaped_id = source_id.replace('"', '\\"')
            self._client.delete(
                collection_name="documents",
                filter=f'source_id == "{escaped_id}"',
            )
            logger.debug("Deleted existing rows for source_id=%s", source_id)
        except Exception:
            logger.exception(
                "Failed to delete rows for source_id=%s", source_id
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rows(
        chunks: list[Chunk],
        embeddings: list[list[float]],
        metadata: DocumentMetadata,
    ) -> list[dict]:
        """Combine chunks, embeddings, and metadata into insert-ready dicts."""
        now = datetime.now(timezone.utc).isoformat()
        rows: list[dict] = []
        for chunk, embedding in zip(chunks, embeddings):
            row: dict = {
                "source_type": metadata.source_type,
                "source_id": metadata.source_id,
                "content": chunk.text[:10000],  # VARCHAR limit
                "embedding": embedding,
                "chunk_index": chunk.chunk_index,
                "updated_date": now,
                # Default empty strings for optional fields
                "created_date": metadata.created_date or "",
                "filename": metadata.filename or "",
                "folder_path": metadata.folder_path or "",
                "file_type": metadata.file_type or "",
                "email_from": metadata.email_from or "",
                "email_to": metadata.email_to or "",
                "email_subject": metadata.email_subject or "",
                "email_date": metadata.email_date or "",
            }
            rows.append(row)
        return rows

    def _batch_insert(self, rows: list[dict]) -> int:
        """Insert *rows* in batches.  On batch failure, retry row-by-row."""
        inserted = 0

        for start in range(0, len(rows), _BATCH_SIZE):
            batch = rows[start : start + _BATCH_SIZE]
            try:
                self._client.insert(
                    collection_name="documents",
                    data=batch,
                )
                inserted += len(batch)
            except Exception:
                logger.warning(
                    "Batch insert failed (rows %d-%d), retrying individually",
                    start,
                    start + len(batch) - 1,
                    exc_info=True,
                )
                inserted += self._insert_individually(batch)

        return inserted

    def _insert_individually(self, rows: list[dict]) -> int:
        """Insert rows one at a time, logging and skipping failures."""
        count = 0
        for row in rows:
            try:
                self._client.insert(
                    collection_name="documents",
                    data=[row],
                )
                count += 1
            except Exception:
                logger.error(
                    "Failed to insert chunk (source_id=%s, chunk_index=%s)",
                    row.get("source_id"),
                    row.get("chunk_index"),
                    exc_info=True,
                )
        return count
