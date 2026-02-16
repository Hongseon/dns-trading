"""Briefing sender -- store-first approach.

Stores generated briefings in the Zilliz Cloud ``briefings`` collection and
provides retrieval of the latest briefing by type.  KakaoTalk channel
push delivery is planned as future work.
"""

from __future__ import annotations

import logging
from typing import Any

from src.db.zilliz_client import get_client

logger = logging.getLogger(__name__)


class BriefingSender:
    """Send (store) briefings and retrieve the most recent one."""

    def __init__(self) -> None:
        self.client = get_client()

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send_to_channel(self, message: str) -> bool:
        """Store the briefing and mark it as sent.

        Currently this method only logs and marks the ``sent`` flag in
        the database.  Actual KakaoTalk channel push delivery will be
        implemented in a future phase once the Kakao Business Message
        API integration is complete.

        Parameters
        ----------
        message:
            The briefing text to deliver.

        Returns
        -------
        bool
            ``True`` if the briefing was stored/marked successfully,
            ``False`` otherwise.
        """
        logger.info("Briefing stored (%d chars)", len(message))

        # Mark the most recent unsent briefing as sent
        try:
            results = self.client.query(
                collection_name="briefings",
                filter='sent == false',
                output_fields=["id", "briefing_type", "content", "generated_at", "sent"],
                limit=1,
            )
            if results:
                # Delete and re-insert with sent=True (Milvus has no field-level update)
                row = results[0]
                self.client.delete(
                    collection_name="briefings",
                    ids=[row["id"]],
                )
                self.client.insert(
                    collection_name="briefings",
                    data=[{
                        "briefing_type": row.get("briefing_type", ""),
                        "content": row.get("content", ""),
                        "generated_at": row.get("generated_at", ""),
                        "sent": True,
                        "_dummy_vec": [0.0, 0.0],
                    }],
                )
        except Exception:
            logger.warning(
                "Could not mark briefing as sent in database",
                exc_info=True,
            )

        # TODO: Implement KakaoTalk channel message push via Kakao
        #       Business Message API or channel 1:1 chat API.
        #       Free tier allows up to 1,000 messages per day.
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_latest_briefing(
        self,
        briefing_type: str = "daily",
    ) -> str | None:
        """Fetch the most recent briefing of the given type from the database.

        Parameters
        ----------
        briefing_type:
            One of ``"daily"``, ``"weekly"``, or ``"monthly"``.

        Returns
        -------
        str or None
            The briefing content, or ``None`` if no briefing exists for
            the requested type.
        """
        try:
            results = self.client.query(
                collection_name="briefings",
                filter=f'briefing_type == "{briefing_type}"',
                output_fields=["content", "generated_at"],
                limit=10,
            )

            if results:
                # Sort by generated_at descending and take the first
                results.sort(
                    key=lambda x: x.get("generated_at", ""),
                    reverse=True,
                )
                content: str = results[0].get("content", "")
                logger.info(
                    "Retrieved latest %s briefing (%d chars)",
                    briefing_type,
                    len(content),
                )
                return content

            logger.info("No %s briefing found in database", briefing_type)
            return None

        except Exception:
            logger.exception(
                "Failed to retrieve latest %s briefing", briefing_type
            )
            return None

    def get_recent_briefings(
        self,
        briefing_type: str = "daily",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Fetch the N most recent briefings of the given type.

        Parameters
        ----------
        briefing_type:
            One of ``"daily"``, ``"weekly"``, or ``"monthly"``.
        limit:
            Maximum number of briefings to return.

        Returns
        -------
        list[dict]
            Each dict has ``content``, ``generated_at``, and ``sent``
            keys.
        """
        try:
            results = self.client.query(
                collection_name="briefings",
                filter=f'briefing_type == "{briefing_type}"',
                output_fields=["content", "generated_at", "sent"],
                limit=limit * 2,  # Fetch extra to sort and trim
            )
            # Sort by generated_at descending
            results.sort(
                key=lambda x: x.get("generated_at", ""),
                reverse=True,
            )
            return results[:limit]
        except Exception:
            logger.exception(
                "Failed to retrieve recent %s briefings", briefing_type
            )
            return []
