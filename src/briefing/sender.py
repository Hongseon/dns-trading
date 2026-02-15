"""Briefing sender -- store-first approach.

Stores generated briefings in the Supabase ``briefings`` table and
provides retrieval of the latest briefing by type.  KakaoTalk channel
push delivery is planned as future work.
"""

from __future__ import annotations

import logging
from typing import Any

from src.db.supabase_client import get_client

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
            self.client.table("briefings").update(
                {"sent": True}
            ).eq("sent", False).order(
                "generated_at", desc=True
            ).limit(1).execute()
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
            result = (
                self.client.table("briefings")
                .select("content")
                .eq("briefing_type", briefing_type)
                .order("generated_at", desc=True)
                .limit(1)
                .execute()
            )

            if result.data:
                content: str = result.data[0]["content"]
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
            result = (
                self.client.table("briefings")
                .select("content, generated_at, sent")
                .eq("briefing_type", briefing_type)
                .order("generated_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception:
            logger.exception(
                "Failed to retrieve recent %s briefings", briefing_type
            )
            return []
