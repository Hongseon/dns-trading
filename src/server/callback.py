"""Background callback processing for KakaoTalk OpenBuilder.

The KakaoTalk skill server must reply within 5 seconds.  For heavier
operations (full RAG, briefing generation) we return an immediate
placeholder with ``useCallback: true`` and then POST the real result
to the one-time ``callbackUrl`` provided by the OpenBuilder.

Constraints of the callback URL (KakaoTalk):
  * Valid for **1 minute** after the original request.
  * Usable exactly **once**.
  * Cannot be tested in the bot-tester -- deploy and test in the
    real KakaoTalk client.
"""

from __future__ import annotations

import logging

import httpx

from src.rag.chain import get_chain

logger = logging.getLogger(__name__)

_CALLBACK_TIMEOUT = 10.0  # seconds for the outbound POST


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_callback_payload(text: str) -> dict:
    """Build the JSON payload to POST back to the KakaoTalk callback URL.

    KakaoTalk limits each simpleText to 1 000 characters and allows up to
    3 outputs per response.  If *text* exceeds 1 000 chars, it is split
    into multiple bubbles at line boundaries.
    """
    chunks = _split_text_for_kakao(text, max_chars=1000, max_outputs=3)
    outputs = [{"simpleText": {"text": c}} for c in chunks]
    return {
        "version": "2.0",
        "template": {"outputs": outputs},
    }


def _split_text_for_kakao(
    text: str, max_chars: int = 1000, max_outputs: int = 3
) -> list[str]:
    """Split *text* into up to *max_outputs* chunks of at most *max_chars*.

    Splits at the last newline before the limit so sections stay intact.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining and len(chunks) < max_outputs:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        # Find a newline near the end of the allowed range
        cut = remaining.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars  # no good newline, hard cut

        chunk = remaining[:cut].rstrip()
        remaining = remaining[cut:].lstrip("\n")

        if chunk:
            chunks.append(chunk)

    # If there's still text left after max_outputs, append ellipsis
    if remaining and len(chunks) == max_outputs:
        last = chunks[-1]
        if len(last) + 4 <= max_chars:
            chunks[-1] = last + "\n..."
        else:
            chunks[-1] = last[:max_chars - 3] + "..."

    return chunks


async def _post_callback(callback_url: str, payload: dict) -> None:
    """POST *payload* to the KakaoTalk callback URL."""
    try:
        async with httpx.AsyncClient(timeout=_CALLBACK_TIMEOUT) as client:
            resp = await client.post(callback_url, json=payload)
            resp.raise_for_status()
            logger.info("Callback POST succeeded (%s)", resp.status_code)
    except httpx.TimeoutException:
        logger.error("Callback POST timed out for URL: %s", callback_url)
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Callback POST returned %s: %s",
            exc.response.status_code,
            exc.response.text[:200],
        )
    except Exception:
        logger.exception("Callback POST failed unexpectedly")


# ------------------------------------------------------------------
# Public coroutines (launched via asyncio.create_task)
# ------------------------------------------------------------------


async def process_and_callback(utterance: str, callback_url: str) -> None:
    """Run the full RAG pipeline in the background, then POST the answer.

    Parameters
    ----------
    utterance:
        The user's question extracted from the KakaoTalk skill request.
    callback_url:
        The one-time callback URL provided by the KakaoTalk OpenBuilder.
    """
    try:
        chain = get_chain()
        answer = await chain.run(utterance)
    except Exception:
        logger.exception("RAG processing failed for: %s", utterance[:60])
        answer = "문서 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    payload = _build_callback_payload(answer)
    await _post_callback(callback_url, payload)


async def process_briefing_and_callback(
    briefing_type: str,
    callback_url: str,
) -> None:
    """Generate a briefing in the background, then POST it via callback.

    Parameters
    ----------
    briefing_type:
        One of ``"daily"``, ``"weekly"``, or ``"monthly"``.
    callback_url:
        The one-time callback URL provided by the KakaoTalk OpenBuilder.
    """
    try:
        # Late import -- the briefing module is optional and may not yet
        # exist during early development.
        from src.briefing.generator import BriefingGenerator

        bg = BriefingGenerator()
        content = await bg.generate(briefing_type)
    except ImportError:
        logger.error("Briefing module not available (src.briefing.generator)")
        content = "브리핑 모듈이 아직 준비되지 않았습니다."
    except Exception:
        logger.exception("Briefing generation failed (type=%s)", briefing_type)
        content = "브리핑 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    payload = _build_callback_payload(content)
    await _post_callback(callback_url, payload)
