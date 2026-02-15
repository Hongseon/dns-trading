"""KakaoTalk OpenBuilder skill endpoints.

Handles incoming skill requests from the Kakao i OpenBuilder, dispatches
them to the RAG chain (with callback support for the 5-second timeout),
and returns properly formatted KakaoTalk JSON responses.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request

from src.rag.chain import get_chain
from src.server.callback import process_and_callback, process_briefing_and_callback

logger = logging.getLogger(__name__)

router = APIRouter()

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def make_kakao_response(text: str, use_callback: bool = False) -> dict:
    """Build a KakaoTalk OpenBuilder JSON response.

    Parameters
    ----------
    text:
        The reply text.  Automatically truncated to 1 000 characters.
    use_callback:
        If ``True``, include ``"useCallback": True`` so the OpenBuilder
        knows a callback will follow.
    """
    response: dict = {
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": text[:1000]}}
            ],
        },
    }
    if use_callback:
        response["useCallback"] = True
    return response


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/skill/query")
async def skill_query(request: Request):
    """RAG query skill -- the fallback block in OpenBuilder routes here.

    * **With callbackUrl** (AI chatbot mode): returns an immediate
      placeholder, then fires a background task that POSTs the real
      answer to the callback URL within 1 minute.
    * **Without callbackUrl**: attempts a quick RAG run within 4.5 s.
      If it times out, returns a friendly error.
    """
    body = await request.json()

    utterance: str = body.get("userRequest", {}).get("utterance", "").strip()
    callback_url: str | None = body.get("userRequest", {}).get("callbackUrl")

    if not utterance:
        return make_kakao_response("질문을 입력해 주세요.")

    logger.info("skill/query  utterance=%s  callback=%s", utterance[:60], bool(callback_url))

    # -- Callback mode (AI chatbot) --------------------------------
    if callback_url:
        asyncio.create_task(process_and_callback(utterance, callback_url))
        return make_kakao_response("문서를 검색하고 있습니다...", use_callback=True)

    # -- Direct mode (must reply within 5 s) -----------------------
    try:
        chain = get_chain()
        answer = await asyncio.wait_for(chain.quick_run(utterance), timeout=4.5)
        return make_kakao_response(answer)
    except asyncio.TimeoutError:
        logger.warning("quick_run timed out for: %s", utterance[:60])
        return make_kakao_response(
            "응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
        )
    except Exception:
        logger.exception("skill/query direct mode failed")
        return make_kakao_response(
            "문서 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        )


@router.post("/skill/briefing")
async def skill_briefing(request: Request):
    """Briefing skill -- generates daily/weekly/monthly briefings.

    The utterance is inspected to determine the briefing type:

    * ``"주간"`` or ``"이번 주"`` -> ``"weekly"``
    * ``"월간"`` or ``"이번 달"`` -> ``"monthly"``
    * everything else           -> ``"daily"``
    """
    body = await request.json()

    utterance: str = body.get("userRequest", {}).get("utterance", "").strip()
    callback_url: str | None = body.get("userRequest", {}).get("callbackUrl")

    # Determine briefing type from natural-language utterance
    briefing_type = _detect_briefing_type(utterance)

    logger.info(
        "skill/briefing  type=%s  utterance=%s  callback=%s",
        briefing_type,
        utterance[:60],
        bool(callback_url),
    )

    if callback_url:
        asyncio.create_task(
            process_briefing_and_callback(briefing_type, callback_url)
        )
        return make_kakao_response("브리핑을 생성하고 있습니다...", use_callback=True)

    # Briefing generation is too slow for the 5-second limit.
    return make_kakao_response(
        "브리핑 생성에는 시간이 필요합니다. 콜백이 활성화되지 않았습니다."
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_WEEKLY_KEYWORDS = ("주간", "이번 주", "이번주", "금주")
_MONTHLY_KEYWORDS = ("월간", "이번 달", "이번달", "금월")


def _detect_briefing_type(utterance: str) -> str:
    """Infer the briefing period from the user's utterance."""
    for kw in _WEEKLY_KEYWORDS:
        if kw in utterance:
            return "weekly"
    for kw in _MONTHLY_KEYWORDS:
        if kw in utterance:
            return "monthly"
    return "daily"
