"""Gemini LLM wrapper with primary/fallback model support.

Uses the ``google-genai`` SDK to call Gemini models for RAG answer
generation.  If the primary model (configurable, defaults to
``gemini-3-flash-preview``) fails, the generator automatically retries
with a fallback model (``gemini-2.5-flash``).
"""

from __future__ import annotations

import logging

from google import genai
from google.genai import types as genai_types

from src.config import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "당신은 팀의 업무 문서와 이메일을 기반으로 질문에 답변하는 AI 어시스턴트입니다.\n"
    "\n"
    "규칙:\n"
    "1. 제공된 문서 컨텍스트만을 기반으로 답변하세요.\n"
    "2. 문서에 없는 내용은 \"해당 정보를 찾을 수 없습니다\"라고 답변하세요.\n"
    "3. 답변 끝에 출처(파일명 또는 이메일 제목)를 표기하세요.\n"
    "4. 한국어로 답변하세요.\n"
    "5. 간결하되 핵심 정보를 빠뜨리지 마세요.\n"
    "6. 답변은 800자 이내로 작성하세요."
)

# ------------------------------------------------------------------
# User prompt template
# ------------------------------------------------------------------

_USER_PROMPT_TEMPLATE = (
    "다음은 질문과 관련된 문서 컨텍스트입니다:\n"
    "\n"
    "{context}\n"
    "\n"
    "---\n"
    "질문: {query}\n"
    "\n"
    "위 컨텍스트를 기반으로 질문에 답변하세요."
)


class Generator:
    """Async Gemini LLM generator with automatic model fallback."""

    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY must be set.  Check your .env file or "
                "environment variables."
            )

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self.fallback_model = settings.gemini_fallback_model

        logger.info(
            "Generator initialised (primary=%s, fallback=%s)",
            self.model,
            self.fallback_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        query: str,
        context: str,
        sources: list[str],
    ) -> str:
        """Generate an RAG answer from *context* for the user *query*.

        The method builds a user prompt that includes the retrieved context
        and the original question, calls the LLM, and appends source
        citations if the model did not already include them.

        Parameters
        ----------
        query:
            The user's natural-language question.
        context:
            Formatted document context from :meth:`Retriever.format_context`.
        sources:
            List of citation strings from :meth:`Retriever.extract_sources`.

        Returns
        -------
        str
            The generated answer with source citations appended.
        """
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        # Try primary model, fall back on any exception
        answer = await self._call_with_fallback(user_prompt)

        # Append source citations if the model did not include them
        answer = self._ensure_sources(answer, sources)

        return answer

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------

    async def _call_with_fallback(
        self,
        user_prompt: str,
        system_instruction: str | None = None,
        max_output_tokens: int = 1024,
    ) -> str:
        """Try the primary model; on failure, retry with the fallback.

        Parameters
        ----------
        user_prompt:
            The prompt to send.
        system_instruction:
            Optional override for the system prompt.  Defaults to
            :data:`SYSTEM_PROMPT` when ``None``.
        max_output_tokens:
            Maximum output tokens (default 1024).
        """
        try:
            return await self._call_llm(
                self.model, user_prompt, system_instruction, max_output_tokens,
            )
        except Exception:
            logger.warning(
                "Primary model %s failed, falling back to %s",
                self.model,
                self.fallback_model,
                exc_info=True,
            )

        try:
            return await self._call_llm(
                self.fallback_model, user_prompt, system_instruction, max_output_tokens,
            )
        except Exception:
            logger.exception(
                "Fallback model %s also failed", self.fallback_model
            )
            return (
                "죄송합니다. 현재 답변을 생성할 수 없습니다. "
                "잠시 후 다시 시도해 주세요."
            )

    async def _call_llm(
        self,
        model: str,
        user_prompt: str,
        system_instruction: str | None = None,
        max_output_tokens: int = 1024,
    ) -> str:
        """Call a single Gemini model and return the text response.

        Parameters
        ----------
        model:
            The model identifier (e.g. ``"gemini-3-flash-preview"``).
        user_prompt:
            The fully assembled user prompt including context and query.
        system_instruction:
            Optional override for the system prompt.
        max_output_tokens:
            Maximum output tokens.

        Returns
        -------
        str
            The model's text response.

        Raises
        ------
        Exception
            Any exception from the GenAI SDK is propagated so that
            :meth:`_call_with_fallback` can handle it.
        """
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction or SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=max_output_tokens,
        )

        logger.debug("Calling model=%s, prompt length=%d", model, len(user_prompt))

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config,
        )

        # Extract text from the response
        text = response.text or ""
        text = text.strip()

        if not text:
            logger.warning("Model %s returned empty response", model)
            return "답변을 생성하지 못했습니다. 다시 시도해 주세요."

        logger.info(
            "Model %s responded (%d chars)", model, len(text)
        )
        return text

    # ------------------------------------------------------------------
    # Source citation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_sources(answer: str, sources: list[str]) -> str:
        """Append source citations to *answer* if not already present.

        The model is instructed to include sources, but this method acts as
        a safety net.  If none of the source strings appear in the answer,
        a ``출처`` section is appended.
        """
        if not sources:
            return answer

        # Check if any source already appears in the answer
        sources_present = any(src in answer for src in sources)
        if sources_present:
            return answer

        # Also check for a Korean "출처" section that the model might have
        # written with slightly different formatting
        if "출처:" in answer or "출처 :" in answer:
            return answer

        # Append formatted source section
        source_lines = "\n".join(f"  - {src}" for src in sources)
        return f"{answer}\n\n출처:\n{source_lines}"
