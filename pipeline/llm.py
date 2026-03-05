"""LLM classifier (Level 2+) — GPT-4.1-mini relevance classification."""

from __future__ import annotations

import logging
from enum import Enum

from openai import AsyncOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a message classifier. Classify the message into exactly one category:\n"
    "- OFFER: someone is offering/advertising something (rental, sale, service, event)\n"
    "- SEEK: someone is looking for/requesting something\n"
    "- IRRELEVANT: off-topic, chitchat, questions, memes, unrelated\n\n"
    "Reply with ONLY the category name: OFFER, SEEK, or IRRELEVANT."
)


class Verdict(str, Enum):
    OFFER = "OFFER"
    SEEK = "SEEK"
    IRRELEVANT = "IRRELEVANT"


class LLMClassifier:
    """Classifies messages using OpenAI GPT-4.1-mini.

    Fail-open: on timeout or error, returns OFFER (better a false alert than missed one).
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    async def start(self) -> None:
        if settings.openai_api_key:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            logger.warning("OPENAI_API_KEY not set, LLM classifier disabled")

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def classify(self, text: str, watcher_prompt: str = "") -> Verdict:
        """Classify a message text. Returns Verdict enum.

        Args:
            text: Message text (truncated to 1000 chars internally).
            watcher_prompt: Optional watcher-specific context for the LLM.
        """
        if not self._client:
            return Verdict.OFFER  # fail-open

        user_content = f"{watcher_prompt}\n\n{text[:1000]}" if watcher_prompt else text[:1000]

        try:
            response = await self._client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=10,
                temperature=0,
                timeout=settings.llm_timeout_seconds,
            )
            raw = response.choices[0].message.content.strip().upper()
            try:
                return Verdict(raw)
            except ValueError:
                logger.warning("LLM returned unexpected verdict: %r, defaulting to OFFER", raw)
                return Verdict.OFFER

        except Exception as e:
            logger.error("LLM classification failed (fail-open → OFFER): %s", e)
            return Verdict.OFFER
