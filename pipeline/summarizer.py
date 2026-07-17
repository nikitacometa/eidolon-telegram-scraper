"""Daily summarizer — generates evening digests of monitored chats."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import date

from openai import AsyncOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a concise digest writer. Summarize the day's messages from Telegram group chats.\n"
    "Focus on: rental offers, events, invitations, local news, important announcements.\n"
    "Skip: greetings, casual chat, memes, off-topic.\n"
    "Format: bullet points grouped by topic. Use original language of messages.\n"
    "If nothing noteworthy happened, say so in one sentence."
)

MAX_CONTEXT_CHARS = 12000


class DailySummarizer:
    """Generates daily summaries using OpenAI GPT-4.1-mini."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    async def start(self) -> None:
        if settings.openai_api_key:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            logger.warning("OPENAI_API_KEY not set, summarizer disabled")

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def summarize(
        self,
        messages: Sequence[Mapping[str, object]],
        watcher_name: str,
        target_date: date | None = None,
    ) -> str | None:
        """Generate a summary from a list of message dicts.

        Args:
            messages: List of dicts with chat_title, sender_name, text, date.
            watcher_name: Name of the watcher for context.
            target_date: Date being summarized (for the header).

        Returns:
            Summary text, or None if no client or no messages.
        """
        if not self._client:
            return None
        if not messages:
            return None

        date_str = (target_date or date.today()).isoformat()
        transcript = _build_transcript(messages)

        try:
            response = await self._client.chat.completions.create(
                model=settings.summary_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Watcher: {watcher_name}\n"
                            f"Date: {date_str}\n"
                            f"Messages ({len(messages)} total):\n\n{transcript}"
                        ),
                    },
                ],
                max_tokens=1000,
                temperature=0.3,
                timeout=30,
            )
            content = response.choices[0].message.content
            return content.strip() if content else None

        except Exception as error:
            logger.error("Summary generation failed: %s", type(error).__name__)
            return None


def _build_transcript(messages: Sequence[Mapping[str, object]]) -> str:
    """Build a text transcript from message dicts, truncated to MAX_CONTEXT_CHARS."""
    lines: list[str] = []
    total = 0
    for msg in messages:
        chat_title = str(msg.get("chat_title") or "Unknown")
        sender_name = str(msg.get("sender_name") or "Unknown")
        text = str(msg.get("text") or "")
        line = f"[{chat_title}] {sender_name}: {text}"
        if total + len(line) > MAX_CONTEXT_CHARS:
            lines.append(f"... ({len(messages) - len(lines)} more messages truncated)")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)
