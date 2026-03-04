"""Alert dispatcher — sends filtered messages via Telegram bot."""

from __future__ import annotations

import logging
from datetime import datetime

import aiohttp

from config.settings import settings

logger = logging.getLogger(__name__)

BOT_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class AlertDispatcher:
    """Sends alerts through @ClaudePantheon_Bot to the configured chat."""

    def __init__(self) -> None:
        self._url = BOT_API_URL.format(token=settings.pantheon_bot_token)
        self._chat_id = settings.pantheon_chat_id
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Initialize the HTTP session."""
        self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def send_alert(
        self,
        *,
        watcher_name: str,
        chat_title: str,
        sender_name: str,
        text: str,
        matched_keyword: str | None = None,
        filter_level: int = 1,
    ) -> bool:
        """Send an alert message via the Pantheon bot.

        Returns True if sent successfully, False otherwise.
        """
        if not self._session:
            logger.error("Dispatcher not started. Call start() first.")
            return False

        if not settings.pantheon_bot_token:
            logger.warning("PANTHEON_BOT_TOKEN not set, skipping alert")
            return False

        message = _format_alert(
            watcher_name=watcher_name,
            chat_title=chat_title,
            sender_name=sender_name,
            text=text,
            matched_keyword=matched_keyword,
            filter_level=filter_level,
        )

        try:
            async with self._session.post(
                self._url,
                json={
                    "chat_id": self._chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    logger.info("Alert sent: [%s] %s", watcher_name, text[:50])
                    return True
                body = await resp.text()
                logger.error("Bot API error %d: %s", resp.status, body)
                return False
        except aiohttp.ClientError as e:
            logger.error("Failed to send alert: %s", e)
            return False

    async def send_echo(
        self,
        *,
        chat_title: str,
        sender_name: str,
        text: str,
    ) -> None:
        """Forward a message as-is for debug purposes (echo mode)."""
        if not self._session or not settings.pantheon_bot_token:
            return

        display_text = text[:300] + "..." if len(text) > 300 else text
        message = f"🔊 <b>{chat_title}</b> → {sender_name}\n\n{display_text}"

        try:
            async with self._session.post(
                self._url,
                json={
                    "chat_id": self._chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Echo send failed %d: %s", resp.status, body)
        except aiohttp.ClientError as e:
            logger.warning("Echo send error: %s", e)


def _format_alert(
    *,
    watcher_name: str,
    chat_title: str,
    sender_name: str,
    text: str,
    matched_keyword: str | None,
    filter_level: int,
) -> str:
    """Format an alert message for Telegram."""
    now = datetime.now().strftime("%H:%M")
    keyword_line = f"\n🔑 <b>Keyword:</b> {matched_keyword}" if matched_keyword else ""
    # Truncate long messages
    display_text = text[:500] + "..." if len(text) > 500 else text

    return (
        f"👁 <b>Eidolon Alert</b> — <code>{watcher_name}</code>\n"
        f"⏰ {now} | L{filter_level}\n"
        f"💬 <b>{chat_title}</b> → {sender_name}\n"
        f"{keyword_line}\n\n"
        f"{display_text}"
    )
