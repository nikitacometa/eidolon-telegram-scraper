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
        token = settings.eidolon_bot_token or settings.pantheon_bot_token
        self._url = BOT_API_URL.format(token=token)
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

        if not settings.eidolon_bot_token and not settings.pantheon_bot_token:
            logger.warning("No bot token configured, skipping alert")
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
        if not self._session or not (settings.eidolon_bot_token or settings.pantheon_bot_token):
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


    async def send_summary(
        self,
        *,
        watcher_name: str,
        summary: str,
        date_str: str,
        message_count: int,
    ) -> bool:
        """Send a daily summary, splitting into chunks if needed (Telegram 4096 char limit)."""
        if not self._session or not (settings.eidolon_bot_token or settings.pantheon_bot_token):
            return False

        header = (
            f"📋 <b>Daily Digest</b> — <code>{watcher_name}</code>\n"
            f"📅 {date_str} | {message_count} messages\n\n"
        )
        full_text = header + summary

        chunks = _split_message(full_text, max_len=4096)
        all_sent = True
        for chunk in chunks:
            try:
                async with self._session.post(
                    self._url,
                    json={
                        "chat_id": self._chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("Summary send failed %d: %s", resp.status, body)
                        all_sent = False
            except aiohttp.ClientError as e:
                logger.error("Summary send error: %s", e)
                all_sent = False

        if all_sent:
            logger.info("Summary sent: [%s] %s", watcher_name, date_str)
        return all_sent


def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split a long message into chunks that fit Telegram's limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Find last newline before limit
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


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
