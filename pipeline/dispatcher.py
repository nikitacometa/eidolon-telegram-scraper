"""Alert dispatcher — sends filtered messages via Telegram bot."""

from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime

import aiohttp

from config.settings import settings
from pipeline.models import DeliveryResult

logger = logging.getLogger(__name__)

BOT_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class AlertDispatcher:
    """Sends escaped alerts through a configured Telegram bot."""

    def __init__(
        self,
        *,
        token: str | None = None,
        chat_id: int | None = None,
        max_attempts: int = 3,
    ) -> None:
        resolved_token = (
            token
            if token is not None
            else settings.eidolon_bot_token or settings.pantheon_bot_token
        )
        self._url = BOT_API_URL.format(token=resolved_token)
        self._chat_id = chat_id if chat_id is not None else settings.pantheon_chat_id
        self._enabled = bool(resolved_token and self._chat_id)
        self._max_attempts = max(1, max_attempts)
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
        """Compatibility wrapper returning only whether delivery succeeded."""
        return (
            await self.deliver_alert(
                watcher_name=watcher_name,
                chat_title=chat_title,
                sender_name=sender_name,
                text=text,
                matched_keyword=matched_keyword,
                filter_level=filter_level,
            )
        ).sent

    async def deliver_alert(
        self,
        *,
        watcher_name: str,
        chat_title: str,
        sender_name: str,
        text: str,
        matched_keyword: str | None = None,
        filter_level: int = 1,
    ) -> DeliveryResult:
        """Deliver an alert and preserve retry semantics for a durable outbox."""
        if not self._session:
            logger.error("Dispatcher not started. Call start() first.")
            return DeliveryResult(
                sent=False,
                retryable=True,
                error_code="dispatcher_not_started",
                retry_after=1,
            )

        if not self._enabled:
            logger.warning("No bot token configured, skipping alert")
            return DeliveryResult(
                sent=False,
                retryable=False,
                error_code="dispatcher_disabled",
            )

        message = _format_alert(
            watcher_name=watcher_name,
            chat_title=chat_title,
            sender_name=sender_name,
            text=text,
            matched_keyword=matched_keyword,
            filter_level=filter_level,
        )

        result = await self._deliver_html(message)
        if result.sent:
            logger.info("Alert sent for watcher=%s", watcher_name)
        else:
            logger.warning(
                "Alert delivery failed: watcher=%s code=%s retryable=%s",
                watcher_name,
                result.error_code,
                result.retryable,
            )
        return result

    async def send_echo(
        self,
        *,
        chat_title: str,
        sender_name: str,
        text: str,
    ) -> None:
        """Forward a message as-is for debug purposes (echo mode)."""
        if not self._session or not self._enabled:
            return

        await self._send_html(
            _format_echo(
                chat_title=chat_title,
                sender_name=sender_name,
                text=text,
            )
        )

    async def send_summary(
        self,
        *,
        watcher_name: str,
        summary: str,
        date_str: str,
        message_count: int,
    ) -> bool:
        """Send a daily summary, splitting into chunks if needed (Telegram 4096 char limit)."""
        if not self._session or not self._enabled:
            return False

        header = (
            f"📋 <b>Daily Digest</b> — <code>{html.escape(watcher_name)}</code>\n"
            f"📅 {html.escape(date_str)} | {message_count} messages\n\n"
        )

        escaped_chunks = _escape_and_split(summary, max_len=3800)
        chunks = [header + escaped_chunks[0], *escaped_chunks[1:]]
        all_sent = True
        for chunk in chunks:
            if not await self._send_html(chunk):
                all_sent = False

        if all_sent:
            logger.info("Summary sent: [%s] %s", watcher_name, date_str)
        return all_sent

    async def _send_html(self, message: str) -> bool:
        """Compatibility wrapper for echo and summary delivery."""
        return (await self._deliver_html(message)).sent

    async def _deliver_html(self, message: str) -> DeliveryResult:
        """Send one safe HTML message with bounded in-process retry."""
        if not self._session or not self._enabled:
            if not self._session:
                return DeliveryResult(
                    sent=False,
                    retryable=True,
                    error_code="dispatcher_not_started",
                    retry_after=1,
                )
            return DeliveryResult(
                sent=False,
                retryable=False,
                error_code="dispatcher_disabled",
            )

        for attempt in range(1, self._max_attempts + 1):
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
                ) as response:
                    if response.status == 200:
                        return DeliveryResult.success()

                    retry_after = await _retry_after_seconds(response, attempt)
                    is_transient = response.status == 429 or response.status >= 500
                    if not is_transient:
                        return DeliveryResult(
                            sent=False,
                            retryable=False,
                            error_code=f"telegram_http_{response.status}",
                        )
                    if attempt == self._max_attempts:
                        error_code = (
                            "telegram_rate_limited"
                            if response.status == 429
                            else "telegram_server_error"
                        )
                        return DeliveryResult(
                            sent=False,
                            retryable=True,
                            error_code=error_code,
                            retry_after=retry_after,
                        )
            except TimeoutError:
                if attempt == self._max_attempts:
                    return DeliveryResult(
                        sent=False,
                        retryable=True,
                        error_code="telegram_timeout",
                        retry_after=min(2 ** (attempt - 1), 8),
                    )
                retry_after = min(2 ** (attempt - 1), 8)
            except aiohttp.ClientError:
                if attempt == self._max_attempts:
                    return DeliveryResult(
                        sent=False,
                        retryable=True,
                        error_code="telegram_network_error",
                        retry_after=min(2 ** (attempt - 1), 8),
                    )
                retry_after = min(2 ** (attempt - 1), 8)

            await asyncio.sleep(retry_after)

        return DeliveryResult(
            sent=False,
            retryable=True,
            error_code="telegram_delivery_exhausted",
            retry_after=8,
        )


async def _retry_after_seconds(
    response: aiohttp.ClientResponse,
    attempt: int,
) -> int:
    """Read Telegram's retry hint without logging a potentially sensitive body."""
    fallback: int = min(2 ** (attempt - 1), 8)
    if response.status != 429:
        return fallback
    try:
        payload: object = await response.json(content_type=None)
        if not isinstance(payload, dict):
            return fallback
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            return fallback
        value: object = parameters.get("retry_after", fallback)
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return fallback
        parsed = int(value)
        return max(1, min(parsed, 60))
    except (aiohttp.ClientError, TypeError, ValueError):
        return fallback


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


def _escape_and_split(text: str, max_len: int) -> list[str]:
    """Escape untrusted text and split without cutting an HTML entity."""
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for character in text:
        escaped = html.escape(character, quote=False)
        if current and current_length + len(escaped) > max_len:
            chunks.append("".join(current))
            current = []
            current_length = 0
        current.append(escaped)
        current_length += len(escaped)

    chunks.append("".join(current))
    return chunks


def _format_echo(*, chat_title: str, sender_name: str, text: str) -> str:
    """Format an escaped debug echo."""
    display_text = text[:300] + "..." if len(text) > 300 else text
    return (
        f"🔊 <b>{html.escape(chat_title)}</b> → {html.escape(sender_name)}\n\n"
        f"{html.escape(display_text, quote=False)}"
    )


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
    safe_watcher = html.escape(watcher_name)
    safe_chat = html.escape(chat_title)
    safe_sender = html.escape(sender_name)
    keyword_line = f"\n🔑 <b>Keyword:</b> {html.escape(matched_keyword)}" if matched_keyword else ""
    # Truncate long messages
    display_text = text[:500] + "..." if len(text) > 500 else text
    safe_text = html.escape(display_text, quote=False)

    return (
        f"👁 <b>Eidolon Alert</b> — <code>{safe_watcher}</code>\n"
        f"⏰ {now} | L{filter_level}\n"
        f"💬 <b>{safe_chat}</b> → {safe_sender}\n"
        f"{keyword_line}\n\n"
        f"{safe_text}"
    )
