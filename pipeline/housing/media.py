"""Fetching the photographs of an advertisement, under the action governor.

Nothing is downloaded because a message has a picture. A download happens only
after the text has been read, the advertisement has been judged worth telling
the owner about, and a criterion is still unanswered that a photograph could
answer. That ordering is what keeps the volume proportional to listings rather
than to traffic.

Every fetch goes through the same governor the crawl uses, on its own action
class, so a burst of listings cannot spend the budget that history walking
needs and neither can starve the other.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from pipeline.governor import ActionStatus, TelegramActionGovernor
from pipeline.recon_models import ActionKind
from storage.housing import HousingStore

logger = logging.getLogger(__name__)

POLL_SECONDS = 5.0
IDLE_POLL_SECONDS = 60.0
# Beyond this a photograph is not worth more of the account's standing.
MAX_ATTEMPTS = 4
# Telegram's own ceiling is far higher; this is about not filling a small VPS
# disk with holiday snapshots.
MAX_BYTES = 12 * 1024 * 1024


class MediaDownloadWorker:
    """Fetches queued photographs one at a time, oldest request first."""

    def __init__(
        self,
        *,
        store: HousingStore,
        client: Any,
        governor: TelegramActionGovernor,
        media_root: Path,
        priority: str = "live",
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self._store = store
        self._client = client
        self._governor = governor
        self._root = media_root
        self._priority = priority
        self._poll_seconds = poll_seconds
        self._kind = (
            ActionKind.MEDIA_DOWNLOAD_LIVE
            if priority == "live"
            else ActionKind.MEDIA_DOWNLOAD_BACKFILL
        )

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        """Fetch photographs until asked to stop."""
        logger.info("Housing media download worker started (%s)", self._priority)
        while not shutdown.is_set():
            delay = self._poll_seconds
            try:
                if not await self.run_once():
                    delay = IDLE_POLL_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Media download cycle failed")
                delay = IDLE_POLL_SECONDS
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
            except TimeoutError:
                continue
        logger.info("Housing media download worker stopped (%s)", self._priority)

    async def run_once(self) -> bool:
        """Fetch at most one photograph. Returns whether work was done."""
        pending = await self._store.next_media_download(priority=self._priority)
        if pending is None:
            return False

        unit_key = str(pending["unit_key"])
        chat_id = int(pending["chat_id"])
        telegram_msg_id = int(pending["telegram_msg_id"])
        attempts = int(pending["attempts"])

        result = await self._governor.run(
            self._kind,
            f"media:{chat_id}:{telegram_msg_id}:{attempts}",
            lambda: self._download(chat_id, telegram_msg_id),
        )

        if result.status in {ActionStatus.DENIED, ActionStatus.HALTED, ActionStatus.FLOOD_WAIT}:
            # The budget or a flood wait, not the photograph's fault: come back
            # later without spending an attempt on the message itself.
            await self._store.settle_media(
                unit_key=unit_key,
                telegram_msg_id=telegram_msg_id,
                status="pending",
                error=result.error_code or result.status.value,
                retry_in_seconds=result.retry_after_seconds or 900,
            )
            return False

        if not result.ok:
            terminal = attempts + 1 >= MAX_ATTEMPTS
            await self._store.settle_media(
                unit_key=unit_key,
                telegram_msg_id=telegram_msg_id,
                status="failed" if terminal else "pending",
                error=result.error_code or "download_failed",
                retry_in_seconds=None if terminal else 600,
            )
            return True

        payload = result.value
        if payload is None:
            # The message is gone: deleted, or the chat is no longer readable.
            # Retrying cannot bring it back, so this is terminal and quiet.
            await self._store.settle_media(
                unit_key=unit_key,
                telegram_msg_id=telegram_msg_id,
                status="failed_gone",
                error="message_unavailable",
            )
            return True

        path, size = payload
        await self._store.settle_media(
            unit_key=unit_key,
            telegram_msg_id=telegram_msg_id,
            status="downloaded",
            local_path=str(path),
            byte_size=size,
        )
        logger.info("Downloaded photo for %s (%d bytes)", unit_key, size)
        return True

    async def _download(self, chat_id: int, telegram_msg_id: int) -> tuple[Path, int] | None:
        """Re-fetch the message, then save its photograph.

        The message is fetched again rather than the earlier object reused:
        Telegram's file references expire, and a stale one fails in a way that
        looks like a missing file rather than like the stale reference it is.
        """
        messages = await self._client.get_messages(chat_id, ids=telegram_msg_id)
        message = messages[0] if isinstance(messages, list) else messages
        if message is None or not getattr(message, "media", None):
            return None

        directory = self._root / str(abs(chat_id))
        target = directory / f"{telegram_msg_id}.jpg"
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        saved = await self._client.download_media(message, file=str(target))
        if saved is None:
            return None
        return await asyncio.to_thread(_measure, Path(str(saved)))


def _measure(path: Path) -> tuple[Path, int]:
    """Size a saved file, refusing one too large to be worth keeping.

    Filesystem calls block, so this runs off the event loop; the daemon's
    single thread is also serving live Telegram updates.
    """
    size = path.stat().st_size if path.exists() else 0
    if size > MAX_BYTES:
        path.unlink(missing_ok=True)
        raise ValueError(f"photograph exceeds {MAX_BYTES} bytes")
    return path, size
