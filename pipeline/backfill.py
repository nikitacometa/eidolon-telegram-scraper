"""Walking chat history backwards in the background, a page at a time.

Reading two years of a busy chat is thousands of requests, so it cannot be a
task with a deadline. It is a standing intent instead: whenever the daemon has
no live message to process and the account has budget to spare, one more page
of one chat is fetched. Progress is a cursor in the database, so restarts,
deploys, and rate limits cost a page rather than an archive.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from pipeline.crawler import TelegramCrawler
from pipeline.governor import ActionStatus
from pipeline.recon_models import BackfillState, BackfillTarget
from storage.scout import ScoutDatabase

logger = logging.getLogger(__name__)

# Spacing between pages. Well below anything Telegram is known to tolerate,
# because this task has all the time in the world and the account does not.
DEFAULT_PAGE_INTERVAL_SECONDS = 8.0
# How long to wait before looking for work again when there is none.
IDLE_POLL_SECONDS = 60.0
# How far back a target reaches by default: two years of context.
DEFAULT_TARGET_DAYS = 730


class BackfillWorker:
    """Fills in chat history whenever the daemon is otherwise idle."""

    def __init__(
        self,
        *,
        scout: ScoutDatabase,
        crawler: TelegramCrawler,
        resolve_peer: Callable[[int], Awaitable[object | None]],
        is_idle: Callable[[], bool] | None = None,
        page_interval_seconds: float = DEFAULT_PAGE_INTERVAL_SECONDS,
    ) -> None:
        self._scout = scout
        self._crawler = crawler
        self._resolve_peer = resolve_peer
        self._is_idle = is_idle or (lambda: True)
        self._page_interval = page_interval_seconds

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        """Fetch pages until asked to stop."""
        logger.info("History backfill worker started")
        while not shutdown.is_set():
            delay = self._page_interval
            try:
                if self._is_idle():
                    did_work = await self.run_once()
                    if not did_work:
                        delay = IDLE_POLL_SECONDS
                else:
                    # Live traffic takes priority; come back shortly.
                    delay = self._page_interval
            except asyncio.CancelledError:
                raise
            except Exception:
                # A crawl problem must never take the monitor down with it.
                logger.exception("Backfill cycle failed")
                delay = IDLE_POLL_SECONDS

            with_timeout = asyncio.wait_for(shutdown.wait(), timeout=delay)
            try:
                await with_timeout
            except TimeoutError:
                continue
        logger.info("History backfill worker stopped")

    async def run_once(self) -> bool:
        """Fetch one page for one target. Returns whether work was done."""
        target = await self._scout.next_backfill_target()
        if target is None:
            return False

        peer = await self._resolve_peer(target.chat_id)
        if peer is None:
            # The entity is not in the client's cache yet; retry later rather
            # than spending a resolve call from inside a background task.
            await self._scout.defer_backfill_target(
                target.chat_id,
                seconds=900,
                error="peer_unresolved",
            )
            return False

        cutoff = datetime.now(UTC) - timedelta(days=target.target_days)
        result = await self._crawler.history_page(
            chat_id=target.chat_id,
            peer=peer,
            offset_id=target.oldest_message_id or 0,
            not_before=cutoff,
        )

        if result.status is ActionStatus.HALTED:
            await self._scout.defer_backfill_target(
                target.chat_id,
                seconds=6 * 60 * 60,
                error=result.error_code or "halted",
            )
            return False
        if not result.ok or result.value is None:
            await self._scout.defer_backfill_target(
                target.chat_id,
                seconds=result.retry_after_seconds or 900,
                error=result.error_code or result.status.value,
            )
            return False

        page = result.value
        stored = await self._scout.store_messages(list(page.messages))
        oldest = page.next_offset_id
        oldest_date = page.messages[-1].date if page.messages else None
        finished = _finished_state(target, page.exhausted, oldest)

        await self._scout.record_backfill_page(
            target.chat_id,
            stored=stored,
            oldest_message_id=oldest,
            oldest_message_date=oldest_date,
            finished=finished,
        )
        logger.info(
            "Backfill %s: +%d messages (page %d, oldest id %s)%s",
            target.label or target.chat_id,
            stored,
            target.pages_done + 1,
            oldest,
            f" — {finished.value}" if finished else "",
        )
        return True


def _finished_state(
    target: BackfillTarget,
    exhausted: bool,
    oldest: int | None,
) -> BackfillState | None:
    """Decide whether this target still has history worth asking for."""
    if not exhausted and oldest is not None:
        return None
    # A short page means either the chat ran out of history or the walk
    # crossed the requested horizon. Both end the target; only the reason
    # differs, and the distinction matters when raising the horizon later.
    if oldest is None:
        return BackfillState.EXHAUSTED
    return BackfillState.COMPLETE
