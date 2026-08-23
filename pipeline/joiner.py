"""Joining queued chats one at a time, from inside the daemon.

Joining is the only reconnaissance action other people can see, and the
account's standing is spent per attempt. So it is a queue worked at the pace
the budget allows rather than a batch: a list of ten chats becomes ten hours
of quiet, resumable work instead of ten joins in one minute.

The worker lives in the daemon because the daemon owns the Telegram session.
An external scheduler would have to stop it for every single join.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from pipeline.crawler import TelegramCrawler
from pipeline.governor import ActionStatus
from pipeline.models import ObservationMode, ObservationSource
from pipeline.recon_models import ChatMembership, JoinQueueState, QueuedJoin, invite_hash
from storage.db import Database
from storage.scout import ScoutDatabase

logger = logging.getLogger(__name__)

# How often to look at the queue. The budget decides whether anything
# actually happens; this only decides how promptly the worker notices.
DEFAULT_POLL_SECONDS = 300.0
# A refused join is usually the hourly budget, so wait out roughly that long.
BUDGET_BACKOFF_SECONDS = 20 * 60


class JoinWorker:
    """Works the join queue at whatever pace the budget permits."""

    def __init__(
        self,
        *,
        scout: ScoutDatabase,
        db: Database,
        crawler: TelegramCrawler,
        resolve_entity: Callable[[str], Awaitable[object | None]],
        on_joined: Callable[[], Awaitable[None]] | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self._scout = scout
        self._db = db
        self._crawler = crawler
        self._resolve_entity = resolve_entity
        self._on_joined = on_joined
        self._poll_seconds = poll_seconds

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        """Work the queue until asked to stop."""
        logger.info("Join queue worker started")
        while not shutdown.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A join problem must never take the monitor down with it.
                logger.exception("Join queue cycle failed")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                continue
        logger.info("Join queue worker stopped")

    async def run_once(self) -> bool:
        """Attempt at most one join. Returns whether an attempt was made."""
        queued = await self._scout.next_queued_join()
        if queued is None:
            return False

        # An invite link resolves to nothing until it is used, so the entity
        # arrives with the join result instead of before it.
        entity: object | None = None
        secret = invite_hash(queued.chat_ref)
        if secret is not None:
            result = await self._crawler.join_invite(
                invite_hash=secret,
                chat_ref=queued.chat_ref,
            )
        else:
            entity = await self._resolve_entity(queued.chat_ref)
            if entity is None:
                await self._scout.settle_queued_join(
                    queued.chat_ref,
                    state=JoinQueueState.FAILED,
                    error="unresolvable",
                )
                logger.warning("Cannot resolve %s; dropped from the join queue", queued.chat_ref)
                return False
            result = await self._crawler.join(
                channel=entity,
                chat_ref=queued.chat_ref,
            )

        if result.status in {ActionStatus.DENIED, ActionStatus.FLOOD_WAIT}:
            await self._scout.defer_queued_join(
                queued.chat_ref,
                seconds=result.retry_after_seconds or BUDGET_BACKOFF_SECONDS,
                error=result.error_code or result.status.value,
            )
            return False
        if result.status is ActionStatus.HALTED:
            await self._scout.defer_queued_join(
                queued.chat_ref,
                seconds=6 * 60 * 60,
                error=result.error_code or "halted",
            )
            logger.error("Join queue halted on %s", queued.chat_ref)
            return False
        if result.status is ActionStatus.REPLAYED:
            # Already attempted in an earlier life of the process; the real
            # membership has to be checked by hand rather than guessed at.
            await self._scout.settle_queued_join(
                queued.chat_ref,
                state=JoinQueueState.FAILED,
                error="already_attempted",
            )
            return False

        outcome = result.value
        membership = outcome.membership if outcome else ChatMembership.FAILED
        if membership is ChatMembership.REQUESTED:
            await self._scout.settle_queued_join(
                queued.chat_ref,
                state=JoinQueueState.REQUESTED,
                error="awaiting_admin_approval",
            )
            logger.info("Join request for %s awaits an admin", queued.chat_ref)
            return True
        if membership is not ChatMembership.MEMBER:
            await self._scout.settle_queued_join(
                queued.chat_ref,
                state=JoinQueueState.FAILED,
                error=(outcome.error_code if outcome else None)
                or result.error_code
                or "join_failed",
            )
            return True

        if entity is None and outcome is not None:
            entity = outcome.chat
        chat_id = await self._chat_id_of(entity) if entity is not None else None
        if chat_id is None:
            await self._scout.settle_queued_join(
                queued.chat_ref,
                state=JoinQueueState.FAILED,
                error="joined_but_unidentified",
            )
            return True

        await self._activate(queued, chat_id)
        return True

    async def _activate(self, queued: QueuedJoin, chat_id: int) -> None:
        """Turn a fresh membership into monitoring and an archive target."""
        await self._db.observe_chat(
            chat_id=chat_id,
            mode=ObservationMode.MONITOR if queued.watcher_name else ObservationMode.RECON,
            source=ObservationSource.RECON,
            title=queued.label,
        )
        if queued.watcher_name:
            await self._db.bind_policy(chat_id=chat_id, watcher_name=queued.watcher_name)
        await self._scout.add_backfill_target(
            chat_id=chat_id,
            label=queued.label or queued.chat_ref,
            target_days=queued.target_days,
        )
        await self._scout.settle_queued_join(
            queued.chat_ref,
            state=JoinQueueState.JOINED,
            joined_chat_id=chat_id,
        )
        if self._on_joined is not None:
            # Routing is built from a snapshot, so it has to be rebuilt before
            # the next message from this chat arrives.
            await self._on_joined()
        logger.info(
            "Joined %s (chat_id=%s); monitoring=%s, archive target set",
            queued.chat_ref,
            chat_id,
            queued.watcher_name or "none",
        )

    async def _chat_id_of(self, entity: object) -> int | None:
        """Derive the -100… chat id used everywhere else from an entity."""
        from telethon import utils

        try:
            return int(utils.get_peer_id(entity))
        except (TypeError, ValueError):
            return None
