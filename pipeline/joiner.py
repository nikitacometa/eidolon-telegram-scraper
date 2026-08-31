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
import time
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
# How often the reconciliation pass compares the queue's unresolved rows
# (admin-approval requests, interrupted joins) against the live dialog list.
RECONCILE_SECONDS = 60 * 60
# An unanswered invite request is re-checked at most this often: the check
# spends the invite_check budget and admins answer on human timescales.
INVITE_RECHECK_SECONDS = 6 * 60 * 60


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
        dialog_index: Callable[[], Awaitable[dict[str, int]]] | None = None,
    ) -> None:
        self._scout = scout
        self._db = db
        self._crawler = crawler
        self._resolve_entity = resolve_entity
        self._on_joined = on_joined
        self._poll_seconds = poll_seconds
        self._dialog_index = dialog_index
        self._last_reconcile = 0.0

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
                if time.monotonic() - self._last_reconcile >= RECONCILE_SECONDS:
                    self._last_reconcile = time.monotonic()
                    await self.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Join reconciliation failed")
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
                attempt=queued.attempts,
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
                attempt=queued.attempts,
            )

        if result.status in {ActionStatus.DENIED, ActionStatus.FLOOD_WAIT}:
            await self._scout.defer_queued_join(
                queued.chat_ref,
                seconds=result.retry_after_seconds or BUDGET_BACKOFF_SECONDS,
                error=result.error_code or result.status.value,
                # A budget denial never reached Telegram; a FloodWait did,
                # and that attempt is spent — the retry needs a fresh key.
                count_attempt=result.status is ActionStatus.FLOOD_WAIT,
            )
            return False
        if result.status is ActionStatus.HALTED:
            await self._scout.defer_queued_join(
                queued.chat_ref,
                seconds=6 * 60 * 60,
                error=result.error_code or "halted",
                # denial is set when the halt came from the reservation gate
                # (nothing sent); absent, the halt came out of a real call.
                count_attempt=result.denial is None,
            )
            logger.error("Join queue halted on %s", queued.chat_ref)
            return False
        if result.status is ActionStatus.REPLAYED:
            # An earlier life of the process died with this join in flight,
            # so only Telegram knows whether it went through. Settling FAILED
            # here would bury a chat we may in fact be a member of; parking
            # the row for the reconciliation pass lets the dialog list decide.
            await self._scout.defer_queued_join(
                queued.chat_ref,
                seconds=RECONCILE_SECONDS,
                error="already_attempted",
            )
            logger.warning(
                "Join for %s was interrupted mid-flight; awaiting reconciliation",
                queued.chat_ref,
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

    async def reconcile(self) -> int:
        """Resolve the joins whose real outcome only Telegram knows.

        Admin-approval requests and interrupted joins both end here. The
        dialog list is ground truth for membership: a request an admin
        approved, and a join whose process died after the call went through,
        both show up as a dialog. Returns how many rows were resolved.
        """
        rows = await self._scout.joins_awaiting_reconciliation()
        if not rows or self._dialog_index is None:
            return 0
        dialogs = await self._dialog_index()
        resolved = 0
        for queued in rows:
            secret = invite_hash(queued.chat_ref)
            if secret is None:
                chat_id = dialogs.get(queued.chat_ref.lower())
                if chat_id is not None:
                    await self._activate(queued, chat_id)
                    resolved += 1
                elif queued.last_error == "already_attempted":
                    # Positively absent from the dialogs: the interrupted
                    # call evidently never went through, so a retry under a
                    # fresh key is safe. (A retry that races a slow approval
                    # merely gets UserAlreadyParticipant, which is MEMBER.)
                    await self._scout.requeue_join_attempt(queued.chat_ref)
                    resolved += 1
                continue
            result = await self._crawler.check_invite(
                invite_hash=secret,
                chat_ref=queued.chat_ref,
                attempt=queued.attempts,
            )
            outcome = result.value if result.ok else None
            if outcome is not None and outcome.membership is ChatMembership.MEMBER:
                chat_id = (
                    await self._chat_id_of(outcome.chat) if outcome.chat is not None else None
                )
                if chat_id is not None:
                    await self._activate(queued, chat_id)
                    resolved += 1
                    continue
            if outcome is not None and outcome.membership is ChatMembership.FAILED:
                await self._scout.settle_queued_join(
                    queued.chat_ref,
                    state=JoinQueueState.FAILED,
                    error=outcome.error_code or "invite_invalid",
                )
                resolved += 1
                continue
            # Still unanswered; ask again no sooner than the recheck window,
            # and spend the attempt so the next check gets its own key.
            await self._scout.defer_queued_join(
                queued.chat_ref,
                seconds=INVITE_RECHECK_SECONDS,
                error=queued.last_error,
                count_attempt=result.ok,
            )
        if resolved:
            logger.info("Join reconciliation resolved %d row(s)", resolved)
        return resolved

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
