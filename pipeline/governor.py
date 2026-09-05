"""The single gate through which reconnaissance touches Telegram.

Every crawl call goes through :class:`TelegramActionGovernor`, so budgets,
cooldowns, and throttling live in one place instead of being spread across
each caller. Live monitoring does not pass through here: an inbound update
costs no request, and the daemon must keep alerting even while the crawl is
halted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from telethon import errors

from pipeline.recon_models import (
    ActionKind,
    ActionOutcome,
    ActionReservation,
    BudgetDenial,
    BudgetRule,
)
from storage.scout import ScoutDatabase

logger = logging.getLogger(__name__)

# A FloodWait this long stops the whole crawl rather than one action class:
# repeated waits at this scale mean Telegram is pushing back on the account,
# not on a single endpoint.
CRAWL_HALT_FLOOD_SECONDS = 15 * 60
# Beyond this, an operator has to look at the account before it resumes.
MANUAL_HALT_FLOOD_SECONDS = 6 * 60 * 60
# Extra margin on top of the wait Telegram asks for, so the next attempt is
# not racing the exact expiry instant.
FLOOD_COOLDOWN_MARGIN_SECONDS = 60
# A call that takes this long without raising almost certainly slept through a
# short FloodWait inside the library, which is otherwise invisible.
HIDDEN_FLOOD_SUSPICION_MS = 5_000.0


class ActionStatus(StrEnum):
    """How a governed call ended, from the caller's point of view."""

    OK = "ok"
    DENIED = "denied"
    REPLAYED = "replayed"
    FLOOD_WAIT = "flood_wait"
    HALTED = "halted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ActionResult[T]:
    """Outcome of one governed call.

    Expected refusals are values rather than exceptions: a crawl spends most
    of its life being told to wait, and that is normal control flow. Anything
    unexpected still propagates.
    """

    status: ActionStatus
    value: T | None = None
    denial: BudgetDenial | None = None
    flood_wait_seconds: int | None = None
    error_code: str | None = None
    duration_ms: float | None = None

    @property
    def ok(self) -> bool:
        """Whether the call ran and returned a value."""
        return self.status is ActionStatus.OK

    @property
    def retry_after_seconds(self) -> int | None:
        """How long the caller should wait before trying this action again."""
        if self.flood_wait_seconds is not None:
            return self.flood_wait_seconds + FLOOD_COOLDOWN_MARGIN_SECONDS
        if self.denial is not None:
            return self.denial.retry_after_seconds
        return None


class TelegramActionGovernor:
    """Serializes, budgets, and throttles reconnaissance calls."""

    def __init__(
        self,
        *,
        scout: ScoutDatabase,
        account_id: str = "owner-primary",
        policy: dict[ActionKind, BudgetRule] | None = None,
    ) -> None:
        self._scout = scout
        self._account_id = account_id
        self._policy = policy
        # One crawl request in flight at a time. Concurrency here buys nothing
        # (the account is the bottleneck) and costs the ability to stop cleanly.
        self._slot = asyncio.Semaphore(1)

    @property
    def account_id(self) -> str:
        """Account these budgets belong to."""
        return self._account_id

    async def run[T](
        self,
        kind: ActionKind,
        idempotency_key: str,
        call: Callable[[], Awaitable[T]],
        *,
        job_id: str | None = None,
        candidate_id: int | None = None,
    ) -> ActionResult[T]:
        """Reserve a slot, run one Telegram call, and record what happened."""
        reservation = await self._scout.reserve_action(
            account_id=self._account_id,
            kind=kind,
            idempotency_key=idempotency_key,
            job_id=job_id,
            candidate_id=candidate_id,
            policy=self._policy,
        )
        if isinstance(reservation, BudgetDenial):
            logger.info(
                "Action refused: kind=%s scope=%s used=%s cap=%s",
                kind.value,
                reservation.scope.value,
                reservation.used,
                reservation.cap,
            )
            status = (
                ActionStatus.HALTED
                if reservation.retry_after_seconds is None
                else ActionStatus.DENIED
            )
            return ActionResult(status=status, denial=reservation)

        if reservation.replayed and kind.is_mutating:
            # The first attempt already reached Telegram. Running it again is
            # exactly the blind retry that turns one ambiguous join into two,
            # so the caller has to reconcile the real state instead.
            logger.warning(
                "Refusing to repeat a %s already attempted under key %s",
                kind.value,
                idempotency_key,
            )
            return ActionResult(status=ActionStatus.REPLAYED, error_code="already_attempted")

        async with self._slot:
            return await self._execute(kind, reservation, call)

    async def _execute[T](
        self,
        kind: ActionKind,
        reservation: ActionReservation,
        call: Callable[[], Awaitable[T]],
    ) -> ActionResult[T]:
        started = time.monotonic()
        try:
            value = await call()
        except errors.FloodWaitError as error:
            duration_ms = (time.monotonic() - started) * 1000
            return await self._handle_flood_wait(kind, reservation, int(error.seconds), duration_ms)
        except (errors.PeerFloodError, errors.UserBannedInChannelError) as error:
            duration_ms = (time.monotonic() - started) * 1000
            return await self._handle_spam_limit(kind, reservation, error, duration_ms)
        except errors.ChannelsTooMuchError:
            duration_ms = (time.monotonic() - started) * 1000
            await self._settle(reservation, ActionOutcome.FAILED, duration_ms, "channels_too_much")
            await self._scout.set_cooldown(
                account_id=self._account_id,
                scope=ActionKind.JOIN.value,
                seconds=MANUAL_HALT_FLOOD_SECONDS,
                reason="channels_too_much",
                manual_resume_required=True,
            )
            logger.error("Account is at its channel limit; joins halted until capacity is freed")
            return ActionResult(
                status=ActionStatus.HALTED,
                error_code="channels_too_much",
                duration_ms=duration_ms,
            )
        except (
            errors.UsernameInvalidError,
            errors.UsernameNotOccupiedError,
            errors.InviteHashInvalidError,
            errors.InviteHashExpiredError,
            errors.ChannelInvalidError,
            errors.ChannelPrivateError,
            errors.ChatAdminRequiredError,
            errors.PremiumAccountRequiredError,
            # Owner delivery: the source forbids forwarding, the message is
            # gone, or the owner cannot be written to. Each is final for this
            # call and says nothing about the account's standing, so it
            # settles as a rejection the caller can act on rather than as an
            # ambiguous attempt that propagates and kills the delivery loop.
            errors.ChatForwardsRestrictedError,
            errors.MessageIdInvalidError,
            errors.MediaEmptyError,
            errors.UserIsBlockedError,
            errors.UserPrivacyRestrictedError,
            errors.PeerIdInvalidError,
            errors.InputUserDeactivatedError,
            errors.ChatWriteForbiddenError,
        ) as error:
            duration_ms = (time.monotonic() - started) * 1000
            code = type(error).__name__
            await self._settle(reservation, ActionOutcome.FAILED, duration_ms, code)
            return ActionResult(
                status=ActionStatus.REJECTED,
                error_code=code,
                duration_ms=duration_ms,
            )
        except BaseException:
            # The attempt reached Telegram in an unknown state. Recording it as
            # ambiguous keeps the slot spent, which is what stops a crash from
            # turning one join into two.
            duration_ms = (time.monotonic() - started) * 1000
            await self._settle(reservation, ActionOutcome.AMBIGUOUS, duration_ms, "unexpected")
            raise

        duration_ms = (time.monotonic() - started) * 1000
        await self._settle(reservation, ActionOutcome.SUCCEEDED, duration_ms, None)
        if duration_ms >= HIDDEN_FLOOD_SUSPICION_MS:
            # Telethon sleeps through waits below flood_sleep_threshold inside
            # the call, so an unexplained slow call is the only trace we get.
            logger.warning(
                "Slow Telegram call may hide a short FloodWait: kind=%s duration_ms=%.0f",
                kind.value,
                duration_ms,
            )
        return ActionResult(status=ActionStatus.OK, value=value, duration_ms=duration_ms)

    async def _handle_flood_wait[T](
        self,
        kind: ActionKind,
        reservation: ActionReservation,
        seconds: int,
        duration_ms: float,
    ) -> ActionResult[T]:
        await self._scout.settle_action(
            reservation.action_id,
            outcome=ActionOutcome.FLOOD_WAIT,
            duration_ms=duration_ms,
            flood_wait_seconds=seconds,
            error_code="flood_wait",
        )

        if seconds >= MANUAL_HALT_FLOOD_SECONDS:
            await self._scout.set_cooldown(
                account_id=self._account_id,
                scope="all",
                seconds=seconds + FLOOD_COOLDOWN_MARGIN_SECONDS,
                reason=f"flood_wait_{seconds}s",
                manual_resume_required=True,
            )
            logger.error("FloodWait of %ds halted the crawl; manual resume required", seconds)
            status = ActionStatus.HALTED
        elif seconds >= CRAWL_HALT_FLOOD_SECONDS:
            await self._scout.set_cooldown(
                account_id=self._account_id,
                scope="all",
                seconds=seconds + FLOOD_COOLDOWN_MARGIN_SECONDS,
                reason=f"flood_wait_{seconds}s",
            )
            logger.warning("FloodWait of %ds paused all crawl actions", seconds)
            status = ActionStatus.FLOOD_WAIT
        else:
            await self._scout.set_cooldown(
                account_id=self._account_id,
                scope=kind.value,
                seconds=seconds + FLOOD_COOLDOWN_MARGIN_SECONDS,
                reason=f"flood_wait_{seconds}s",
            )
            logger.info("FloodWait of %ds paused %s", seconds, kind.value)
            status = ActionStatus.FLOOD_WAIT

        return ActionResult(
            status=status,
            flood_wait_seconds=seconds,
            error_code="flood_wait",
            duration_ms=duration_ms,
        )

    async def _handle_spam_limit[T](
        self,
        kind: ActionKind,
        reservation: ActionReservation,
        error: BaseException,
        duration_ms: float,
    ) -> ActionResult[T]:
        code = type(error).__name__
        await self._settle(reservation, ActionOutcome.FAILED, duration_ms, code)
        # A spam limitation is about the account, not the endpoint, and it
        # escalates on repetition. Nothing automatic should touch Telegram
        # again until a human has looked at @SpamBot.
        await self._scout.set_cooldown(
            account_id=self._account_id,
            scope="all",
            seconds=MANUAL_HALT_FLOOD_SECONDS,
            reason=code,
            manual_resume_required=True,
        )
        logger.error("Spam limitation on %s; crawl halted until manually resumed", kind.value)
        return ActionResult(status=ActionStatus.HALTED, error_code=code, duration_ms=duration_ms)

    async def _settle(
        self,
        reservation: ActionReservation,
        outcome: ActionOutcome,
        duration_ms: float,
        error_code: str | None,
    ) -> None:
        await self._scout.settle_action(
            reservation.action_id,
            outcome=outcome,
            duration_ms=duration_ms,
            error_code=error_code,
        )
