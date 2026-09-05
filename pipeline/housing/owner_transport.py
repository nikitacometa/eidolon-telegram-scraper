"""Delivering housing alerts to the owner through the monitoring account itself.

The bot that delivers every other alert is not a member of the chats where
advertisements are posted, and the owner's own account is not either. So a
report with a t.me link is a report with a dead link: Telegram refuses to open
a message in a chat the reader has not joined, and the bot cannot forward from
a chat it is not in. The one party that can put the original in front of the
owner is the account that read it. This module is that account talking to the
owner: a composed report, then the original forwarded as-is — full text, every
photograph, the author's name in the forward header.

Every call goes through the action governor under its own two kinds, so a
burst of alerts is paced like a person forwarding by hand, a FloodWait pauses
this class alone, and the crawl's budgets are never spent on delivery.
"""

from __future__ import annotations

import asyncio
import html
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from telethon import errors

from pipeline.governor import ActionResult, ActionStatus, TelegramActionGovernor
from pipeline.recon_models import ActionKind, BudgetScope

logger = logging.getLogger(__name__)

# Telegram's ceilings. A report never approaches the first; the copy path
# truncates the advertisement's text to fit the second (or the caption limit
# when photographs travel with it) rather than failing the whole delivery.
MAX_MESSAGE_LENGTH = 4096
MAX_CAPTION_LENGTH = 1024
# Telegram albums hold at most ten items.
MAX_ALBUM_ITEMS = 10
# How often the log repeats that the owner's DM is unreachable. The condition
# does not change between alerts, and one line per alert would bury the cause.
UNREACHABLE_LOG_INTERVAL_SECONDS = 3600.0
# The floor between ANY two messages this process sends to the owner, whatever
# their kind. The governor paces per action kind, so a report and the forward
# after it — two kinds — would otherwise go out back to back, and a backlog
# would drain as fast as the poll loop turns. Measured 2026-09-05: 1.5 s was
# too fast for a DM the owner had never written to — Telegram answered the
# 17th message in 52 s with PEER_FLOOD. Four seconds is the pace of a person
# forwarding by hand, enforced here where both kinds meet.
MIN_GAP_BETWEEN_SENDS_SECONDS = 4.0


class SendStatus(StrEnum):
    """How one owner-bound call ended."""

    SENT = "sent"
    # Come back later: FloodWait, budget, network.
    RETRY = "retry"
    # Final for THIS message: the source forbids forwarding, the message is
    # gone. The alert can still be completed another way.
    REJECTED = "rejected"
    # Final for the OWNER: blocked, privacy, unresolvable. Nothing sent this
    # way will arrive until a person fixes it; deliveries fall back to the bot.
    UNREACHABLE = "unreachable"


@dataclass(frozen=True, slots=True)
class SendOutcome:
    """Result of one call to the owner's DM."""

    status: SendStatus
    message_id: int | None = None
    error_code: str | None = None
    retry_after: int | None = None
    # For a multi-message forward: how many of the requested messages Telegram
    # did NOT return. A deleted album member does not raise — Telethon hands
    # back None in its place — so a "sent" forward can still be incomplete.
    missing: int = 0

    @property
    def sent(self) -> bool:
        """Whether Telegram accepted the message."""
        return self.status is SendStatus.SENT


# Errors that mean the owner cannot be written to at all, as opposed to one
# message failing. The governor settles all of these as REJECTED; the split
# into "this message" and "this owner" is made here, where the fallback lives.
_OWNER_UNREACHABLE_CODES = frozenset(
    {
        "halted",
        "channels_too_much",
        "PeerFloodError",
        "UserBannedInChannelError",
        "UserIsBlockedError",
        "UserPrivacyRestrictedError",
        "PeerIdInvalidError",
        "InputUserDeactivatedError",
        "ChatWriteForbiddenError",
        "owner_unresolved",
    }
)


class OwnerTransport:
    """The monitoring account's line to the owner's DM."""

    def __init__(
        self,
        *,
        client: Any,
        governor: TelegramActionGovernor,
        owner_ref: str,
        min_gap_seconds: float = MIN_GAP_BETWEEN_SENDS_SECONDS,
    ) -> None:
        self._client = client
        self._governor = governor
        self._min_gap = max(0.0, min_gap_seconds)
        # Stored as written ("nikitacometa" or "@nikitacometa"); Telethon
        # accepts both. Resolved lazily, once per process.
        self._owner_ref = owner_ref.strip()
        self._owner: object | None = None
        self._resolve_lock = asyncio.Lock()
        self._unreachable_logged_at: float | None = None
        self._last_send_at: float | None = None
        self._pace_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        """Whether an owner was named at all."""
        return bool(self._owner_ref)

    async def resolve(self) -> SendOutcome:
        """Turn the owner's @username into a peer the account can write to.

        StringSession keeps no entity cache across restarts, so a numeric id
        alone is not enough: the first call after a start pays one governed
        resolve, and the result lives for the life of the process.
        """
        if self._owner is not None:
            return SendOutcome(SendStatus.SENT)
        if not self._owner_ref:
            return SendOutcome(SendStatus.UNREACHABLE, error_code="owner_unresolved")
        async with self._resolve_lock:
            if self._owner is not None:
                return SendOutcome(SendStatus.SENT)
            hour = datetime.now(UTC).strftime("%Y%m%d%H")

            async def call() -> object:
                try:
                    return await self._client.get_entity(self._owner_ref)
                except ValueError as error:
                    # Telethon catches UsernameNotOccupiedError inside
                    # `_get_entity_from_string` and re-raises a bare
                    # ValueError, which the governor's list of rejected RPC
                    # errors never sees: it would settle the slot as ambiguous
                    # and propagate, stalling the delivery loop on a typo in
                    # OWNER_USERNAME. Putting the RPC error back makes it a
                    # clean rejection the fallback can act on.
                    raise errors.UsernameNotOccupiedError(request=None) from error

            result: ActionResult[object] = await self._governor.run(
                ActionKind.RESOLVE_USERNAME,
                f"owner:{self._owner_ref.lstrip('@').lower()}:{hour}",
                call,
            )
            if result.ok and result.value is not None:
                self._owner = result.value
                logger.info("Owner transport ready: alerts go to the owner's DM")
                return SendOutcome(SendStatus.SENT)
            outcome = _from_governed(result)
            if outcome.status is SendStatus.REJECTED:
                # A username that does not resolve is an owner problem, not a
                # message problem: nothing will reach him until it is fixed.
                outcome = SendOutcome(
                    SendStatus.UNREACHABLE, error_code=outcome.error_code or "owner_unresolved"
                )
                self._note_unreachable(outcome)
            return outcome

    async def send_report(self, body_html: str, *, reply_to: int | None = None) -> SendOutcome:
        """Send the composed report; returns its message id on success."""
        ready = await self.resolve()
        if not ready.sent:
            return ready

        async def call() -> Any:
            return await self._client.send_message(
                self._owner,
                body_html,
                parse_mode="html",
                link_preview=False,
                reply_to=reply_to,
            )

        return self._classify(await self._paced(ActionKind.OWNER_MESSAGE, "report", call))

    async def forward(self, *, chat_id: int, message_ids: list[int]) -> SendOutcome:
        """Forward the original message(s) from the source chat, as one call.

        An album's members forwarded together arrive grouped again on the
        owner's side, so the advertisement looks exactly as it did in the
        chat — photographs, caption, and the "forwarded from" header naming
        the author.
        """
        ready = await self.resolve()
        if not ready.sent:
            return ready
        ids = sorted({int(message_id) for message_id in message_ids})
        if not ids:
            return SendOutcome(SendStatus.REJECTED, error_code="nothing_to_forward")

        async def call() -> Any:
            return await self._client.forward_messages(self._owner, ids, from_peer=chat_id)

        forwarded: list[Any] = []

        async def call_and_keep() -> Any:
            value = await call()
            forwarded.append(value)
            return value

        outcome = await self._paced(
            ActionKind.OWNER_FORWARD, f"forward:{chat_id}:{ids[0]}", call_and_keep
        )
        if outcome.sent:
            # Telethon raises MessageIdInvalidError only when EVERY id is
            # gone. When some are, the call succeeds and the returned list
            # carries None where those messages should be — an album with a
            # deleted member arrives incomplete with no exception at all.
            value = forwarded[-1] if forwarded else None
            returned = value if isinstance(value, list) else [value]
            arrived = [item for item in returned if item is not None]
            missing = len(ids) - len(arrived)
            if not arrived:
                outcome = SendOutcome(SendStatus.REJECTED, error_code="MessageIdInvalidError")
            elif missing > 0:
                first = getattr(arrived[0], "id", None)
                outcome = SendOutcome(
                    SendStatus.SENT,
                    message_id=int(first) if isinstance(first, int) else None,
                    missing=missing,
                )
                logger.warning(
                    "Forward from %s arrived incomplete: %d of %d messages missing",
                    chat_id,
                    missing,
                    len(ids),
                )
        return self._classify(outcome)

    async def send_copy(
        self,
        *,
        text: str | None,
        photo_paths: list[str],
        header_html: str,
        reply_to: int | None = None,
    ) -> SendOutcome:
        """Re-send the advertisement ourselves when forwarding is refused.

        A chat that restricts saving content, or a message deleted since the
        alert was judged, cannot be forwarded. The text the daemon read and
        whichever photographs it already downloaded are the next best thing:
        not the original, and labelled as such in the header.
        """
        ready = await self.resolve()
        if not ready.sent:
            return ready
        existing = (await asyncio.to_thread(_existing_files, photo_paths))[:MAX_ALBUM_ITEMS]
        body = header_html + "\n\n" + html.escape(text or "", quote=False)
        if existing:
            caption = _fit(body, MAX_CAPTION_LENGTH)

            async def send_files() -> Any:
                sent = await self._client.send_file(
                    self._owner, existing, caption=caption, parse_mode="html", reply_to=reply_to
                )
                return sent[0] if isinstance(sent, list) and sent else sent

            call = send_files
        else:
            message = _fit(body, MAX_MESSAGE_LENGTH)

            async def send_text() -> Any:
                return await self._client.send_message(
                    self._owner,
                    message,
                    parse_mode="html",
                    link_preview=False,
                    reply_to=reply_to,
                )

            call = send_text

        return self._classify(await self._paced(ActionKind.OWNER_MESSAGE, "copy", call))

    async def edit_report(self, message_id: int, body_html: str) -> SendOutcome:
        """Rewrite an already-sent report, e.g. to say the original is unavailable."""
        ready = await self.resolve()
        if not ready.sent:
            return ready

        async def call() -> Any:
            return await self._client.edit_message(
                self._owner, message_id, body_html, parse_mode="html", link_preview=False
            )

        return self._classify(await self._paced(ActionKind.OWNER_MESSAGE, "edit", call))

    async def _paced(self, kind: ActionKind, label: str, call: Any) -> SendOutcome:
        """Run one owner-bound call: wait out the joint pace, then the governor."""
        async with self._pace_lock:
            now = asyncio.get_event_loop().time()
            if self._last_send_at is not None:
                wait = self._min_gap - (now - self._last_send_at)
                if wait > 0:
                    await asyncio.sleep(wait)
            result = await self._governor.run(kind, _once_key(label), call)
            if result.ok:
                self._last_send_at = asyncio.get_event_loop().time()
        return _from_governed(result)

    def _classify(self, outcome: SendOutcome) -> SendOutcome:
        """Split final refusals into "this message" and "this owner"."""
        if outcome.status is SendStatus.REJECTED and outcome.error_code in _OWNER_UNREACHABLE_CODES:
            outcome = SendOutcome(SendStatus.UNREACHABLE, error_code=outcome.error_code)
            # The peer may have been resolved to a stale access hash; forget
            # it so the next attempt resolves again rather than failing the
            # same way forever.
            if outcome.error_code == "PeerIdInvalidError":
                self._owner = None
        if outcome.status is SendStatus.UNREACHABLE:
            self._note_unreachable(outcome)
        return outcome

    def _note_unreachable(self, outcome: SendOutcome) -> None:
        now = asyncio.get_event_loop().time()
        if (
            self._unreachable_logged_at is None
            or now - self._unreachable_logged_at >= UNREACHABLE_LOG_INTERVAL_SECONDS
        ):
            self._unreachable_logged_at = now
            logger.error(
                "Owner DM unreachable (%s); housing alerts fall back to the bot with a link",
                outcome.error_code,
            )


def _from_governed(result: ActionResult[Any]) -> SendOutcome:
    """Translate a governed call's result into a delivery outcome."""
    if result.ok:
        value = result.value
        message_id = getattr(value, "id", None)
        if isinstance(value, list) and value:
            message_id = getattr(value[0], "id", None)
        return SendOutcome(
            SendStatus.SENT,
            message_id=int(message_id) if isinstance(message_id, int) else None,
        )
    if result.status is ActionStatus.REJECTED:
        return SendOutcome(SendStatus.REJECTED, error_code=result.error_code or "rejected")
    if result.status is ActionStatus.HALTED:
        # A halt is account-wide and needs a person: Telegram pushed back on
        # the account itself, and nothing more should leave it until someone
        # has looked. The owner is still told — through the bot, with a link
        # — rather than left waiting on a halt he does not know about.
        return SendOutcome(SendStatus.UNREACHABLE, error_code=result.error_code or "halted")
    retry_after = result.retry_after_seconds
    if result.status is ActionStatus.DENIED and result.denial is not None:
        # The pace or the budget said "not yet". That is the governor doing
        # its job, not a delivery failing, so the outbox is told to come
        # back rather than to count an attempt.
        code = "paced" if result.denial.scope is BudgetScope.COOLDOWN else "budget"
        return SendOutcome(
            SendStatus.RETRY,
            error_code=code,
            retry_after=max(1, int(retry_after)) if retry_after is not None else 30,
        )
    return SendOutcome(
        SendStatus.RETRY,
        error_code=result.error_code or result.status.value,
        retry_after=max(2, int(retry_after)) if retry_after is not None else 30,
    )


def _existing_files(paths: list[str]) -> list[str]:
    """Keep the photographs still on disk; runs off the event loop."""
    return [path for path in paths if Path(path).is_file()]


def _once_key(label: str) -> str:
    """An idempotency key that never replays.

    Owner messages are not joins: the row in housing_alerts is what makes a
    delivery idempotent (the report id is persisted before the forward), so
    the ledger key only has to be unique.
    """
    return f"owner:{label}:{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}"


def _fit(body_html: str, limit: int) -> str:
    """Trim a copy to Telegram's limit without cutting inside an HTML entity."""
    if len(body_html) <= limit:
        return body_html
    cut = body_html[: limit - 2]
    amp = cut.rfind("&")
    if amp != -1 and ";" not in cut[amp:]:
        cut = cut[:amp]
    return cut.rstrip() + "…"
