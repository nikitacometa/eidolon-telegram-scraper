"""Joining chats and reading their history, one governed page at a time.

Joining is the only part of reconnaissance that changes anything outside this
process, and history is the only part that runs for hours. Both are therefore
expressed as small, resumable steps: one join attempt, one page of a hundred
messages, each with its own budget reservation and its own durable outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from telethon import errors
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.functions.messages import (
    CheckChatInviteRequest,
    GetHistoryRequest,
    ImportChatInviteRequest,
)

from pipeline.discovery import ChatLink, extract_chat_links
from pipeline.governor import ActionResult, ActionStatus, TelegramActionGovernor
from pipeline.ingestion import format_sender_identity, reply_to_message_id
from pipeline.recon_models import (
    ActionKind,
    ChatMembership,
    ScoutMessage,
)

logger = logging.getLogger(__name__)

# Telegram serves at most 100 messages per history call whatever limit is
# asked for, so a page is the natural unit of progress.
HISTORY_PAGE_SIZE = 100


@dataclass(frozen=True, slots=True)
class JoinOutcome:
    """What a join attempt actually achieved."""

    membership: ChatMembership
    error_code: str | None = None
    # Only an invite join learns the chat from Telegram's answer; a username
    # join already held the entity before asking.
    chat: object | None = None

    @property
    def is_member(self) -> bool:
        """Whether the account can now read the chat."""
        return self.membership is ChatMembership.MEMBER


@dataclass(frozen=True, slots=True)
class HistoryPage:
    """One page of history plus what it points at."""

    messages: tuple[ScoutMessage, ...]
    links: tuple[ChatLink, ...]
    next_offset_id: int | None
    exhausted: bool
    # True when the walk crossed the requested lookback horizon: Telegram had
    # more history, we no longer wanted it. Distinguishes COMPLETE from
    # EXHAUSTED when a page ends exactly on the horizon and has no survivors.
    crossed_cutoff: bool = False


class TelegramCrawler:
    """Performs the two expensive reconnaissance actions."""

    def __init__(self, *, client: object, governor: TelegramActionGovernor) -> None:
        self._client = client
        self._governor = governor

    async def join(
        self,
        *,
        channel: object,
        chat_ref: str,
        job_id: str | None = None,
        candidate_id: int | None = None,
        attempt: int = 0,
    ) -> ActionResult[JoinOutcome]:
        """Attempt to join one public chat.

        A request awaiting admin approval is reported as ``REQUESTED``, never
        as membership: code that reads "no exception" as "joined" would start
        backfilling a chat it cannot read.

        ``job_id`` is optional: the standing join queue belongs to no crawl,
        and its attempts must not claim a foreign key into one. The chat
        reference alone already makes the idempotency key unique.
        """

        async def call() -> JoinOutcome:
            try:
                await self._client(JoinChannelRequest(channel=channel))  # type: ignore[operator]
            except errors.InviteRequestSentError:
                logger.info("Join request for %s awaits admin approval", chat_ref)
                return JoinOutcome(membership=ChatMembership.REQUESTED)
            except errors.UserAlreadyParticipantError:
                return JoinOutcome(membership=ChatMembership.MEMBER)
            return JoinOutcome(membership=ChatMembership.MEMBER)

        result: ActionResult[JoinOutcome] = await self._governor.run(
            ActionKind.JOIN,
            f"{job_id or 'queue'}:join:{chat_ref}:{attempt}",
            call,
            job_id=job_id,
            candidate_id=candidate_id,
        )
        if result.status is ActionStatus.REJECTED:
            return ActionResult(
                status=result.status,
                value=JoinOutcome(
                    membership=ChatMembership.FAILED,
                    error_code=result.error_code,
                ),
                error_code=result.error_code,
                duration_ms=result.duration_ms,
            )
        return result

    async def check_invite(
        self,
        *,
        invite_hash: str,
        chat_ref: str,
        attempt: int = 0,
    ) -> ActionResult[JoinOutcome]:
        """Ask whether a pending invite request has since been approved.

        CheckChatInvite never joins anything: on an approved membership it
        answers ChatInviteAlready with the chat, otherwise it describes the
        invite. Spends the invite_check budget, which exists exactly for
        this kind of look."""

        async def call() -> JoinOutcome:
            try:
                invite = await self._client(CheckChatInviteRequest(hash=invite_hash))  # type: ignore[operator]
            except (errors.InviteHashExpiredError, errors.InviteHashInvalidError):
                return JoinOutcome(membership=ChatMembership.FAILED, error_code="invite_invalid")
            chat = getattr(invite, "chat", None)
            if chat is not None and not getattr(invite, "request_needed", False):
                # ChatInviteAlready carries the chat and only comes back to
                # a member; a plain ChatInvite for a public preview keeps
                # request_needed/False semantics apart via the chat check
                # below.
                already = type(invite).__name__ == "ChatInviteAlready"
                if already:
                    return JoinOutcome(membership=ChatMembership.MEMBER, chat=chat)
            return JoinOutcome(membership=ChatMembership.REQUESTED)

        return await self._governor.run(
            ActionKind.INVITE_CHECK,
            f"reconcile:invite:{chat_ref}:{attempt}",
            call,
        )

    async def join_invite(
        self,
        *,
        invite_hash: str,
        chat_ref: str,
        attempt: int = 0,
    ) -> ActionResult[JoinOutcome]:
        """Join one chat through its invite link.

        Spends the same join budget as a username join: to the account's
        standing an invite is a join like any other. A dead or expired hash is
        a terminal failure, not something to retry into; ``REQUESTED`` keeps
        its meaning of "an admin has not answered yet".

        The chat comes back inside Telegram's reply, which is the only place it
        can come from: an invite link names nothing until it is used.
        """

        async def call() -> JoinOutcome:
            try:
                response = await self._client(ImportChatInviteRequest(hash=invite_hash))  # type: ignore[operator]
            except errors.InviteRequestSentError:
                logger.info("Join request for %s awaits admin approval", chat_ref)
                return JoinOutcome(membership=ChatMembership.REQUESTED)
            except errors.UserAlreadyParticipantError:
                # The invite cannot be imported twice, but checking it still
                # tells us which chat we are already in.
                invite = await self._client(CheckChatInviteRequest(hash=invite_hash))  # type: ignore[operator]
                return JoinOutcome(
                    membership=ChatMembership.MEMBER,
                    chat=getattr(invite, "chat", None),
                )
            except (errors.InviteHashExpiredError, errors.InviteHashInvalidError):
                return JoinOutcome(membership=ChatMembership.FAILED, error_code="invite_invalid")
            chats = list(getattr(response, "chats", ()) or ())
            return JoinOutcome(
                membership=ChatMembership.MEMBER,
                chat=chats[0] if chats else None,
            )

        result: ActionResult[JoinOutcome] = await self._governor.run(
            ActionKind.JOIN,
            f"queue:join:{chat_ref}:{attempt}",
            call,
        )
        if result.status is ActionStatus.REJECTED:
            return ActionResult(
                status=result.status,
                value=JoinOutcome(
                    membership=ChatMembership.FAILED,
                    error_code=result.error_code,
                ),
                error_code=result.error_code,
                duration_ms=result.duration_ms,
            )
        return result

    async def history_page(
        self,
        *,
        chat_id: int,
        peer: object,
        offset_id: int = 0,
        min_id: int = 0,
        not_before: datetime | None = None,
        job_id: str | None = None,
    ) -> ActionResult[HistoryPage]:
        """Read one page of history, oldest-bound by ``min_id``.

        Paging walks backwards from ``offset_id``; ``min_id`` is where a
        previous run stopped, so a resumed crawl does not re-read what it
        already stored. ``not_before`` is the job's lookback window: messages
        older than it are dropped and end the walk, so a job that asked for a
        week does not quietly archive a year.

        ``job_id`` is optional because the background archive is a standing
        intent rather than a job: its pages belong to no crawl and must not
        claim a foreign key into one.
        """

        async def call() -> HistoryPage:
            response = await self._client(  # type: ignore[operator]
                GetHistoryRequest(
                    peer=peer,
                    offset_id=offset_id,
                    offset_date=None,
                    add_offset=0,
                    limit=HISTORY_PAGE_SIZE,
                    max_id=0,
                    min_id=min_id,
                    hash=0,
                )
            )
            return self._read_page(response, chat_id=chat_id, not_before=not_before)

        return await self._governor.run(
            ActionKind.HISTORY_PAGE,
            f"{job_id or 'backfill'}:history:{chat_id}:{offset_id}",
            call,
            job_id=job_id,
        )

    def _read_page(
        self,
        response: object,
        *,
        chat_id: int,
        not_before: datetime | None = None,
    ) -> HistoryPage:
        raw = list(getattr(response, "messages", ()) or ())
        crossed_cutoff = False
        messages: list[ScoutMessage] = []
        links: list[ChatLink] = []
        seen_links: set[str] = set()
        # A history page carries its senders in a side list, not on the
        # messages. Dropping it made every backfilled message anonymous, so
        # "who keeps announcing events here" was answerable over the few
        # thousand live messages and not over the sixty thousand archived ones.
        senders = _sender_names(response)

        for item in raw:
            message_id = getattr(item, "id", None)
            if message_id is None:
                continue
            posted = getattr(item, "date", None)
            if not_before is not None and isinstance(posted, datetime) and posted < not_before:
                crossed_cutoff = True
                continue
            text = getattr(item, "message", None)
            entities = getattr(item, "entities", None)
            forward = getattr(item, "fwd_from", None)
            forward_peer = getattr(forward, "from_id", None)

            sender_id = _peer_user_id(getattr(item, "from_id", None))
            sender_name = senders.get(sender_id) if sender_id is not None else None
            messages.append(
                ScoutMessage(
                    chat_id=chat_id,
                    telegram_msg_id=int(message_id),
                    date=str(getattr(item, "date", "")),
                    text=text,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    entities=[],
                    forward_chat_id=_peer_channel_id(forward_peer),
                    forward_message_id=_optional_int(getattr(forward, "channel_post", None)),
                    reply_to_message_id=reply_to_message_id(item),
                    source="backfill",
                )
            )

            for link in extract_chat_links(text, entities):
                key = f"invite:{link.invite_hash}" if link.invite_hash else f"name:{link.username}"
                if key not in seen_links:
                    seen_links.add(key)
                    links.append(link)

        oldest = min((message.telegram_msg_id for message in messages), default=None)
        return HistoryPage(
            messages=tuple(messages),
            links=tuple(links),
            next_offset_id=oldest,
            # Only an empty page proves Telegram has nothing older. A SHORT
            # page does not: getHistory returns fewer than `limit` in the
            # middle of a chat whose id range is sparse with deletions, and
            # reading that as the end truncated one chat at message 4879 of
            # 6939 while marking it complete. Crossing the lookback window is
            # the other ending: there is more, we no longer want it.
            exhausted=not raw or crossed_cutoff,
            crossed_cutoff=crossed_cutoff,
        )


def _sender_names(response: object) -> dict[int, str]:
    """Map user id to a handle-first identity from the page's side list."""
    names: dict[int, str] = {}
    for user in getattr(response, "users", ()) or ():
        user_id = _optional_int(getattr(user, "id", None))
        if user_id is None:
            continue
        label = format_sender_identity(user)
        if label:
            names[user_id] = label
    return names


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def _peer_user_id(peer: object) -> int | None:
    return _optional_int(getattr(peer, "user_id", None))


def _peer_channel_id(peer: object) -> int | None:
    channel_id = _optional_int(getattr(peer, "channel_id", None))
    if channel_id is None:
        return None
    # Match the -100… form used everywhere else so forward origins can be
    # compared against chat ids without a second conversion.
    return int(f"-100{channel_id}")
