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
from telethon.tl.functions.messages import GetHistoryRequest

from pipeline.discovery import ChatLink, extract_chat_links
from pipeline.governor import ActionResult, ActionStatus, TelegramActionGovernor
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


class TelegramCrawler:
    """Performs the two expensive reconnaissance actions."""

    def __init__(self, *, client: object, governor: TelegramActionGovernor) -> None:
        self._client = client
        self._governor = governor

    async def join(
        self,
        *,
        job_id: str,
        candidate_id: int,
        channel: object,
        chat_ref: str,
    ) -> ActionResult[JoinOutcome]:
        """Attempt to join one public chat.

        A request awaiting admin approval is reported as ``REQUESTED``, never
        as membership: code that reads "no exception" as "joined" would start
        backfilling a chat it cannot read.
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
            f"{job_id}:join:{chat_ref}",
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

            messages.append(
                ScoutMessage(
                    chat_id=chat_id,
                    telegram_msg_id=int(message_id),
                    date=str(getattr(item, "date", "")),
                    text=text,
                    sender_id=_peer_user_id(getattr(item, "from_id", None)),
                    entities=[],
                    forward_chat_id=_peer_channel_id(forward_peer),
                    forward_message_id=_optional_int(getattr(forward, "channel_post", None)),
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
            # A short page means Telegram has nothing older to give; crossing
            # the lookback window means we no longer want what it has.
            exhausted=len(raw) < HISTORY_PAGE_SIZE or crossed_cutoff,
        )


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
