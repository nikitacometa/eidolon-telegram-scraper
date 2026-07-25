"""Finding candidate chats through Telegram's own search surfaces.

Telegram publishes no index of its public chats, so discovery is a funnel of
weak signals: hashtag search over public posts, the similar-channel graph,
username search, and links found inside messages already read. Every call here
goes through the governor, and nothing reaches a language model before the
chat's public scope is proven.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from telethon import utils
from telethon.tl.functions.channels import (
    GetChannelRecommendationsRequest,
    SearchPostsRequest,
)
from telethon.tl.functions.contacts import SearchRequest as ContactsSearchRequest
from telethon.tl.types import Channel, Chat, InputPeerEmpty

from pipeline.governor import ActionResult, ActionStatus, TelegramActionGovernor
from pipeline.recon_models import (
    ActionKind,
    ChatIdentity,
    ChatVisibility,
    DiscoverySource,
    Evidence,
    normalize_username,
)

logger = logging.getLogger(__name__)

# t.me/name, t.me/+hash, t.me/joinchat/hash, and bare @mentions. Deliberately
# narrow: query strings and paths beyond the first segment are not part of a
# chat locator.
_TME_LINK = re.compile(
    r"(?:https?://)?t\.me/(?:(?P<invite>\+[A-Za-z0-9_-]+|joinchat/[A-Za-z0-9_-]+)"
    r"|(?P<name>[A-Za-z][A-Za-z0-9_]{3,31}))(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_MENTION = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_]{3,31})(?![A-Za-z0-9_])")

# Usernames that resolve to Telegram's own surfaces rather than to a chat
# worth crawling.
_RESERVED_NAMES = frozenset(
    {
        "telegram",
        "durov",
        "spambot",
        "botfather",
        "joinchat",
        "addstickers",
        "proxy",
        "socks",
        "share",
        "iv",
        "s",
    }
)


@dataclass(frozen=True, slots=True)
class ChatLink:
    """A chat locator found inside a message."""

    username: str | None = None
    invite_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.username and not self.invite_hash:
            raise ValueError("a chat link needs a username or an invite hash")


@dataclass(slots=True)
class DiscoveredChat:
    """A candidate chat plus the signal that surfaced it."""

    identity: ChatIdentity
    evidence: Evidence
    visibility: ChatVisibility = ChatVisibility.UNKNOWN
    title: str | None = None
    username: str | None = None
    chat_type: str | None = None
    participants: int | None = None
    flags: tuple[str, ...] = field(default_factory=tuple)


def extract_chat_links(
    text: str | None, entities: Sequence[object] | None = None
) -> list[ChatLink]:
    """Pull chat locators out of message text.

    Invite hashes are returned as found so the caller can check them without
    joining; they are never resolved into a join on their own, because an
    invite link is not proof that a chat is public.
    """
    if not text:
        return []

    found: list[ChatLink] = []
    seen: set[str] = set()

    for match in _TME_LINK.finditer(text):
        invite = match.group("invite")
        if invite:
            digest = invite.removeprefix("joinchat/").removeprefix("+")
            key = f"invite:{digest}"
            if key not in seen:
                seen.add(key)
                found.append(ChatLink(invite_hash=digest))
            continue
        name = normalize_username(match.group("name"))
        key = f"name:{name}"
        if name and name not in _RESERVED_NAMES and key not in seen:
            seen.add(key)
            found.append(ChatLink(username=name))

    for match in _MENTION.finditer(text):
        name = normalize_username(match.group(1))
        key = f"name:{name}"
        if name and name not in _RESERVED_NAMES and key not in seen:
            seen.add(key)
            found.append(ChatLink(username=name))

    for entity in entities or ():
        url = getattr(entity, "url", None)
        if isinstance(url, str):
            found.extend(link for link in extract_chat_links(url) if _is_new(link, seen))

    return found


def _is_new(link: ChatLink, seen: set[str]) -> bool:
    key = f"invite:{link.invite_hash}" if link.invite_hash else f"name:{link.username}"
    if key in seen:
        return False
    seen.add(key)
    return True


def describe_chat(entity: object) -> DiscoveredChat | None:
    """Turn a Telethon chat object into a candidate description.

    Returns ``None`` for anything that is not a group or channel: users and
    bots are not crawl targets.
    """
    if not isinstance(entity, Channel | Chat):
        return None

    username = getattr(entity, "username", None)
    if not username:
        # Newer layers carry additional usernames alongside the main one.
        extra = getattr(entity, "usernames", None) or []
        for candidate in extra:
            value = getattr(candidate, "username", None)
            if value:
                username = value
                break

    peer_id = utils.get_peer_id(entity)
    normalized = normalize_username(username) if username else None
    flags = tuple(
        flag
        for flag in ("scam", "fake", "restricted", "gigagroup")
        if bool(getattr(entity, flag, False))
    )
    if isinstance(entity, Channel):
        chat_type = "channel" if getattr(entity, "broadcast", False) else "supergroup"
    else:
        chat_type = "group"

    return DiscoveredChat(
        identity=ChatIdentity(
            peer_id=peer_id,
            username=normalized,
            title=str(getattr(entity, "title", "") or "") or None,
            chat_type=chat_type,
        ),
        # Replaced by the caller with the signal that actually found this chat.
        evidence=Evidence(source=DiscoverySource.CONTACTS_SEARCH, origin_key="unset"),
        # A public username is the one form of proof available without joining.
        visibility=ChatVisibility.PUBLIC if normalized else ChatVisibility.UNKNOWN,
        title=str(getattr(entity, "title", "") or "") or None,
        username=normalized,
        chat_type=chat_type,
        participants=getattr(entity, "participants_count", None),
        flags=flags,
    )


class TelegramDiscovery:
    """Reads Telegram's public search surfaces through the governor."""

    def __init__(self, *, client: object, governor: TelegramActionGovernor) -> None:
        self._client = client
        self._governor = governor

    async def search_hashtag(
        self,
        *,
        job_id: str,
        hashtag: str,
        limit: int = 50,
    ) -> ActionResult[list[DiscoveredChat]]:
        """Search public posts by hashtag, including chats we are not in.

        Hashtag search is the free half of ``channels.searchPosts``; the
        full-text half consumes metered slots and is kept for later waves.
        """
        tag = hashtag.lstrip("#").strip()
        request = SearchPostsRequest(
            hashtag=tag,
            offset_rate=0,
            offset_peer=InputPeerEmpty(),
            offset_id=0,
            limit=limit,
        )
        return await self._collect(
            kind=ActionKind.HASHTAG_SEARCH,
            idempotency_key=f"{job_id}:hashtag:{tag}",
            request=request,
            job_id=job_id,
            source=DiscoverySource.HASHTAG_SEARCH,
            origin_key=f"hashtag:{tag}",
        )

    async def search_fulltext(
        self,
        *,
        job_id: str,
        query: str,
        limit: int = 50,
    ) -> ActionResult[list[DiscoveredChat]]:
        """Search public posts by phrase, spending a metered search slot.

        ``allow_paid_stars`` is deliberately absent: the crawl never spends
        Stars on its own.
        """
        request = SearchPostsRequest(
            query=query,
            offset_rate=0,
            offset_peer=InputPeerEmpty(),
            offset_id=0,
            limit=limit,
        )
        return await self._collect(
            kind=ActionKind.FULLTEXT_SEARCH,
            idempotency_key=f"{job_id}:fulltext:{query}",
            request=request,
            job_id=job_id,
            source=DiscoverySource.FULLTEXT_SEARCH,
            origin_key=f"fulltext:{query}",
        )

    async def search_contacts(
        self,
        *,
        job_id: str,
        query: str,
        limit: int = 30,
    ) -> ActionResult[list[DiscoveredChat]]:
        """Search chat titles and usernames."""
        return await self._collect(
            kind=ActionKind.CONTACTS_SEARCH,
            idempotency_key=f"{job_id}:contacts:{query}",
            request=ContactsSearchRequest(q=query, limit=limit),
            job_id=job_id,
            source=DiscoverySource.CONTACTS_SEARCH,
            origin_key=f"contacts:{query}",
        )

    async def similar_channels(
        self,
        *,
        job_id: str,
        channel: object,
        origin_key: str,
    ) -> ActionResult[list[DiscoveredChat]]:
        """Ask Telegram which channels resemble one we already found.

        This is the one discovery source that is a graph rather than a query,
        and it is the reason a good seed matters more than a clever keyword.
        """
        return await self._collect(
            kind=ActionKind.RECOMMENDATIONS,
            idempotency_key=f"{job_id}:recs:{origin_key}",
            request=GetChannelRecommendationsRequest(channel=channel),
            job_id=job_id,
            source=DiscoverySource.RECOMMENDATIONS,
            origin_key=f"similar:{origin_key}",
        )

    async def _collect(
        self,
        *,
        kind: ActionKind,
        idempotency_key: str,
        request: object,
        job_id: str,
        source: DiscoverySource,
        origin_key: str,
    ) -> ActionResult[list[DiscoveredChat]]:
        async def call() -> list[DiscoveredChat]:
            response = await self._client(request)  # type: ignore[operator]
            return self._describe_response(response, source=source, origin_key=origin_key)

        result: ActionResult[list[DiscoveredChat]] = await self._governor.run(
            kind,
            idempotency_key,
            call,
            job_id=job_id,
        )
        if result.status is ActionStatus.OK and result.value is not None:
            logger.info(
                "Discovery %s via %s found %d chats",
                kind.value,
                origin_key,
                len(result.value),
            )
        return result

    def _describe_response(
        self,
        response: object,
        *,
        source: DiscoverySource,
        origin_key: str,
    ) -> list[DiscoveredChat]:
        """Map a Telegram response onto candidate chats.

        Post searches return messages, so the chat comes from each message's
        peer; chat searches return chats directly. Both shapes carry a ``chats``
        list, which is the only part this needs.
        """
        chats = list(getattr(response, "chats", ()) or ())
        by_peer = {utils.get_peer_id(chat): chat for chat in chats}

        wanted: list[object]
        messages = list(getattr(response, "messages", ()) or ())
        if messages:
            wanted = []
            for message in messages:
                peer = getattr(message, "peer_id", None)
                if peer is None:
                    continue
                chat = by_peer.get(utils.get_peer_id(peer))
                if chat is not None and chat not in wanted:
                    wanted.append(chat)
        else:
            wanted = chats

        described: list[DiscoveredChat] = []
        for chat in wanted:
            candidate = describe_chat(chat)
            if candidate is None:
                continue
            # Every chat surfaced by one query shares that query as its origin,
            # so a single search cannot manufacture independent support.
            candidate.evidence = Evidence(source=source, origin_key=origin_key)
            described.append(candidate)
        return described
