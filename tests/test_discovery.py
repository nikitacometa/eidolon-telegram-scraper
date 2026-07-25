"""Tests for pipeline/discovery.py — link extraction and search surfaces."""

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from telethon.tl.types import Channel, Chat, PeerChannel, User

from pipeline.discovery import (
    TelegramDiscovery,
    describe_chat,
    extract_chat_links,
)
from pipeline.governor import ActionStatus, TelegramActionGovernor
from pipeline.recon_models import (
    ActionKind,
    BudgetRule,
    ChatVisibility,
    DiscoverySource,
    JobRequest,
)
from storage.scout import ScoutDatabase


@pytest.fixture
async def scout(tmp_path: Path) -> AsyncIterator[ScoutDatabase]:
    """Open a scout database backed by a temporary file."""
    database = ScoutDatabase(tmp_path / "scout.db")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def job_id(scout: ScoutDatabase) -> str:
    """Create a real job: the action ledger references it."""
    job = await scout.create_job(
        JobRequest(idempotency_key="discovery-test", topic="da nang housing")
    )
    return job.id


def _channel(
    channel_id: int,
    *,
    username: str | None = None,
    title: str = "Chat",
    broadcast: bool = False,
    **flags: bool,
) -> Channel:
    return Channel(
        id=channel_id,
        title=title,
        photo=None,
        date=None,
        username=username,
        broadcast=broadcast,
        megagroup=not broadcast,
        **flags,
    )


# ----------------------------------------------------------------------
# Link extraction
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("join https://t.me/danang_expats now", ["danang_expats"]),
        ("t.me/DaNangHousing", ["dananghousing"]),
        ("ask @danang_rent about it", ["danang_rent"]),
        ("both t.me/first and @second", ["first", "second"]),
        ("no links here", []),
        ("too short @ab", []),
        ("telegram's own @telegram is skipped", []),
    ],
)
def test_username_links_are_extracted(text: str, expected: list[str]) -> None:
    """Chat locators in text become candidates; noise does not."""
    assert [link.username for link in extract_chat_links(text)] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://t.me/+AbCdEf12345", "AbCdEf12345"),
        ("https://t.me/joinchat/XyZ98765", "XyZ98765"),
    ],
)
def test_invite_links_are_extracted_without_joining(text: str, expected: str) -> None:
    """An invite hash is recognised, but it is not proof of a public chat."""
    links = extract_chat_links(text)

    assert len(links) == 1
    assert links[0].invite_hash == expected
    assert links[0].username is None


def test_the_same_chat_mentioned_twice_yields_one_link() -> None:
    """A message repeating a link must not inflate the candidate list."""
    text = "t.me/danang_expats is great, see https://t.me/danang_expats and @danang_expats"

    assert len(extract_chat_links(text)) == 1


def test_links_hidden_behind_entity_urls_are_followed() -> None:
    """Markup can hide the real target behind display text."""
    entities = [SimpleNamespace(url="https://t.me/hidden_chat")]

    links = extract_chat_links("click here", entities)

    assert [link.username for link in links] == ["hidden_chat"]


def test_empty_text_is_handled() -> None:
    """Service messages carry no text at all."""
    assert extract_chat_links(None) == []


# ----------------------------------------------------------------------
# Chat description
# ----------------------------------------------------------------------


def test_public_username_proves_public_scope() -> None:
    """A username is the one public proof available without joining."""
    described = describe_chat(_channel(777, username="danang_expats", title="Da Nang Expats"))

    assert described is not None
    assert described.visibility is ChatVisibility.PUBLIC
    assert described.username == "danang_expats"
    assert described.identity.peer_id == -1000000000777
    assert described.chat_type == "supergroup"


def test_chat_without_a_username_is_not_assumed_public() -> None:
    """Being reachable is not the same as being public."""
    described = describe_chat(_channel(778, username=None))

    assert described is not None
    assert described.visibility is ChatVisibility.UNKNOWN


def test_broadcast_channels_are_labelled() -> None:
    """A channel and a group need different handling later."""
    described = describe_chat(_channel(779, username="danang_news", broadcast=True))

    assert described is not None
    assert described.chat_type == "channel"


def test_scam_and_fake_flags_are_carried_forward() -> None:
    """Telegram's own labels are the cheapest risk signal available."""
    described = describe_chat(_channel(780, username="pump_it", scam=True, fake=True))

    assert described is not None
    assert set(described.flags) == {"scam", "fake"}


def test_users_are_not_crawl_targets() -> None:
    """Search results mix people in with chats."""
    assert describe_chat(User(id=1, first_name="Nikita")) is None


def test_legacy_groups_are_described() -> None:
    """Small groups still come back from search."""
    described = describe_chat(
        Chat(id=5, title="Old Group", photo=None, participants_count=10, date=None, version=1)
    )

    assert described is not None
    assert described.chat_type == "group"


# ----------------------------------------------------------------------
# Search surfaces
# ----------------------------------------------------------------------


class FakeClient:
    """Records requests and replays canned responses."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[object] = []

    async def __call__(self, request: object) -> object:
        self.requests.append(request)
        return self.response


def _post_response(*channels: Channel) -> SimpleNamespace:
    return SimpleNamespace(
        chats=list(channels),
        messages=[
            SimpleNamespace(id=index, peer_id=PeerChannel(channel.id))
            for index, channel in enumerate(channels, start=1)
        ],
        users=[],
    )


async def test_hashtag_search_maps_posts_to_their_chats(scout: ScoutDatabase, job_id: str) -> None:
    """Post search returns messages; the candidate is the chat behind each."""
    client = FakeClient(
        _post_response(
            _channel(1, username="danang_expats", title="Da Nang Expats"),
            _channel(2, username="danang_housing", title="Da Nang Housing"),
        )
    )
    discovery = TelegramDiscovery(
        client=client,
        governor=TelegramActionGovernor(scout=scout),
    )

    result = await discovery.search_hashtag(job_id=job_id, hashtag="#danang")

    assert result.ok
    assert result.value is not None
    assert [chat.username for chat in result.value] == ["danang_expats", "danang_housing"]
    assert all(chat.evidence.source is DiscoverySource.HASHTAG_SEARCH for chat in result.value)


async def test_one_query_is_one_origin(scout: ScoutDatabase, job_id: str) -> None:
    """Chats found by a single query share that query as their origin.

    Otherwise one lucky search would look like several independent signals.
    """
    client = FakeClient(
        _post_response(
            _channel(1, username="chat_one"),
            _channel(2, username="chat_two"),
        )
    )
    discovery = TelegramDiscovery(
        client=client,
        governor=TelegramActionGovernor(scout=scout),
    )

    result = await discovery.search_hashtag(job_id=job_id, hashtag="danang")

    assert result.value is not None
    assert {chat.evidence.origin_key for chat in result.value} == {"hashtag:danang"}


async def test_hashtag_search_never_spends_stars(scout: ScoutDatabase, job_id: str) -> None:
    """Paid search must not happen without an explicit decision."""
    client = FakeClient(_post_response(_channel(1, username="danang_expats")))
    discovery = TelegramDiscovery(
        client=client,
        governor=TelegramActionGovernor(scout=scout),
    )

    await discovery.search_hashtag(job_id=job_id, hashtag="danang")

    request = client.requests[0]
    assert request.hashtag == "danang"
    assert request.query is None
    assert request.allow_paid_stars is None


async def test_similar_channels_returns_the_chat_list(scout: ScoutDatabase, job_id: str) -> None:
    """The recommendation graph returns chats rather than posts."""
    client = FakeClient(
        SimpleNamespace(chats=[_channel(9, username="danang_food", broadcast=True)], messages=[])
    )
    discovery = TelegramDiscovery(
        client=client,
        governor=TelegramActionGovernor(scout=scout),
    )

    result = await discovery.similar_channels(
        job_id=job_id,
        channel=object(),
        origin_key="danang_expats",
    )

    assert result.value is not None
    assert [chat.username for chat in result.value] == ["danang_food"]
    assert result.value[0].evidence.origin_key == "similar:danang_expats"


async def test_contacts_search_reads_the_chat_list(scout: ScoutDatabase, job_id: str) -> None:
    """Username search returns users and chats together."""
    client = FakeClient(
        SimpleNamespace(
            chats=[_channel(3, username="danang_bikes")],
            users=[User(id=1, first_name="Someone")],
            messages=[],
        )
    )
    discovery = TelegramDiscovery(
        client=client,
        governor=TelegramActionGovernor(scout=scout),
    )

    result = await discovery.search_contacts(job_id=job_id, query="danang")

    assert result.value is not None
    assert [chat.username for chat in result.value] == ["danang_bikes"]


async def test_exhausted_search_budget_stops_discovery(scout: ScoutDatabase, job_id: str) -> None:
    """Discovery is throttled by the same budgets as everything else."""
    client = FakeClient(_post_response(_channel(1, username="danang_expats")))
    discovery = TelegramDiscovery(
        client=client,
        governor=TelegramActionGovernor(
            scout=scout,
            policy={ActionKind.HASHTAG_SEARCH: BudgetRule(per_day=1)},
        ),
    )

    await discovery.search_hashtag(job_id=job_id, hashtag="danang")
    second = await discovery.search_hashtag(job_id=job_id, hashtag="hoian")

    assert second.status is ActionStatus.DENIED
    assert len(client.requests) == 1


async def test_repeated_search_within_a_job_is_not_repeated(
    scout: ScoutDatabase, job_id: str
) -> None:
    """The same query twice in one job must not spend two slots."""
    client = FakeClient(_post_response(_channel(1, username="danang_expats")))
    discovery = TelegramDiscovery(
        client=client,
        governor=TelegramActionGovernor(scout=scout),
    )

    await discovery.search_hashtag(job_id=job_id, hashtag="danang")
    await discovery.search_hashtag(job_id=job_id, hashtag="danang")

    assert await scout.budget_usage(
        account_id="owner-primary",
        kind=ActionKind.HASHTAG_SEARCH,
    ) == (1, 1)
