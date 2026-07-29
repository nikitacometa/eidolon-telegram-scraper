"""Tests for the paced join queue."""

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from telethon import errors
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.types import Channel

from pipeline.crawler import TelegramCrawler
from pipeline.governor import TelegramActionGovernor
from pipeline.joiner import JoinWorker
from pipeline.models import ObservationMode
from pipeline.recon_models import ActionKind, BudgetRule, JoinQueueState
from storage.db import Database
from storage.scout import ScoutDatabase

CHANNEL = Channel(
    id=101,
    title="Da Nang Events",
    photo=None,
    date=None,
    username="danangevents",
    broadcast=False,
    megagroup=True,
)
CHAT_ID = -1000000000101


@pytest.fixture
async def scout(tmp_path: Path) -> AsyncIterator[ScoutDatabase]:
    """Reconnaissance state in a temporary file."""
    database = ScoutDatabase(tmp_path / "scout.db")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Live monitoring state in a temporary file."""
    database = Database(tmp_path / "eidolon.db")
    await database.connect()
    yield database
    await database.close()


class FakeJoins:
    """Answers join requests, optionally with an error."""

    def __init__(self, error: type[BaseException] | None = None) -> None:
        self.error = error
        self.joined: list[str] = []

    async def __call__(self, request: object) -> object:
        assert isinstance(request, JoinChannelRequest)
        if self.error is not None:
            raise self.error(request=None)
        self.joined.append(str(getattr(request.channel, "username", request.channel)))
        return SimpleNamespace(chats=[], users=[])


def _worker(
    scout: ScoutDatabase,
    db: Database,
    client: FakeJoins,
    *,
    policy: dict[ActionKind, BudgetRule] | None = None,
    entity: object | None = CHANNEL,
    reloads: list[int] | None = None,
) -> JoinWorker:
    governor = TelegramActionGovernor(scout=scout, policy=policy)

    async def resolve(ref: str) -> object | None:
        return entity

    async def on_joined() -> None:
        if reloads is not None:
            reloads.append(1)

    return JoinWorker(
        scout=scout,
        db=db,
        crawler=TelegramCrawler(client=client, governor=governor),
        resolve_entity=resolve,
        on_joined=on_joined,
        poll_seconds=0.01,
    )


async def test_a_join_wires_up_monitoring_and_the_archive(
    scout: ScoutDatabase, db: Database
) -> None:
    """One successful join must leave the chat watched and queued for history."""
    client = FakeJoins()
    reloads: list[int] = []
    worker = _worker(scout, db, client, reloads=reloads)
    await scout.enqueue_join(
        chat_ref="@danangevents",
        label="Danang Events",
        watcher_name="danang-signal",
        target_days=730,
    )

    assert await worker.run_once()

    assert client.joined == ["danangevents"]
    snapshot = await db.observation_snapshot()
    assert snapshot[CHAT_ID].mode is ObservationMode.MONITOR
    assert snapshot[CHAT_ID].watcher_names == ("danang-signal",)
    targets = await scout.backfill_targets()
    assert targets[0].chat_id == CHAT_ID
    assert targets[0].target_days == 730
    assert reloads == [1]
    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.JOINED
    assert queue[0].joined_chat_id == CHAT_ID


async def test_queue_normalizes_the_reference(scout: ScoutDatabase) -> None:
    """The same chat written three ways is one queue entry."""
    await scout.enqueue_join(chat_ref="@DanangEvents")
    await scout.enqueue_join(chat_ref="https://t.me/danangevents")
    await scout.enqueue_join(chat_ref="danangevents")

    assert len(await scout.join_queue()) == 1


async def test_budget_refusal_keeps_the_chat_queued(scout: ScoutDatabase, db: Database) -> None:
    """Being refused by the budget is not a failed join.

    Nothing reached Telegram, so the chat keeps its place and its attempt
    count instead of being burned.
    """
    client = FakeJoins()
    worker = _worker(scout, db, client, policy={ActionKind.JOIN: BudgetRule(per_day=1)})
    await scout.enqueue_join(chat_ref="first", watcher_name="danang-signal")
    await scout.enqueue_join(chat_ref="second", watcher_name="danang-signal")

    assert await worker.run_once()
    assert not await worker.run_once()

    queue = {entry.chat_ref: entry for entry in await scout.join_queue()}
    assert queue["second"].state is JoinQueueState.PENDING
    assert queue["second"].attempts == 0
    assert len(client.joined) == 1


async def test_deferred_chat_is_not_retried_immediately(scout: ScoutDatabase, db: Database) -> None:
    """A deferred entry waits out its delay before coming back."""
    client = FakeJoins()
    worker = _worker(scout, db, client, policy={ActionKind.JOIN: BudgetRule(per_day=1)})
    await scout.enqueue_join(chat_ref="first")
    await scout.enqueue_join(chat_ref="second")

    await worker.run_once()
    await worker.run_once()

    assert await scout.next_queued_join() is None


async def test_moderated_chat_is_recorded_as_requested(scout: ScoutDatabase, db: Database) -> None:
    """A pending admin approval is not membership and starts no archive."""
    client = FakeJoins(error=errors.InviteRequestSentError)
    worker = _worker(scout, db, client)
    await scout.enqueue_join(chat_ref="moderated", watcher_name="danang-signal")

    assert await worker.run_once()

    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.REQUESTED
    assert await scout.backfill_targets() == []
    assert await db.observation_snapshot() == {}


async def test_dead_username_fails_without_retrying_forever(
    scout: ScoutDatabase, db: Database
) -> None:
    """An unresolvable chat leaves the queue instead of blocking it."""
    client = FakeJoins()
    worker = _worker(scout, db, client, entity=None)
    await scout.enqueue_join(chat_ref="gone")

    assert not await worker.run_once()

    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.FAILED
    assert queue[0].last_error == "unresolvable"
    assert await scout.next_queued_join() is None


async def test_spam_limit_stops_reaching_telegram_at_all(
    scout: ScoutDatabase, db: Database
) -> None:
    """After a halt the queue keeps its entries and sends nothing more.

    The account-wide cooldown refuses each later attempt before it becomes a
    request, so the remaining chats are deferred rather than burned.
    """
    client = FakeJoins(error=errors.PeerFloodError)
    worker = _worker(scout, db, client)
    await scout.enqueue_join(chat_ref="first")
    await scout.enqueue_join(chat_ref="second")

    assert not await worker.run_once()
    calls_after_halt = len(client.joined)
    assert not await worker.run_once()

    assert len(client.joined) == calls_after_halt
    queue = {entry.chat_ref: entry for entry in await scout.join_queue()}
    assert queue["first"].state is JoinQueueState.PENDING
    assert queue["second"].state is JoinQueueState.PENDING
    assert await scout.next_queued_join() is None


async def test_chat_without_a_policy_is_only_archived(scout: ScoutDatabase, db: Database) -> None:
    """Joining to read is allowed without turning on alerts."""
    client = FakeJoins()
    worker = _worker(scout, db, client)
    await scout.enqueue_join(chat_ref="@danangevents", target_days=365)

    await worker.run_once()

    snapshot = await db.observation_snapshot()
    assert snapshot[CHAT_ID].mode is ObservationMode.RECON
    assert snapshot[CHAT_ID].watcher_names == ()
    assert (await scout.backfill_targets())[0].target_days == 365


async def test_requeueing_a_joined_chat_does_not_rejoin(scout: ScoutDatabase, db: Database) -> None:
    """The queue is an intent to join once, not to keep trying."""
    client = FakeJoins()
    worker = _worker(scout, db, client)
    await scout.enqueue_join(chat_ref="@danangevents", watcher_name="danang-signal")
    await worker.run_once()

    await scout.enqueue_join(chat_ref="@danangevents", watcher_name="danang-signal")

    assert not await worker.run_once()
    assert client.joined == ["danangevents"]
