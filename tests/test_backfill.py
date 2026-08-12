"""Tests for the background history archive."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from telethon import errors
from telethon.tl.functions.messages import GetHistoryRequest

from pipeline.backfill import BackfillWorker
from pipeline.crawler import HISTORY_PAGE_SIZE, TelegramCrawler
from pipeline.governor import TelegramActionGovernor
from pipeline.recon_models import ActionKind, BackfillState, BudgetRule
from storage.scout import ScoutDatabase

CHAT = -1000000000101


async def _peer_from_cache(chat_id: int) -> object:
    """Stand in for the client's entity cache."""
    return SimpleNamespace(chat_id=chat_id)


async def _no_peer(chat_id: int) -> object | None:
    """A chat the client cannot resolve yet."""
    return None


@pytest.fixture
async def scout(tmp_path: Path) -> AsyncIterator[ScoutDatabase]:
    """Reconnaissance state in a temporary file."""
    database = ScoutDatabase(tmp_path / "scout.db")
    await database.connect()
    yield database
    await database.close()


class FakeHistory:
    """Serves pages of history walking backwards from a fixed corpus."""

    def __init__(self, *, total: int, page_size: int = HISTORY_PAGE_SIZE) -> None:
        self.total = total
        self.page_size = page_size
        self.calls: list[int] = []
        self.error: BaseException | None = None

    async def __call__(self, request: object) -> object:
        assert isinstance(request, GetHistoryRequest)
        if self.error is not None:
            raise self.error
        # Record what was actually sent; 0 means "start from the newest".
        self.calls.append(request.offset_id)
        offset = request.offset_id or (self.total + 1)
        ids = list(range(offset - 1, 0, -1))[: self.page_size]
        return SimpleNamespace(
            messages=[
                SimpleNamespace(
                    id=index,
                    message=f"message {index}",
                    date=datetime(2026, 7, 1, tzinfo=UTC),
                    entities=None,
                    from_id=None,
                    fwd_from=None,
                )
                for index in ids
            ]
        )


class ShortPageThenMore(FakeHistory):
    """A chat that hands back one undersized page while more history remains.

    Models the sparse-id-range case: the requested window covers a stretch
    where most ids were deleted, so Telegram fills the page with what exists
    rather than reaching further back for the rest.
    """

    def __init__(self, *, total: int, short_at_offset: int, short_size: int) -> None:
        super().__init__(total=total)
        self.short_at_offset = short_at_offset
        self.short_size = short_size

    async def __call__(self, request: object) -> object:
        assert isinstance(request, GetHistoryRequest)
        original = self.page_size
        if request.offset_id == self.short_at_offset:
            self.page_size = self.short_size
        try:
            return await super().__call__(request)
        finally:
            self.page_size = original


def _worker(
    scout: ScoutDatabase,
    client: object,
    *,
    is_idle: bool = True,
    policy: dict[ActionKind, BudgetRule] | None = None,
) -> BackfillWorker:
    governor = TelegramActionGovernor(scout=scout, policy=policy)
    return BackfillWorker(
        scout=scout,
        crawler=TelegramCrawler(client=client, governor=governor),
        resolve_peer=_peer_from_cache,
        is_idle=lambda: is_idle,
        page_interval_seconds=0.01,
    )


async def test_target_is_registered_with_a_horizon(scout: ScoutDatabase) -> None:
    """A target is a standing intent measured in days."""
    await scout.add_backfill_target(chat_id=CHAT, label="danang", target_days=730)

    target = await scout.next_backfill_target()

    assert target is not None
    assert target.chat_id == CHAT
    assert target.target_days == 730
    assert target.is_active


async def test_raising_the_horizon_reopens_a_finished_target(scout: ScoutDatabase) -> None:
    """Asking for two years after finishing one must resume, not restart."""
    await scout.add_backfill_target(chat_id=CHAT, target_days=365)
    await scout.record_backfill_page(
        CHAT,
        stored=100,
        oldest_message_id=50,
        oldest_message_date=None,
        finished=BackfillState.COMPLETE,
    )
    assert await scout.next_backfill_target() is None

    await scout.add_backfill_target(chat_id=CHAT, target_days=730)

    target = await scout.next_backfill_target()
    assert target is not None
    assert target.target_days == 730
    assert target.oldest_message_id == 50


async def test_lowering_the_horizon_does_not_reopen(scout: ScoutDatabase) -> None:
    """A smaller ask must not undo a bigger archive."""
    await scout.add_backfill_target(chat_id=CHAT, target_days=730)
    await scout.record_backfill_page(
        CHAT,
        stored=10,
        oldest_message_id=5,
        oldest_message_date=None,
        finished=BackfillState.COMPLETE,
    )

    await scout.add_backfill_target(chat_id=CHAT, target_days=30)

    assert await scout.next_backfill_target() is None
    targets = await scout.backfill_targets()
    assert targets[0].target_days == 730


async def test_pages_advance_the_cursor(scout: ScoutDatabase) -> None:
    """Each page walks further back and is resumable from the cursor."""
    client = FakeHistory(total=250)
    worker = _worker(scout, client)
    await scout.add_backfill_target(chat_id=CHAT, label="danang", target_days=730)

    assert await worker.run_once()
    assert await worker.run_once()

    target = await scout.next_backfill_target()
    assert target is not None
    assert target.pages_done == 2
    assert target.messages_stored == 200
    assert target.oldest_message_id == 51
    assert client.calls == [0, 151]


async def test_target_is_exhausted_when_history_runs_out(scout: ScoutDatabase) -> None:
    """Only an empty page ends the walk, and it ends it as exhausted."""
    client = FakeHistory(total=30)
    worker = _worker(scout, client)
    await scout.add_backfill_target(chat_id=CHAT, target_days=730)

    assert await worker.run_once()
    # The first page was short but not empty, so it settles nothing.
    assert await scout.next_backfill_target() is not None
    assert await worker.run_once()

    assert await scout.next_backfill_target() is None
    targets = await scout.backfill_targets()
    assert targets[0].state is BackfillState.EXHAUSTED
    assert targets[0].messages_stored == 30


async def test_a_short_page_mid_history_does_not_end_the_walk(scout: ScoutDatabase) -> None:
    """Telegram returns short pages mid-chat; reading one as the end truncates.

    Observed in production: `Дананг Объявления` stopped at message 4879 of
    6939 and was recorded complete, because a page came back with 66 of the
    100 requested and that was taken for the bottom of the chat.
    """
    client = ShortPageThenMore(total=250, short_at_offset=151, short_size=40)
    worker = _worker(scout, client)
    await scout.add_backfill_target(chat_id=CHAT, target_days=730)

    for _ in range(10):
        if not await worker.run_once():
            break

    targets = await scout.backfill_targets()
    assert targets[0].state is BackfillState.EXHAUSTED
    assert targets[0].messages_stored == 250, "the walk stopped at the short page"


async def test_least_advanced_target_goes_first(scout: ScoutDatabase) -> None:
    """Comparable history across chats beats one deep hole."""
    client = FakeHistory(total=1000)
    worker = _worker(scout, client)
    await scout.add_backfill_target(chat_id=-1, label="a", target_days=730)
    await scout.add_backfill_target(chat_id=-2, label="b", target_days=730)

    first = await scout.next_backfill_target()
    assert first is not None
    await worker.run_once()
    second = await scout.next_backfill_target()

    assert second is not None
    assert second.chat_id != first.chat_id


async def test_flood_wait_defers_instead_of_failing(scout: ScoutDatabase) -> None:
    """Being told to wait is the normal case, not an error."""
    client = FakeHistory(total=1000)
    error = errors.FloodWaitError(request=None)
    error.seconds = 120
    client.error = error
    worker = _worker(scout, client)
    await scout.add_backfill_target(chat_id=CHAT, target_days=730)

    assert not await worker.run_once()

    assert await scout.next_backfill_target() is None
    targets = await scout.backfill_targets()
    assert targets[0].state is BackfillState.PENDING
    assert targets[0].last_error is not None


async def test_exhausted_budget_defers_the_target(scout: ScoutDatabase) -> None:
    """The archive yields to whatever else the account needs to do."""
    client = FakeHistory(total=1000)
    worker = _worker(
        scout,
        client,
        policy={ActionKind.HISTORY_PAGE: BudgetRule(per_day=1)},
    )
    await scout.add_backfill_target(chat_id=CHAT, target_days=730)

    assert await worker.run_once()
    assert not await worker.run_once()

    targets = await scout.backfill_targets()
    assert targets[0].state is BackfillState.PENDING
    assert targets[0].pages_done == 1


async def test_unresolvable_peer_is_deferred_not_dropped(scout: ScoutDatabase) -> None:
    """A chat missing from the client cache is retried later."""
    client = FakeHistory(total=100)
    governor = TelegramActionGovernor(scout=scout)
    worker = BackfillWorker(
        scout=scout,
        crawler=TelegramCrawler(client=client, governor=governor),
        resolve_peer=_no_peer,
        page_interval_seconds=0.01,
    )
    await scout.add_backfill_target(chat_id=CHAT, target_days=730)

    assert not await worker.run_once()

    targets = await scout.backfill_targets()
    assert targets[0].state is BackfillState.PENDING
    assert targets[0].last_error == "peer_unresolved"
    assert client.calls == []


async def test_busy_daemon_is_left_alone(scout: ScoutDatabase) -> None:
    """Live messages take priority over the archive."""
    client = FakeHistory(total=1000)
    worker = _worker(scout, client, is_idle=False)
    await scout.add_backfill_target(chat_id=CHAT, target_days=730)
    shutdown = asyncio.Event()

    task = asyncio.create_task(worker.run_forever(shutdown))
    await asyncio.sleep(0.05)
    shutdown.set()
    await task

    assert client.calls == []


async def test_worker_survives_a_failing_cycle(scout: ScoutDatabase) -> None:
    """A crawl problem must never take the monitor down with it."""
    client = FakeHistory(total=1000)
    client.error = RuntimeError("telegram exploded")
    worker = _worker(scout, client)
    await scout.add_backfill_target(chat_id=CHAT, target_days=730)
    shutdown = asyncio.Event()

    task = asyncio.create_task(worker.run_forever(shutdown))
    await asyncio.sleep(0.05)
    shutdown.set()
    await task

    assert task.done()
    assert task.exception() is None


async def test_worker_stops_promptly_on_shutdown(scout: ScoutDatabase) -> None:
    """Shutdown must not wait out the idle poll interval."""
    client = FakeHistory(total=1000)
    worker = _worker(scout, client)
    shutdown = asyncio.Event()

    task = asyncio.create_task(worker.run_forever(shutdown))
    await asyncio.sleep(0.02)
    shutdown.set()
    await asyncio.wait_for(task, timeout=2.0)

    assert task.done()


class NamedHistory(FakeHistory):
    """A history page that carries its senders the way Telegram does."""

    async def __call__(self, request: object) -> object:
        page = await super().__call__(request)
        for index, message in enumerate(page.messages):
            message.from_id = SimpleNamespace(user_id=500 + (index % 2))
        page.users = [
            SimpleNamespace(id=500, first_name="Аня", last_name="Организатор", username="anya"),
            SimpleNamespace(id=501, first_name=None, last_name=None, username="quiet_one"),
        ]
        return page


async def test_backfilled_messages_carry_their_senders_name(scout: ScoutDatabase) -> None:
    """Names ride in the page's side list, not on the messages themselves."""
    worker = _worker(scout, NamedHistory(total=4))
    await scout.add_backfill_target(chat_id=CHAT, target_days=730)

    await worker.run_once()

    rows = await scout.conn.execute_fetchall(
        "SELECT sender_id, sender_name FROM scout_messages ORDER BY telegram_msg_id"
    )
    names = {row[0]: row[1] for row in rows}
    assert names[500] == "@anya (Аня Организатор)"
    # No first or last name: the username is the only thing left to show.
    assert names[501] == "@quiet_one"
