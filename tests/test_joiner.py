"""Tests for the paced join queue."""

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from telethon import errors
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.functions.messages import CheckChatInviteRequest, ImportChatInviteRequest
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


INVITE_CHANNEL = Channel(
    id=202,
    title="Далат 🇻🇳 Чат TravelAsk",
    photo=None,
    date=None,
    username=None,
    broadcast=False,
    megagroup=True,
)
INVITE_CHAT_ID = -1000000000202


class FakeInviteJoins:
    """Answers invite imports the way Telegram does: the chat rides in the reply."""

    def __init__(
        self,
        error: type[BaseException] | None = None,
        *,
        already_member: bool = False,
    ) -> None:
        self.error = error
        self.already_member = already_member
        self.imported: list[str] = []
        self.checked: list[str] = []

    async def __call__(self, request: object) -> object:
        if isinstance(request, CheckChatInviteRequest):
            self.checked.append(request.hash)
            return SimpleNamespace(chat=INVITE_CHANNEL)
        assert isinstance(request, ImportChatInviteRequest)
        if self.error is not None:
            raise self.error(request=None)
        if self.already_member:
            raise errors.UserAlreadyParticipantError(request=None)
        self.imported.append(request.hash)
        return SimpleNamespace(chats=[INVITE_CHANNEL], users=[])


def _worker(
    scout: ScoutDatabase,
    db: Database,
    client: object,
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


async def test_an_invite_link_is_joined_through_its_hash(
    scout: ScoutDatabase, db: Database
) -> None:
    """A t.me/+hash reference needs no entity up front: the reply names the chat."""
    client = FakeInviteJoins()
    worker = _worker(scout, db, client, entity=None)
    await scout.enqueue_join(
        chat_ref="https://t.me/+mqaI5aYDQuI5ZWFi",
        label="Далат TravelAsk",
        watcher_name="dalat-signal",
    )

    assert await worker.run_once()

    assert client.imported == ["mqaI5aYDQuI5ZWFi"]
    snapshot = await db.observation_snapshot()
    assert snapshot[INVITE_CHAT_ID].mode is ObservationMode.MONITOR
    assert snapshot[INVITE_CHAT_ID].watcher_names == ("dalat-signal",)
    assert (await scout.backfill_targets())[0].chat_id == INVITE_CHAT_ID
    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.JOINED
    assert queue[0].joined_chat_id == INVITE_CHAT_ID


async def test_an_invite_we_already_used_still_identifies_the_chat(
    scout: ScoutDatabase, db: Database
) -> None:
    """Being a member already is success, and the chat id comes from the invite check."""
    client = FakeInviteJoins(already_member=True)
    worker = _worker(scout, db, client, entity=None)
    await scout.enqueue_join(chat_ref="t.me/+mqaI5aYDQuI5ZWFi")

    assert await worker.run_once()

    assert client.checked == ["mqaI5aYDQuI5ZWFi"]
    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.JOINED
    assert queue[0].joined_chat_id == INVITE_CHAT_ID


@pytest.mark.parametrize("error", [errors.InviteHashExpiredError, errors.InviteHashInvalidError])
async def test_a_dead_invite_fails_without_retrying(
    scout: ScoutDatabase, db: Database, error: type[BaseException]
) -> None:
    """An expired or mistyped hash will never open; burning joins on it is waste."""
    client = FakeInviteJoins(error=error)
    worker = _worker(scout, db, client, entity=None)
    await scout.enqueue_join(chat_ref="t.me/+deadhash")

    assert await worker.run_once()

    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.FAILED
    assert queue[0].last_error == "invite_invalid"
    assert await scout.backfill_targets() == []
    assert not await worker.run_once()


async def test_an_invite_awaiting_approval_is_not_membership(
    scout: ScoutDatabase, db: Database
) -> None:
    """A moderated invite is recorded as requested, exactly like a moderated username."""
    client = FakeInviteJoins(error=errors.InviteRequestSentError)
    worker = _worker(scout, db, client, entity=None)
    await scout.enqueue_join(chat_ref="t.me/+moderated", watcher_name="dalat-signal")

    assert await worker.run_once()

    assert (await scout.join_queue())[0].state is JoinQueueState.REQUESTED
    assert await db.observation_snapshot() == {}


def _worker_with_dialogs(
    scout: ScoutDatabase,
    db: Database,
    client: object,
    dialogs: dict[str, int],
    *,
    entity: object | None = CHANNEL,
) -> JoinWorker:
    # No pacing: these tests exercise outcome handling, not the budget.
    governor = TelegramActionGovernor(
        scout=scout,
        policy={ActionKind.JOIN: BudgetRule(), ActionKind.INVITE_CHECK: BudgetRule()},
    )

    async def resolve(ref: str) -> object | None:
        return entity

    async def dialog_index() -> dict[str, int]:
        return dialogs

    return JoinWorker(
        scout=scout,
        db=db,
        crawler=TelegramCrawler(client=client, governor=governor),
        resolve_entity=resolve,
        poll_seconds=0.01,
        dialog_index=dialog_index,
    )


async def test_an_interrupted_join_is_reconciled_not_buried(
    scout: ScoutDatabase, db: Database
) -> None:
    """A process death mid-join leaves an unsettled reservation. The retry
    must not settle FAILED blindly: the dialog list knows the truth."""
    await scout.enqueue_join(chat_ref="@danangevents", watcher_name=None)
    # The first life reserved the slot and died before Telegram answered.
    await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="queue:join:danangevents:0",
    )

    client = FakeJoins()
    worker = _worker_with_dialogs(scout, db, client, {"danangevents": CHAT_ID})
    assert await worker.run_once() is False
    queue = await scout.join_queue()
    # Not FAILED: parked for reconciliation.
    assert queue[0].state is JoinQueueState.PENDING
    assert queue[0].last_error == "already_attempted"
    assert client.joined == []

    # Reconciliation finds the chat among the dialogs: the join DID land.
    async with scout.conn.execute(
        "UPDATE join_queue SET not_before = NULL WHERE chat_ref = 'danangevents'"
    ):
        pass
    await scout.conn.commit()
    resolved = await worker.reconcile()

    assert resolved == 1
    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.JOINED
    assert queue[0].joined_chat_id == CHAT_ID


async def test_an_interrupted_join_absent_from_dialogs_is_retried_fresh(
    scout: ScoutDatabase, db: Database
) -> None:
    """Positively absent from the dialogs means the call never went through;
    the retry runs under a fresh idempotency key and actually joins."""
    await scout.enqueue_join(chat_ref="@danangevents")
    await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="queue:join:danangevents:0",
    )

    client = FakeJoins()
    worker = _worker_with_dialogs(scout, db, client, {})
    assert await worker.run_once() is False

    async with scout.conn.execute(
        "UPDATE join_queue SET not_before = NULL WHERE chat_ref = 'danangevents'"
    ):
        pass
    await scout.conn.commit()
    assert await worker.reconcile() == 1
    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.PENDING
    assert queue[0].attempts == 1

    # The fresh attempt uses a new key and reaches Telegram this time.
    assert await worker.run_once() is True
    assert client.joined == ["danangevents"]
    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.JOINED


async def test_an_approved_admin_request_is_activated_by_reconciliation(
    scout: ScoutDatabase, db: Database
) -> None:
    """An admin approving a week-old request must not go unnoticed forever."""
    client = FakeJoins(error=errors.InviteRequestSentError)
    worker = _worker_with_dialogs(scout, db, client, {"danangevents": CHAT_ID})
    await scout.enqueue_join(chat_ref="@danangevents", watcher_name="danang-signal")
    assert await worker.run_once() is True
    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.REQUESTED

    resolved = await worker.reconcile()

    assert resolved == 1
    queue = await scout.join_queue()
    assert queue[0].state is JoinQueueState.JOINED
    snapshot = await db.observation_snapshot()
    assert snapshot[CHAT_ID].mode is ObservationMode.MONITOR


async def test_a_flood_wait_spends_the_attempt_so_the_retry_is_real(
    scout: ScoutDatabase, db: Database
) -> None:
    """A FloodWait reached Telegram; retrying under the same key would replay
    the settled reservation and wedge the row forever."""

    class FloodThenJoin:
        def __init__(self) -> None:
            self.calls = 0
            self.joined: list[str] = []

        async def __call__(self, request: object) -> object:
            assert isinstance(request, JoinChannelRequest)
            self.calls += 1
            if self.calls == 1:
                raise errors.FloodWaitError(request=None, capture=1)
            self.joined.append(str(getattr(request.channel, "username", request.channel)))
            return SimpleNamespace(chats=[], users=[])

    client = FloodThenJoin()
    worker = _worker_with_dialogs(scout, db, client, {})
    await scout.enqueue_join(chat_ref="@danangevents")

    assert await worker.run_once() is False
    queue = await scout.join_queue()
    assert queue[0].attempts == 1

    async with scout.conn.execute(
        "UPDATE join_queue SET not_before = NULL WHERE chat_ref = 'danangevents'"
    ):
        pass
    # The flood cooldown has since expired.
    await scout.conn.execute("DELETE FROM account_cooldowns")
    await scout.conn.commit()

    assert await worker.run_once() is True
    assert client.joined == ["danangevents"]


async def test_requeueing_a_failed_chat_revives_it(scout: ScoutDatabase) -> None:
    """A terminal failure plus a fresh instruction is a retry, not a no-op."""
    await scout.enqueue_join(chat_ref="@danangevents")
    await scout.settle_queued_join(
        "danangevents", state=JoinQueueState.FAILED, error="unresolvable"
    )

    await scout.enqueue_join(chat_ref="@danangevents")

    queued = await scout.next_queued_join()
    assert queued is not None
    assert queued.state is JoinQueueState.PENDING
    assert queued.attempts == 2  # settle counted one, the revival bumped past it


async def test_a_flooded_invite_check_moves_to_a_fresh_key(
    scout: ScoutDatabase, db: Database
) -> None:
    """A FloodWait on the re-check settles its reservation; re-using the key
    would replay it forever and the invite request would wedge."""

    class FloodThenAlready:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, request: object) -> object:
            assert isinstance(request, CheckChatInviteRequest)
            self.calls += 1
            if self.calls == 1:
                raise errors.FloodWaitError(request=None, capture=1)
            return SimpleNamespace(chat=INVITE_CHANNEL)  # no request_needed

    # Manufacture the ChatInviteAlready shape the crawler looks for.
    class ChatInviteAlready(SimpleNamespace):
        pass

    class FloodThenMember:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, request: object) -> object:
            assert isinstance(request, CheckChatInviteRequest)
            self.calls += 1
            if self.calls == 1:
                raise errors.FloodWaitError(request=None, capture=1)
            return ChatInviteAlready(chat=INVITE_CHANNEL, request_needed=False)

    client = FloodThenMember()
    worker = _worker_with_dialogs(scout, db, client, {})
    await scout.enqueue_join(chat_ref="https://t.me/+SecretHash123")
    ref = (await scout.join_queue())[0].chat_ref
    await scout.settle_queued_join(
        ref, state=JoinQueueState.REQUESTED, error="awaiting_admin_approval"
    )

    # First reconcile: flood — the attempt must be spent regardless.
    await worker.reconcile()
    queue = await scout.join_queue()
    assert queue[0].attempts == 2  # settle counted 1, the flooded check bumped past it

    async with scout.conn.execute(
        "UPDATE join_queue SET not_before = NULL WHERE chat_ref = ?", (ref,)
    ):
        pass
    await scout.conn.execute("DELETE FROM account_cooldowns")
    await scout.conn.commit()

    # Second reconcile runs under a fresh key and reaches Telegram.
    resolved = await worker.reconcile()
    assert client.calls == 2
    assert resolved == 1
    assert (await scout.join_queue())[0].state is JoinQueueState.JOINED
