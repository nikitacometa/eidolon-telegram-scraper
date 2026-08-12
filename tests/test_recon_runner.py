"""End-to-end tests for pipeline/recon.py against a fake Telegram."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from telethon import errors
from telethon.tl.functions.channels import (
    JoinChannelRequest,
    SearchPostsRequest,
)
from telethon.tl.functions.contacts import SearchRequest as ContactsSearchRequest
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import Channel, PeerChannel

from pipeline.crawler import TelegramCrawler
from pipeline.discovery import TelegramDiscovery
from pipeline.governor import TelegramActionGovernor
from pipeline.models import ObservationMode
from pipeline.recon import ReconRunner
from pipeline.recon_models import (
    ActionKind,
    BudgetRule,
    CandidateState,
    ChatMembership,
    JobRequest,
    ReconJobStatus,
)
from storage.db import Database
from storage.scout import ScoutDatabase

DANANG_JOB = JobRequest(
    idempotency_key="recon-danang",
    topic="housing rent",
    location="Da Nang, Vietnam",
    max_waves=2,
)


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


def _channel(channel_id: int, username: str, title: str, participants: int = 5000) -> Channel:
    return Channel(
        id=channel_id,
        title=title,
        photo=None,
        date=None,
        username=username,
        broadcast=False,
        megagroup=True,
        participants_count=participants,
    )


# Recon filters history against a sliding lookback window computed from now(), so a fixture
# pinned to an absolute date silently expires once that date falls out of the window.
# One value per run keeps the fixture deterministic within a run while staying inside it.
FRESH_MESSAGE_DATE = datetime.now(UTC) - timedelta(days=1)


def _message(message_id: int, text: str, chat_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        message=text,
        date=FRESH_MESSAGE_DATE,
        entities=None,
        from_id=SimpleNamespace(user_id=1000 + message_id),
        fwd_from=None,
        peer_id=PeerChannel(chat_id),
    )


class FakeTelegram:
    """A Telegram that answers the four calls reconnaissance makes."""

    def __init__(
        self,
        *,
        search_results: list[Channel] | None = None,
        history: dict[str, list[SimpleNamespace]] | None = None,
        join_error: type[BaseException] | None = None,
    ) -> None:
        self.search_results = search_results or []
        self.history = history or {}
        self.join_error = join_error
        self.joined: list[str] = []
        self.history_calls = 0
        self.entities = {
            channel.username: channel for channel in (search_results or []) if channel.username
        }

    async def get_entity(self, reference: str) -> Channel:
        entity = self.entities.get(str(reference).lstrip("@"))
        if entity is None:
            raise errors.UsernameNotOccupiedError(request=None)
        return entity

    async def get_input_entity(self, reference: str) -> SimpleNamespace:
        name = str(reference).lstrip("@")
        if name not in self.entities:
            raise errors.UsernameNotOccupiedError(request=None)
        return SimpleNamespace(username=name)

    async def __call__(self, request: object) -> object:
        if isinstance(request, SearchPostsRequest):
            return SimpleNamespace(
                chats=list(self.search_results),
                messages=[
                    SimpleNamespace(id=index, peer_id=PeerChannel(channel.id))
                    for index, channel in enumerate(self.search_results, start=1)
                ],
                users=[],
            )
        if isinstance(request, ContactsSearchRequest):
            return SimpleNamespace(chats=[], users=[], messages=[])
        if isinstance(request, JoinChannelRequest):
            name = str(getattr(request.channel, "username", request.channel))
            if self.join_error is not None:
                raise self.join_error(request=None)
            self.joined.append(name)
            return SimpleNamespace(chats=[], users=[])
        if isinstance(request, GetHistoryRequest):
            self.history_calls += 1
            name = str(getattr(request.peer, "username", request.peer))
            messages = self.history.get(name, [])
            # Real Telegram pages backwards from offset_id. Without this the fake replays the
            # same page forever, so "history ran out" can never be observed.
            if request.offset_id:
                messages = [message for message in messages if message.id < request.offset_id]
            return SimpleNamespace(messages=messages)
        raise AssertionError(f"unexpected request: {type(request).__name__}")


def _runner(
    scout: ScoutDatabase,
    db: Database,
    client: FakeTelegram,
    *,
    policy: dict[ActionKind, BudgetRule] | None = None,
    pages: int = 5,
) -> ReconRunner:
    governor = TelegramActionGovernor(scout=scout, policy=policy)
    return ReconRunner(
        scout=scout,
        db=db,
        client=client,
        governor=governor,
        discovery=TelegramDiscovery(client=client, governor=governor),
        crawler=TelegramCrawler(client=client, governor=governor),
        backfill_pages_per_chat=pages,
    )


async def test_a_topic_becomes_joined_chats_with_history(
    scout: ScoutDatabase, db: Database
) -> None:
    """The whole point: one phrase in, watched chats and their history out."""
    housing = _channel(101, "danang_housing", "Da Nang Housing and Rent")
    client = FakeTelegram(
        search_results=[housing],
        history={
            "danang_housing": [
                _message(10, "Villa for rent, see t.me/danang_villas", 101),
                _message(9, "Room available in An Thuong", 101),
            ]
        },
    )
    job = await scout.create_job(DANANG_JOB)

    report = await _runner(scout, db, client, pages=1).run(job)

    assert report.status is ReconJobStatus.COMPLETED
    assert client.joined == ["danang_housing"]
    assert [finding.chat_ref for finding in report.joined] == ["danang_housing"]
    assert report.messages_stored == 2
    assert await scout.message_count(-1000000000101) == 2


async def test_joined_chat_is_registered_for_live_capture(
    scout: ScoutDatabase, db: Database
) -> None:
    """Capture must start at the join, not at the promotion.

    Anything posted between the two exists only as a live update; history will
    not serve it again later.
    """
    housing = _channel(102, "danang_rent", "Da Nang Rent")
    client = FakeTelegram(search_results=[housing], history={"danang_rent": []})
    job = await scout.create_job(DANANG_JOB)

    await _runner(scout, db, client, pages=1).run(job)

    snapshot = await db.observation_snapshot()
    assert -1000000000102 in snapshot
    assert snapshot[-1000000000102].mode is ObservationMode.RECON
    assert snapshot[-1000000000102].job_id == job.id


async def test_links_in_history_become_next_wave_candidates(
    scout: ScoutDatabase, db: Database
) -> None:
    """Snowball: a chat found by reading another chat."""
    housing = _channel(103, "danang_housing", "Da Nang Housing")
    client = FakeTelegram(
        search_results=[housing],
        history={
            "danang_housing": [
                _message(5, "also join t.me/danang_villas and @danang_food", 103),
            ]
        },
    )
    job = await scout.create_job(DANANG_JOB)

    await _runner(scout, db, client, pages=1).run(job)

    candidates = await scout.candidates_for_job(job.id)
    refs = set()
    for candidate in candidates:
        chat = await scout.get_chat(candidate.chat_uuid)
        assert chat is not None
        if chat.username:
            refs.add(chat.username)
    assert {"danang_villas", "danang_food"} <= refs


async def test_join_request_awaiting_approval_is_not_treated_as_membership(
    scout: ScoutDatabase, db: Database
) -> None:
    """A moderated chat answers with a request, not with access."""
    moderated = _channel(104, "danang_moderated", "Da Nang Housing Moderated")
    client = FakeTelegram(
        search_results=[moderated],
        join_error=errors.InviteRequestSentError,
    )
    job = await scout.create_job(DANANG_JOB)

    report = await _runner(scout, db, client, pages=1).run(job)

    assert report.joined == []
    assert client.history_calls == 0
    candidates = await scout.candidates_for_job(job.id)
    assert candidates[0].state is CandidateState.JOIN_REQUESTED
    chat = await scout.get_chat(candidates[0].chat_uuid)
    assert chat is not None
    assert chat.membership is ChatMembership.REQUESTED
    assert chat.joined_at is None


async def test_irrelevant_chats_are_never_joined(scout: ScoutDatabase, db: Database) -> None:
    """Search returns mostly noise; the account pays for every join."""
    junk = _channel(105, "danang_pump", "Da Nang PUMP 100x signals")
    elsewhere = _channel(106, "bali_housing", "Bali Housing")
    client = FakeTelegram(search_results=[junk, elsewhere])
    job = await scout.create_job(DANANG_JOB)

    report = await _runner(scout, db, client, pages=1).run(job)

    assert client.joined == []
    assert report.rejected == 2
    assert report.joined == []


async def test_exhausted_join_budget_ends_the_job_partially(
    scout: ScoutDatabase, db: Database
) -> None:
    """Running out of budget is a partial result, not a failure."""
    first = _channel(107, "danang_housing", "Da Nang Housing")
    second = _channel(108, "danang_rent", "Da Nang Rent Apartments")
    client = FakeTelegram(
        search_results=[first, second],
        history={"danang_housing": [], "danang_rent": []},
    )
    job = await scout.create_job(DANANG_JOB)

    report = await _runner(
        scout,
        db,
        client,
        policy={ActionKind.JOIN: BudgetRule(per_day=1)},
        pages=1,
    ).run(job)

    assert len(client.joined) == 1
    assert report.status is ReconJobStatus.COMPLETED_PARTIAL
    assert report.stop_reason == "budget exhausted"


async def test_a_halt_stops_the_job_immediately(scout: ScoutDatabase, db: Database) -> None:
    """A spam limitation must not be worked around by continuing."""
    housing = _channel(109, "danang_housing", "Da Nang Housing")
    client = FakeTelegram(search_results=[housing], join_error=errors.PeerFloodError)
    job = await scout.create_job(DANANG_JOB)

    report = await _runner(scout, db, client, pages=1).run(job)

    assert report.status is ReconJobStatus.COMPLETED_PARTIAL
    assert report.stop_reason.startswith("halted")


async def test_backfill_stops_when_history_runs_out(scout: ScoutDatabase, db: Database) -> None:
    """A short page means there is nothing older to ask for."""
    housing = _channel(110, "danang_housing", "Da Nang Housing")
    client = FakeTelegram(
        search_results=[housing],
        history={"danang_housing": [_message(3, "hello", 110)]},
    )
    job = await scout.create_job(DANANG_JOB)

    await _runner(scout, db, client, pages=10).run(job)

    assert client.history_calls == 1


async def test_rerunning_a_job_does_not_rejoin(scout: ScoutDatabase, db: Database) -> None:
    """Replaying a job must not spend a second join on the same chat."""
    housing = _channel(111, "danang_housing", "Da Nang Housing")
    client = FakeTelegram(search_results=[housing], history={"danang_housing": []})
    job = await scout.create_job(DANANG_JOB)
    runner = _runner(scout, db, client, pages=1)

    await runner.run(job)
    reloaded = await scout.get_job(job.id)
    assert reloaded is not None

    assert await scout.budget_usage(account_id="owner-primary", kind=ActionKind.JOIN) == (1, 1)
    assert client.joined == ["danang_housing"]


async def test_platform_scam_label_survives_storage_and_blocks_the_join(
    scout: ScoutDatabase, db: Database
) -> None:
    """Scoring reads the stored record, so risk labels must be persisted.

    A chat can name the city, look busy, and be flagged by Telegram itself.
    If the flag is dropped between the search response and the score, the
    account joins a chat Telegram already called a scam.
    """
    flagged = Channel(
        id=112,
        title="Da Nang Housing Rent",
        photo=None,
        date=None,
        username="danang_housing_scam",
        broadcast=False,
        megagroup=True,
        participants_count=9000,
        scam=True,
    )
    client = FakeTelegram(search_results=[flagged])
    job = await scout.create_job(DANANG_JOB)

    report = await _runner(scout, db, client, pages=1).run(job)

    assert client.joined == []
    assert report.rejected == 1
    candidates = await scout.candidates_for_job(job.id)
    assert "scam" in candidates[0].risk_flags
    stored = await scout.get_chat(candidates[0].chat_uuid)
    assert stored is not None
    assert stored.risk_flags == ("scam",)
    assert stored.participants == 9000


async def test_join_attempt_cap_is_enforced_during_the_wave(
    scout: ScoutDatabase, db: Database
) -> None:
    """The cap has to bite while joining, not in the report afterwards."""
    channels = [
        _channel(200 + index, f"danang_housing_{index}", f"Da Nang Housing {index}")
        for index in range(4)
    ]
    client = FakeTelegram(
        search_results=channels,
        history={channel.username: [] for channel in channels},
    )
    job = await scout.create_job(
        JobRequest(
            idempotency_key="recon-capped",
            topic="housing rent",
            location="Da Nang, Vietnam",
            max_join_attempts=2,
        )
    )

    report = await _runner(
        scout,
        db,
        client,
        policy={ActionKind.JOIN: BudgetRule(per_hour=None, per_day=10)},
        pages=1,
    ).run(job)

    assert len(client.joined) == 2
    assert report.stop_reason == "join attempt limit reached"


async def test_history_older_than_the_lookback_window_is_not_stored(
    scout: ScoutDatabase, db: Database
) -> None:
    """A job that asked for a week must not quietly archive a year."""
    housing = _channel(210, "danang_housing", "Da Nang Housing")
    recent = _message(20, "recent listing", 210)
    recent.date = datetime.now(UTC)
    ancient = _message(19, "listing from last year", 210)
    ancient.date = datetime(2024, 1, 1, tzinfo=UTC)
    client = FakeTelegram(search_results=[housing], history={"danang_housing": [recent, ancient]})
    job = await scout.create_job(
        JobRequest(
            idempotency_key="recon-lookback",
            topic="housing rent",
            location="Da Nang, Vietnam",
            lookback_days=7,
        )
    )

    report = await _runner(scout, db, client, pages=5).run(job)

    assert report.messages_stored == 1
    assert client.history_calls == 1


async def test_discovery_only_run_scores_without_touching_anything(
    scout: ScoutDatabase, db: Database
) -> None:
    """A read-only pass must survey the ground without stepping on it.

    Nothing the account does becomes visible to a chat admin, so this is the
    safe way to see what a topic actually returns before spending standing.
    """
    good = _channel(300, "danang_housing", "Da Nang Housing and Rent")
    junk = _channel(301, "danang_pump", "Da Nang PUMP 100x")
    client = FakeTelegram(search_results=[good, junk])
    job = await scout.create_job(
        JobRequest(
            idempotency_key="recon-survey",
            topic="housing rent",
            location="Da Nang, Vietnam",
            max_join_attempts=0,
        )
    )

    report = await _runner(scout, db, client, pages=1).run(job)

    assert client.joined == []
    assert client.history_calls == 0
    assert [finding.chat_ref for finding in report.recommended] == ["danang_housing"]
    assert report.rejected == 1
    assert await db.observation_snapshot() == {}
