"""Tests for pipeline/governor.py — the single gate to Telegram."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from telethon import errors

from pipeline.governor import (
    ActionStatus,
    TelegramActionGovernor,
)
from pipeline.recon_models import ActionKind, ActionOutcome, BudgetRule
from storage.scout import ScoutDatabase


@pytest.fixture
async def scout(tmp_path: Path) -> AsyncIterator[ScoutDatabase]:
    """Open a scout database backed by a temporary file."""
    database = ScoutDatabase(tmp_path / "scout.db")
    await database.connect()
    yield database
    await database.close()


def _flood(seconds: int) -> errors.FloodWaitError:
    """Build a FloodWaitError carrying a specific wait."""
    error = errors.FloodWaitError(request=None)
    error.seconds = seconds
    return error


async def _action_row(scout: ScoutDatabase, key: str) -> dict[str, object]:
    cursor = await scout.conn.execute(
        "SELECT outcome, duration_ms, flood_wait_seconds, error_code"
        " FROM telegram_actions WHERE idempotency_key = ?",
        (key,),
    )
    row = await cursor.fetchone()
    assert row is not None
    return dict(row)


async def test_successful_call_returns_the_value(scout: ScoutDatabase) -> None:
    """The happy path passes the result through and records success."""
    governor = TelegramActionGovernor(scout=scout)

    async def call() -> str:
        return "found"

    result = await governor.run(ActionKind.HASHTAG_SEARCH, "search-1", call)

    assert result.ok
    assert result.value == "found"
    row = await _action_row(scout, "search-1")
    assert row["outcome"] == ActionOutcome.SUCCEEDED.value
    assert float(row["duration_ms"] or 0) >= 0


async def test_exhausted_budget_never_reaches_telegram(scout: ScoutDatabase) -> None:
    """A refused action must not call out at all."""
    governor = TelegramActionGovernor(
        scout=scout,
        policy={ActionKind.JOIN: BudgetRule(per_day=1)},
    )
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        return "joined"

    first = await governor.run(ActionKind.JOIN, "join-1", call)
    second = await governor.run(ActionKind.JOIN, "join-2", call)

    assert first.ok
    assert second.status is ActionStatus.DENIED
    assert second.denial is not None
    assert second.retry_after_seconds is not None
    assert calls == 1


async def test_short_flood_wait_pauses_only_its_action_class(scout: ScoutDatabase) -> None:
    """A brief wait on one endpoint must not stop the whole crawl."""
    governor = TelegramActionGovernor(scout=scout)

    async def call() -> str:
        raise _flood(30)

    result = await governor.run(ActionKind.HASHTAG_SEARCH, "search-flood", call)

    assert result.status is ActionStatus.FLOOD_WAIT
    assert result.flood_wait_seconds == 30
    assert result.retry_after_seconds == 90

    async def other_call() -> str:
        return "still allowed"

    other = await governor.run(ActionKind.RECOMMENDATIONS, "recs-1", other_call)
    assert other.ok

    blocked = await governor.run(ActionKind.HASHTAG_SEARCH, "search-again", other_call)
    assert blocked.status is ActionStatus.DENIED


async def test_long_flood_wait_pauses_every_crawl_action(scout: ScoutDatabase) -> None:
    """A long wait is about the account, not one endpoint."""
    governor = TelegramActionGovernor(scout=scout)

    async def call() -> str:
        raise _flood(20 * 60)

    result = await governor.run(ActionKind.HISTORY_PAGE, "page-flood", call)

    assert result.status is ActionStatus.FLOOD_WAIT

    async def other_call() -> str:
        return "should not run"

    blocked = await governor.run(ActionKind.HASHTAG_SEARCH, "search-after", other_call)
    assert blocked.status is ActionStatus.DENIED


async def test_six_hour_flood_wait_requires_a_human(scout: ScoutDatabase) -> None:
    """At this scale the account needs looking at before anything resumes."""
    governor = TelegramActionGovernor(scout=scout)

    async def call() -> str:
        raise _flood(7 * 60 * 60)

    result = await governor.run(ActionKind.JOIN, "join-flood", call)

    assert result.status is ActionStatus.HALTED

    async def other_call() -> str:
        return "should not run"

    blocked = await governor.run(ActionKind.HASHTAG_SEARCH, "search-halted", other_call)
    assert blocked.status is ActionStatus.HALTED
    assert blocked.retry_after_seconds is None


async def test_spam_limitation_halts_the_crawl(scout: ScoutDatabase) -> None:
    """PeerFlood escalates on repetition, so nothing automatic may retry."""
    governor = TelegramActionGovernor(scout=scout)

    async def call() -> str:
        raise errors.PeerFloodError(request=None)

    result = await governor.run(ActionKind.JOIN, "join-peerflood", call)

    assert result.status is ActionStatus.HALTED
    assert result.error_code == "PeerFloodError"

    async def other_call() -> str:
        return "should not run"

    blocked = await governor.run(ActionKind.HISTORY_PAGE, "page-after-flood", other_call)
    assert blocked.status is ActionStatus.HALTED


async def test_channel_limit_halts_joins_but_not_reading(scout: ScoutDatabase) -> None:
    """Being at the channel cap stops joining, not the rest of the crawl."""
    governor = TelegramActionGovernor(scout=scout)

    async def call() -> str:
        raise errors.ChannelsTooMuchError(request=None)

    result = await governor.run(ActionKind.JOIN, "join-too-many", call)

    assert result.status is ActionStatus.HALTED
    assert result.error_code == "channels_too_much"

    async def read() -> str:
        return "history"

    assert (await governor.run(ActionKind.HISTORY_PAGE, "page-ok", read)).ok


async def test_invalid_target_is_rejected_without_a_cooldown(scout: ScoutDatabase) -> None:
    """A dead username is the candidate's problem, not the account's."""
    governor = TelegramActionGovernor(scout=scout)

    async def call() -> str:
        raise errors.UsernameNotOccupiedError(request=None)

    result = await governor.run(ActionKind.RESOLVE_USERNAME, "resolve-dead", call)

    assert result.status is ActionStatus.REJECTED
    assert result.error_code == "UsernameNotOccupiedError"

    async def other_call() -> str:
        return "fine"

    assert (await governor.run(ActionKind.RESOLVE_USERNAME, "resolve-live", other_call)).ok


async def test_unexpected_failure_propagates_and_spends_the_slot(
    scout: ScoutDatabase,
) -> None:
    """An unknown outcome must cost its slot rather than be retried blindly."""
    governor = TelegramActionGovernor(
        scout=scout,
        policy={ActionKind.JOIN: BudgetRule(per_day=1)},
    )

    async def call() -> str:
        raise ConnectionResetError("socket died mid-join")

    with pytest.raises(ConnectionResetError):
        await governor.run(ActionKind.JOIN, "join-ambiguous", call)

    row = await _action_row(scout, "join-ambiguous")
    assert row["outcome"] == ActionOutcome.AMBIGUOUS.value

    async def retry() -> str:
        return "second attempt"

    denied = await governor.run(ActionKind.JOIN, "join-retry", retry)
    assert denied.status is ActionStatus.DENIED


async def test_replayed_key_does_not_double_spend(scout: ScoutDatabase) -> None:
    """Retrying a submitted action must not consume a second slot."""
    governor = TelegramActionGovernor(
        scout=scout,
        policy={ActionKind.JOIN: BudgetRule(per_day=1)},
    )

    async def call() -> str:
        return "joined"

    first = await governor.run(ActionKind.JOIN, "join-once", call)
    second = await governor.run(ActionKind.JOIN, "join-once", call)

    assert first.ok
    assert second.ok
    assert await scout.budget_usage(account_id="owner-primary", kind=ActionKind.JOIN) == (1, 1)


async def test_calls_are_serialized(scout: ScoutDatabase) -> None:
    """Only one crawl request may be in flight on the account."""
    import asyncio

    governor = TelegramActionGovernor(scout=scout)
    concurrent = 0
    peak = 0

    async def call() -> str:
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.01)
        concurrent -= 1
        return "done"

    await asyncio.gather(
        *(governor.run(ActionKind.HISTORY_PAGE, f"page-{index}", call) for index in range(5))
    )

    assert peak == 1


async def test_slow_call_is_flagged_as_a_hidden_flood_wait(
    scout: ScoutDatabase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Telethon sleeps short waits inside the call; duration is the only trace."""
    monkeypatch.setattr("pipeline.governor.HIDDEN_FLOOD_SUSPICION_MS", 1.0)
    governor = TelegramActionGovernor(scout=scout)

    async def call() -> str:
        import asyncio

        await asyncio.sleep(0.01)
        return "slow"

    with caplog.at_level("WARNING"):
        result = await governor.run(ActionKind.HISTORY_PAGE, "page-slow", call)

    assert result.ok
    assert "may hide a short FloodWait" in caplog.text
    row = await _action_row(scout, "page-slow")
    assert float(row["duration_ms"] or 0) >= 10
