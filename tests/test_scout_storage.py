"""Tests for storage/scout.py — reconnaissance state and action budgets."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from pipeline.recon_models import (
    ActionKind,
    ActionOutcome,
    ActionReservation,
    BudgetDenial,
    BudgetRule,
    BudgetScope,
    CandidateState,
    ChatIdentity,
    ChatMembership,
    ChatVisibility,
    DiscoverySource,
    Evidence,
    JobRequest,
    ReconJobStatus,
    normalize_username,
)
from storage.scout import ScoutDatabase, invite_fingerprint


@pytest.fixture
async def scout(tmp_path: Path) -> AsyncIterator[ScoutDatabase]:
    """Open a scout database backed by a temporary file."""
    database = ScoutDatabase(tmp_path / "scout.db")
    await database.connect()
    yield database
    await database.close()


def _request(key: str = "recon-danang-1", **overrides: object) -> JobRequest:
    payload: dict[str, object] = {
        "idempotency_key": key,
        "topic": "housing and local life",
        "location": "Da Nang, Vietnam",
        "languages": ("en", "ru", "vi"),
    }
    payload.update(overrides)
    return JobRequest(**payload)  # type: ignore[arg-type]


async def test_connect_creates_tables(scout: ScoutDatabase) -> None:
    """The schema must be applied on connect."""
    cursor = await scout.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        " ORDER BY name"
    )
    tables = {row[0] for row in await cursor.fetchall()}
    assert tables == {
        "account_cooldowns",
        "candidate_evidence",
        "chat_aliases",
        "job_candidates",
        "recon_frontier",
        "recon_jobs",
        "scout_chats",
        "telegram_actions",
    }


# ----------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------


async def test_repeated_submit_returns_the_same_job(scout: ScoutDatabase) -> None:
    """A retried submit must not start a second crawl."""
    first = await scout.create_job(_request())
    second = await scout.create_job(_request())

    assert second.id == first.id
    cursor = await scout.conn.execute("SELECT COUNT(*) FROM recon_jobs")
    row = await cursor.fetchone()
    assert row is not None
    assert int(row[0]) == 1


async def test_job_stores_request_parameters(scout: ScoutDatabase) -> None:
    """Job fields must survive the round trip."""
    job = await scout.create_job(_request(seeds=("@danang_chat",), lookback_days=45))

    assert job.status is ReconJobStatus.QUEUED
    assert job.languages == ("en", "ru", "vi")
    assert job.seeds == ("@danang_chat",)
    assert job.lookback_days == 45
    assert job.deadline_at is not None


async def test_terminal_status_is_immutable(scout: ScoutDatabase) -> None:
    """A finished job must never regress into a running state."""
    job = await scout.create_job(_request())
    assert await scout.update_job_status(job.id, ReconJobStatus.COMPLETED)

    assert not await scout.update_job_status(job.id, ReconJobStatus.DISCOVERING)

    reloaded = await scout.get_job(job.id)
    assert reloaded is not None
    assert reloaded.status is ReconJobStatus.COMPLETED
    assert reloaded.is_terminal
    assert reloaded.completed_at is not None


async def test_pause_records_resume_time_and_reason(scout: ScoutDatabase) -> None:
    """A rate-limit pause must carry when work may continue."""
    job = await scout.create_job(_request())

    assert await scout.update_job_status(
        job.id,
        ReconJobStatus.PAUSED_RATE_LIMIT,
        stop_reason="flood_wait",
        resume_after_seconds=3600,
        current_wave=1,
    )

    reloaded = await scout.get_job(job.id)
    assert reloaded is not None
    assert reloaded.status is ReconJobStatus.PAUSED_RATE_LIMIT
    assert reloaded.stop_reason == "flood_wait"
    assert reloaded.resume_at is not None
    assert reloaded.current_wave == 1


async def test_active_jobs_excludes_finished_work(scout: ScoutDatabase) -> None:
    """Recovery after restart must only pick up unfinished jobs."""
    running = await scout.create_job(_request("key-running"))
    done = await scout.create_job(_request("key-done"))
    await scout.update_job_status(done.id, ReconJobStatus.CANCELLED)

    assert [job.id for job in await scout.active_jobs()] == [running.id]


async def test_update_of_unknown_job_reports_failure(scout: ScoutDatabase) -> None:
    """Updating a job that does not exist must not silently succeed."""
    assert not await scout.update_job_status("missing", ReconJobStatus.DISCOVERING)


# ----------------------------------------------------------------------
# Chat identity
# ----------------------------------------------------------------------


async def test_same_username_resolves_to_one_chat(scout: ScoutDatabase) -> None:
    """A chat mentioned twice by username stays one node."""
    first = await scout.resolve_chat(ChatIdentity(username="@DaNangChat"))
    second = await scout.resolve_chat(ChatIdentity(username="https://t.me/danangchat"))

    assert first == second


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("@DaNangChat", "danangchat"),
        ("https://t.me/DaNangChat", "danangchat"),
        ("t.me/danangchat/", "danangchat"),
        ("danangchat?start=1", "danangchat"),
    ],
)
def test_username_normalization(raw: str, expected: str) -> None:
    """Locator forms of the same username must compare equal."""
    assert normalize_username(raw) == expected


async def test_resolving_a_peer_merges_duplicate_records(scout: ScoutDatabase) -> None:
    """Learning the peer id behind a username must collapse the two nodes."""
    by_username = await scout.resolve_chat(ChatIdentity(username="danang_housing"))
    by_peer = await scout.resolve_chat(ChatIdentity(peer_id=-1001234567890))
    assert by_username != by_peer

    merged = await scout.resolve_chat(
        ChatIdentity(peer_id=-1001234567890, username="danang_housing")
    )

    cursor = await scout.conn.execute("SELECT COUNT(*) FROM scout_chats")
    row = await cursor.fetchone()
    assert row is not None
    assert int(row[0]) == 1

    cursor = await scout.conn.execute("SELECT DISTINCT chat_uuid FROM chat_aliases")
    owners = [str(alias_row[0]) for alias_row in await cursor.fetchall()]
    assert owners == [merged]

    chat = await scout.get_chat(merged)
    assert chat is not None
    assert chat.telegram_chat_id == -1001234567890
    assert chat.username == "danang_housing"


async def test_merge_keeps_one_candidate_and_its_evidence(scout: ScoutDatabase) -> None:
    """Merging chats inside one job must not duplicate the candidate."""
    job = await scout.create_job(_request())
    username_chat = await scout.resolve_chat(ChatIdentity(username="danang_rent"))
    peer_chat = await scout.resolve_chat(ChatIdentity(peer_id=-100777))

    await scout.add_candidate(
        job_id=job.id,
        chat_uuid=username_chat,
        wave=0,
        evidence=Evidence(source=DiscoverySource.MESSAGE_LINK, origin_key="author:11"),
    )
    await scout.add_candidate(
        job_id=job.id,
        chat_uuid=peer_chat,
        wave=1,
        evidence=Evidence(source=DiscoverySource.FORWARD, origin_key="author:22"),
    )

    await scout.resolve_chat(ChatIdentity(peer_id=-100777, username="danang_rent"))

    candidates = await scout.candidates_for_job(job.id)
    assert len(candidates) == 1
    assert candidates[0].independent_sources == 2


async def test_visibility_and_membership_are_recorded(scout: ScoutDatabase) -> None:
    """Public proof and confirmed membership must be persisted with provenance."""
    chat_uuid = await scout.resolve_chat(ChatIdentity(username="danang_public"))

    await scout.set_chat_visibility(chat_uuid, ChatVisibility.PUBLIC, source="search_posts")
    await scout.set_chat_membership(chat_uuid, ChatMembership.MEMBER)

    chat = await scout.get_chat(chat_uuid)
    assert chat is not None
    assert chat.visibility is ChatVisibility.PUBLIC
    assert chat.visibility_source == "search_posts"
    assert chat.membership is ChatMembership.MEMBER
    assert chat.joined_at is not None


async def test_join_request_is_not_membership(scout: ScoutDatabase) -> None:
    """INVITE_REQUEST_SENT must not be stored as a completed join."""
    chat_uuid = await scout.resolve_chat(ChatIdentity(username="moderated_chat"))

    await scout.set_chat_membership(chat_uuid, ChatMembership.REQUESTED)

    chat = await scout.get_chat(chat_uuid)
    assert chat is not None
    assert chat.membership is ChatMembership.REQUESTED
    assert chat.joined_at is None


def test_invite_fingerprint_is_stable_and_hides_the_hash() -> None:
    """An invite handle must be recognisable later without storing the key."""
    secret = b"test-secret"
    first = invite_fingerprint("https://t.me/+AbCdEf123", secret=secret)
    second = invite_fingerprint("AbCdEf123", secret=secret)

    assert first == second
    assert "AbCdEf123" not in first
    assert first != invite_fingerprint("AbCdEf123", secret=b"other-secret")


# ----------------------------------------------------------------------
# Candidates and evidence
# ----------------------------------------------------------------------


async def test_rediscovery_adds_evidence_instead_of_a_new_candidate(
    scout: ScoutDatabase,
) -> None:
    """Finding a chat by a second route strengthens it, not duplicates it."""
    job = await scout.create_job(_request())
    chat_uuid = await scout.resolve_chat(ChatIdentity(username="danang_expats"))

    first_id = await scout.add_candidate(
        job_id=job.id,
        chat_uuid=chat_uuid,
        wave=0,
        evidence=Evidence(source=DiscoverySource.HASHTAG_SEARCH, origin_key="hashtag:danang"),
    )
    second_id = await scout.add_candidate(
        job_id=job.id,
        chat_uuid=chat_uuid,
        wave=0,
        evidence=Evidence(source=DiscoverySource.RECOMMENDATIONS, origin_key="peer:-100999"),
    )

    assert first_id == second_id
    candidate = await scout.get_candidate(first_id)
    assert candidate is not None
    assert candidate.independent_sources == 2


async def test_repeated_signal_from_one_origin_counts_once(scout: ScoutDatabase) -> None:
    """One actor reposting itself must not manufacture independent support.

    Counting distinct source chats would let a single account cross the
    auto-join bar by dropping the same message into two crawled chats.
    """
    job = await scout.create_job(_request())
    chat_uuid = await scout.resolve_chat(ChatIdentity(username="danang_villas_verified"))

    for source_chat in (-100111, -100222):
        await scout.add_candidate(
            job_id=job.id,
            chat_uuid=chat_uuid,
            wave=0,
            evidence=Evidence(
                source=DiscoverySource.MESSAGE_LINK,
                origin_key="author:5551234",
                source_chat_id=source_chat,
            ),
        )

    candidate = await scout.get_candidate(
        (await scout.candidates_for_job(job.id))[0].id,
    )
    assert candidate is not None
    assert candidate.independent_sources == 1


async def test_one_origin_reaching_us_two_ways_counts_once(scout: ScoutDatabase) -> None:
    """The same actor posting and then forwarding is still one source.

    The unique key alone does not cover this: two different discovery sources
    are two evidence rows, so independence has to be counted by origin.
    """
    job = await scout.create_job(_request())
    chat_uuid = await scout.resolve_chat(ChatIdentity(username="danang_villas_verified"))

    await scout.add_candidate(
        job_id=job.id,
        chat_uuid=chat_uuid,
        wave=0,
        evidence=Evidence(
            source=DiscoverySource.MESSAGE_LINK,
            origin_key="author:5551234",
            source_chat_id=-100111,
        ),
    )
    candidate_id = await scout.add_candidate(
        job_id=job.id,
        chat_uuid=chat_uuid,
        wave=0,
        evidence=Evidence(
            source=DiscoverySource.FORWARD,
            origin_key="author:5551234",
            source_chat_id=-100222,
        ),
    )

    cursor = await scout.conn.execute(
        "SELECT COUNT(*) FROM candidate_evidence WHERE candidate_id = ?",
        (candidate_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert int(row[0]) == 2

    candidate = await scout.get_candidate(candidate_id)
    assert candidate is not None
    assert candidate.independent_sources == 1


async def test_scoring_bumps_version_and_stale_approval_is_refused(
    scout: ScoutDatabase,
) -> None:
    """A decision taken before a rescore must not apply afterwards."""
    job = await scout.create_job(_request())
    chat_uuid = await scout.resolve_chat(ChatIdentity(username="danang_flats"))
    candidate_id = await scout.add_candidate(
        job_id=job.id,
        chat_uuid=chat_uuid,
        wave=0,
        evidence=Evidence(source=DiscoverySource.OWNER_SEED, origin_key="owner"),
    )

    version = await scout.score_candidate(candidate_id, policy_score=88.0)
    await scout.score_candidate(candidate_id, policy_score=31.0, risk_flags=["topic_drift"])

    assert not await scout.transition_candidate(
        candidate_id,
        CandidateState.APPROVED,
        expected_version=version,
    )

    candidate = await scout.get_candidate(candidate_id)
    assert candidate is not None
    assert candidate.state is CandidateState.SCORED
    assert candidate.policy_score == 31.0
    assert candidate.risk_flags == ("topic_drift",)


async def test_approval_cannot_reopen_a_private_candidate(scout: ScoutDatabase) -> None:
    """The public-scope gate must outrank any approval."""
    job = await scout.create_job(_request())
    chat_uuid = await scout.resolve_chat(ChatIdentity(username="secret_group"))
    candidate_id = await scout.add_candidate(
        job_id=job.id,
        chat_uuid=chat_uuid,
        wave=1,
        evidence=Evidence(source=DiscoverySource.MESSAGE_LINK, origin_key="author:9"),
    )
    await scout.transition_candidate(candidate_id, CandidateState.BLOCKED_PRIVATE)

    assert not await scout.transition_candidate(candidate_id, CandidateState.APPROVED)

    candidate = await scout.get_candidate(candidate_id)
    assert candidate is not None
    assert candidate.state is CandidateState.BLOCKED_PRIVATE


async def test_candidates_can_be_filtered_by_state(scout: ScoutDatabase) -> None:
    """Approval flows need to list exactly the pending decisions."""
    job = await scout.create_job(_request())
    waiting = await scout.add_candidate(
        job_id=job.id,
        chat_uuid=await scout.resolve_chat(ChatIdentity(username="chat_a")),
        wave=0,
        evidence=Evidence(source=DiscoverySource.HASHTAG_SEARCH, origin_key="tag:a"),
    )
    await scout.add_candidate(
        job_id=job.id,
        chat_uuid=await scout.resolve_chat(ChatIdentity(username="chat_b")),
        wave=0,
        evidence=Evidence(source=DiscoverySource.HASHTAG_SEARCH, origin_key="tag:b"),
    )
    await scout.transition_candidate(waiting, CandidateState.AWAITING_APPROVAL)

    pending = await scout.candidates_for_job(
        job.id,
        states=[CandidateState.AWAITING_APPROVAL],
    )
    assert [candidate.id for candidate in pending] == [waiting]
    assert await scout.candidates_for_job(job.id, states=[]) == []


# ----------------------------------------------------------------------
# Frontier
# ----------------------------------------------------------------------


async def _seed_candidate(
    scout: ScoutDatabase,
    job_id: str,
    username: str,
    *,
    priority: float,
) -> int:
    return await scout.add_candidate(
        job_id=job_id,
        chat_uuid=await scout.resolve_chat(ChatIdentity(username=username)),
        wave=0,
        evidence=Evidence(source=DiscoverySource.HASHTAG_SEARCH, origin_key=f"tag:{username}"),
        priority=priority,
    )


async def test_frontier_serves_highest_priority_first(scout: ScoutDatabase) -> None:
    """Hubs and owner seeds must be crawled before incidental mentions."""
    job = await scout.create_job(_request())
    await _seed_candidate(scout, job.id, "low", priority=1.0)
    high = await _seed_candidate(scout, job.id, "high", priority=90.0)

    claimed = await scout.claim_frontier(job.id, limit=1)

    assert [item.candidate_id for item in claimed] == [high]


async def test_claimed_work_is_not_handed_to_a_second_worker(
    scout: ScoutDatabase,
) -> None:
    """A live lease must hide the item from other claimers."""
    job = await scout.create_job(_request())
    await _seed_candidate(scout, job.id, "only", priority=5.0)

    first = await scout.claim_frontier(job.id, limit=5)
    second = await scout.claim_frontier(job.id, limit=5)

    assert len(first) == 1
    assert second == []


async def test_expired_lease_is_reclaimed(scout: ScoutDatabase) -> None:
    """Work abandoned by a crashed worker must come back."""
    job = await scout.create_job(_request())
    candidate_id = await _seed_candidate(scout, job.id, "orphan", priority=5.0)
    await scout.claim_frontier(job.id, limit=1)
    await scout.conn.execute(
        "UPDATE recon_frontier SET claimed_until = datetime('now', '-1 minute')"
    )
    await scout.conn.commit()

    reclaimed = await scout.claim_frontier(job.id, limit=1)

    assert [item.candidate_id for item in reclaimed] == [candidate_id]
    assert reclaimed[0].attempts == 2


async def test_stale_worker_cannot_settle_reclaimed_work(scout: ScoutDatabase) -> None:
    """A worker whose lease expired must not overwrite newer state."""
    job = await scout.create_job(_request())
    candidate_id = await _seed_candidate(scout, job.id, "contested", priority=5.0)
    stale = (await scout.claim_frontier(job.id, limit=1))[0]
    await scout.conn.execute(
        "UPDATE recon_frontier SET claimed_until = datetime('now', '-1 minute')"
    )
    await scout.conn.commit()
    fresh = (await scout.claim_frontier(job.id, limit=1))[0]

    assert not await scout.settle_frontier(candidate_id, claim_token=stale.claim_token)
    assert await scout.settle_frontier(candidate_id, claim_token=fresh.claim_token)


async def test_retry_delay_holds_work_back(scout: ScoutDatabase) -> None:
    """Rescheduled work must not be claimable before its delay elapses."""
    job = await scout.create_job(_request())
    await _seed_candidate(scout, job.id, "delayed", priority=5.0)
    claimed = (await scout.claim_frontier(job.id, limit=1))[0]

    await scout.settle_frontier(
        claimed.candidate_id,
        claim_token=claimed.claim_token,
        state="pending",
        retry_after_seconds=600,
        error="flood_wait",
    )

    assert await scout.claim_frontier(job.id, limit=1) == []


# ----------------------------------------------------------------------
# Action budgets
# ----------------------------------------------------------------------


async def test_join_budget_stops_at_the_daily_cap(scout: ScoutDatabase) -> None:
    """The fourth join in a rolling day must be refused whatever the score."""
    policy = {ActionKind.JOIN: BudgetRule(per_hour=None, per_day=3)}
    for index in range(3):
        granted = await scout.reserve_action(
            account_id="owner-primary",
            kind=ActionKind.JOIN,
            idempotency_key=f"join-{index}",
            policy=policy,
        )
        assert isinstance(granted, ActionReservation)

    denied = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="join-4",
        policy=policy,
    )

    assert isinstance(denied, BudgetDenial)
    assert denied.scope is BudgetScope.DAY
    assert denied.used == 3
    assert denied.cap == 3
    assert denied.retry_after_seconds is not None


async def test_hourly_cap_applies_before_the_daily_cap(scout: ScoutDatabase) -> None:
    """A tight hourly window must throttle even when the day has room."""
    policy = {ActionKind.JOIN: BudgetRule(per_hour=1, per_day=3)}
    await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="join-hour-1",
        policy=policy,
    )

    denied = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="join-hour-2",
        policy=policy,
    )

    assert isinstance(denied, BudgetDenial)
    assert denied.scope is BudgetScope.HOUR


async def test_rolling_window_forgets_old_actions(scout: ScoutDatabase) -> None:
    """Yesterday's joins must not block today, and midnight must not matter."""
    policy = {ActionKind.JOIN: BudgetRule(per_day=1)}
    await scout.conn.execute(
        """
        INSERT INTO telegram_actions (account_id, kind, idempotency_key, outcome, reserved_at)
        VALUES ('owner-primary', 'join', 'old-join', 'succeeded', datetime('now', '-25 hours'))
        """
    )
    await scout.conn.commit()

    granted = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="fresh-join",
        policy=policy,
    )

    assert isinstance(granted, ActionReservation)


async def test_replayed_reservation_does_not_consume_a_second_slot(
    scout: ScoutDatabase,
) -> None:
    """A retried reservation must return the original slot, not a new one."""
    policy = {ActionKind.JOIN: BudgetRule(per_day=1)}
    first = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="join-retry",
        policy=policy,
    )
    second = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="join-retry",
        policy=policy,
    )

    assert isinstance(first, ActionReservation)
    assert isinstance(second, ActionReservation)
    assert second.action_id == first.action_id
    assert second.replayed
    assert await scout.budget_usage(account_id="owner-primary", kind=ActionKind.JOIN) == (1, 1)


async def test_failed_attempt_still_costs_its_slot(scout: ScoutDatabase) -> None:
    """A reservation is never refunded: the attempt reached Telegram."""
    policy = {ActionKind.JOIN: BudgetRule(per_day=1)}
    granted = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="join-doomed",
        policy=policy,
    )
    assert isinstance(granted, ActionReservation)
    await scout.settle_action(
        granted.action_id,
        outcome=ActionOutcome.AMBIGUOUS,
        error_code="disconnected",
    )

    denied = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="join-next",
        policy=policy,
    )

    assert isinstance(denied, BudgetDenial)


async def test_budgets_are_tracked_per_action_class(scout: ScoutDatabase) -> None:
    """Exhausting joins must not block reading history."""
    policy = {
        ActionKind.JOIN: BudgetRule(per_day=1),
        ActionKind.HISTORY_PAGE: BudgetRule(per_day=10),
    }
    await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="join-only",
        policy=policy,
    )

    history = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.HISTORY_PAGE,
        idempotency_key="page-1",
        policy=policy,
    )

    assert isinstance(history, ActionReservation)


async def test_budgets_are_tracked_per_account(scout: ScoutDatabase) -> None:
    """A future scout account must carry its own budget, not inherit one."""
    policy = {ActionKind.JOIN: BudgetRule(per_day=1)}
    await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="primary-join",
        policy=policy,
    )

    scout_account = await scout.reserve_action(
        account_id="scout-1",
        kind=ActionKind.JOIN,
        idempotency_key="scout-join",
        policy=policy,
    )

    assert isinstance(scout_account, ActionReservation)


async def test_cooldown_blocks_the_whole_account(scout: ScoutDatabase) -> None:
    """A spam limitation must stop every crawl action, not just joins."""
    await scout.set_cooldown(
        account_id="owner-primary",
        scope="all",
        seconds=3600,
        reason="peer_flood",
    )

    denied = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.HASHTAG_SEARCH,
        idempotency_key="search-during-cooldown",
    )

    assert isinstance(denied, BudgetDenial)
    assert denied.scope is BudgetScope.COOLDOWN
    assert denied.reason == "peer_flood"


async def test_manual_cooldown_offers_no_automatic_retry(scout: ScoutDatabase) -> None:
    """A safety halt must require an operator, not a timer."""
    await scout.set_cooldown(
        account_id="owner-primary",
        scope="all",
        seconds=21_600,
        reason="flood_wait_6h",
        manual_resume_required=True,
    )

    denied = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="join-halted",
    )

    assert isinstance(denied, BudgetDenial)
    assert denied.retry_after_seconds is None


async def test_cooldown_can_be_lifted(scout: ScoutDatabase) -> None:
    """Clearing a halt must let work resume."""
    await scout.set_cooldown(
        account_id="owner-primary",
        scope="join",
        seconds=600,
        reason="flood_wait",
    )
    await scout.clear_cooldown(account_id="owner-primary", scope="join")

    granted = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="join-after-clear",
    )

    assert isinstance(granted, ActionReservation)


async def test_scoped_cooldown_leaves_other_actions_alone(scout: ScoutDatabase) -> None:
    """Pausing one action class must not halt the rest of the crawl."""
    await scout.set_cooldown(
        account_id="owner-primary",
        scope="join",
        seconds=600,
        reason="flood_wait",
    )

    search = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.HASHTAG_SEARCH,
        idempotency_key="search-while-join-paused",
    )

    assert isinstance(search, ActionReservation)


async def test_settled_action_keeps_flood_and_timing_evidence(
    scout: ScoutDatabase,
) -> None:
    """Call duration is the only trace of a FloodWait the library slept through."""
    granted = await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.HISTORY_PAGE,
        idempotency_key="page-timed",
    )
    assert isinstance(granted, ActionReservation)

    await scout.settle_action(
        granted.action_id,
        outcome=ActionOutcome.FLOOD_WAIT,
        duration_ms=41_200.0,
        flood_wait_seconds=41,
    )

    cursor = await scout.conn.execute(
        "SELECT outcome, duration_ms, flood_wait_seconds, settled_at FROM telegram_actions"
        " WHERE id = ?",
        (granted.action_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert str(row["outcome"]) == ActionOutcome.FLOOD_WAIT.value
    assert float(row["duration_ms"]) == 41_200.0
    assert int(row["flood_wait_seconds"]) == 41
    assert row["settled_at"] is not None


async def test_budget_usage_separates_hour_from_day(scout: ScoutDatabase) -> None:
    """Usage reporting must distinguish the two rolling windows."""
    await scout.conn.execute(
        """
        INSERT INTO telegram_actions (account_id, kind, idempotency_key, outcome, reserved_at)
        VALUES ('owner-primary', 'join', 'earlier-today', 'succeeded', datetime('now', '-5 hours'))
        """
    )
    await scout.conn.commit()
    await scout.reserve_action(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
        idempotency_key="just-now",
    )

    used_hour, used_day = await scout.budget_usage(
        account_id="owner-primary",
        kind=ActionKind.JOIN,
    )

    assert used_hour == 1
    assert used_day == 2


# ----------------------------------------------------------------------
# Request validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"idempotency_key": "  "},
        {"topic": ""},
        {"max_waves": 3},
        {"lookback_days": 0},
        {"lookback_days": 91},
        {"max_candidates": 0},
        {"deadline_hours": 0},
    ],
)
def test_invalid_job_requests_are_rejected(overrides: dict[str, object]) -> None:
    """Server-side clamps must not be bypassable by the calling agent."""
    with pytest.raises(ValueError):
        _request(**overrides)


def test_identity_requires_at_least_one_alias() -> None:
    """A chat with no locator at all cannot be tracked."""
    with pytest.raises(ValueError):
        ChatIdentity()
