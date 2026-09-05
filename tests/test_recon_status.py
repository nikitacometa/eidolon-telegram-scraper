"""Tests for the reconnaissance status state machine.

The owner asked for a status when a task starts and a report when it ends —
not the same numbers twice a day for a month. The decision is pure, so these
tests drive it with snapshots and clocks and never touch Telegram.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pipeline.recon_status import (
    Phase,
    ReconSnapshot,
    ReconStatusState,
    decide,
)

NOW = datetime(2026, 9, 5, 4, 0, tzinfo=UTC)


def _idle_snapshot(**overrides: object) -> ReconSnapshot:
    values: dict[str, object] = {
        "joined": ("danang_special", "danangevents", "dalatrus"),
        "pending": (),
        "requested": ("dalat_info", "phangane"),
        "failed": (),
        "backfill_pending": 0,
        "backfill_done": 46,
        "messages_stored": 307390,
        "oldest": "2024-07-30",
        "newest": "2026-09-04",
    }
    values.update(overrides)
    return ReconSnapshot(**values)  # type: ignore[arg-type]


def test_nothing_in_progress_means_nothing_is_sent() -> None:
    """The twice-daily message the owner asked to stop: 44 joined, 0 queued."""
    decision = decide(_idle_snapshot(), ReconStatusState(), now=NOW)

    assert decision.message is None
    assert decision.state.phase is Phase.IDLE


def test_chats_awaiting_an_admin_do_not_count_as_work() -> None:
    decision = decide(
        _idle_snapshot(requested=("a", "b", "c", "d", "e", "f", "g")),
        ReconStatusState(),
        now=NOW,
    )

    assert decision.message is None


def test_a_new_task_is_announced_once_with_its_queue() -> None:
    snapshot = _idle_snapshot(pending=("phangan_rent", "kohphangan_housing"), backfill_pending=1)

    decision = decide(snapshot, ReconStatusState(), now=NOW)

    assert decision.message is not None
    assert "Разведка: новое задание" in decision.message
    assert "• @phangan_rent" in decision.message
    assert "Докачивается история:</b> 1 чат" in decision.message
    assert "Дананг" not in decision.message
    assert decision.state.phase is Phase.WORKING
    assert decision.state.baseline_joined == snapshot.joined
    assert decision.state.baseline_messages == 307390
    assert decision.state.started_at == NOW

    # An hour later the task is still running: silence, not a repeat.
    later = decide(snapshot, decision.state, now=NOW + timedelta(hours=1))
    assert later.message is None
    assert later.state.phase is Phase.WORKING


def test_a_long_task_reports_progress_at_most_twice_a_day() -> None:
    started = decide(_idle_snapshot(pending=("a", "b", "c")), ReconStatusState(), now=NOW)
    progressed = _idle_snapshot(
        joined=("danang_special", "danangevents", "dalatrus", "a"),
        pending=("b", "c"),
        backfill_pending=1,
        messages_stored=307390 + 4200,
    )

    quiet = decide(progressed, started.state, now=NOW + timedelta(hours=11))
    loud = decide(progressed, started.state, now=NOW + timedelta(hours=12))

    assert quiet.message is None
    assert loud.message is not None
    assert "Разведка идёт" in loud.message
    assert "Вступил:</b> 1 из 3, в очереди 2" in loud.message
    assert "Скачано:</b> 4 200 сообщений" in loud.message
    assert loud.state.baseline_joined == started.state.baseline_joined
    assert loud.state.started_at == NOW


def test_finishing_sends_one_report_with_what_the_task_added_then_goes_quiet() -> None:
    started = decide(_idle_snapshot(pending=("a", "b")), ReconStatusState(), now=NOW)
    done = _idle_snapshot(
        joined=("danang_special", "danangevents", "dalatrus", "a", "b"),
        requested=("dalat_info",),
        failed=(("ghost_chat", "UsernameNotOccupiedError"),),
        messages_stored=307390 + 12000,
    )

    finished = decide(done, started.state, now=NOW + timedelta(hours=5))

    assert finished.message is not None
    assert "Разведка завершена" in finished.message
    assert "Заняло:</b> ~5 часов" in finished.message
    assert "Вступил в 2 чата" in finished.message
    assert "• @a" in finished.message and "• @b" in finished.message
    assert "• @danang_special" not in finished.message  # only what the task added
    assert "Скачано:</b> 12 000 сообщений" in finished.message
    assert "Ждут одобрения админа (1)" in finished.message
    assert "@ghost_chat — UsernameNotOccupiedError" in finished.message
    assert "Всего читаю 5 чатов" in finished.message
    assert finished.state.phase is Phase.IDLE

    again = decide(done, finished.state, now=NOW + timedelta(hours=6))
    assert again.message is None


def test_the_state_survives_a_round_trip_through_json() -> None:
    started = decide(_idle_snapshot(pending=("a",)), ReconStatusState(), now=NOW)

    restored = ReconStatusState.from_dict(started.state.as_dict())

    assert restored.phase is Phase.WORKING
    assert restored.baseline_joined == started.state.baseline_joined
    assert restored.started_at == NOW
    assert restored.last_sent_at == NOW


def test_a_missing_or_broken_state_file_reads_as_idle() -> None:
    assert ReconStatusState.from_dict({}).phase is Phase.IDLE
    assert ReconStatusState.from_dict({"phase": "working-now"}).phase is Phase.IDLE
    assert (
        ReconStatusState.from_dict({"phase": "idle", "last_sent_at": "garbage"}).last_sent_at
        is None
    )
