"""When to tell the owner about reconnaissance, and what to say.

The join queue and the history archive advance on their own for hours, so a
timer reports on them. But a report that says the same thing twice a day for a
month is noise, and noise gets muted: what the owner wants is one message when
a task starts, one when it finishes, and a word of progress in between only if
it is taking long. Everything else is silence.

The decision is a pure function over two snapshots — the current state and the
one recorded when the last report went out — so it is tested without Telegram
and the script around it only reads, sends and stores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

# A task still in progress is reported again no more often than this.
PROGRESS_INTERVAL_HOURS = 12


class Phase(StrEnum):
    """Whether reconnaissance has anything to do right now."""

    IDLE = "idle"
    WORKING = "working"


@dataclass(frozen=True, slots=True)
class ReconSnapshot:
    """Reconnaissance as it stands at one moment."""

    joined: tuple[str, ...]
    pending: tuple[str, ...]
    requested: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]
    backfill_pending: int
    backfill_done: int
    messages_stored: int
    oldest: str | None = None
    newest: str | None = None

    @property
    def active_work(self) -> int:
        """Joins still queued plus chats still downloading.

        A chat awaiting an admin is not work the daemon is doing; it counts
        only once the admin lets it in and the download starts.
        """
        return len(self.pending) + self.backfill_pending


@dataclass(slots=True)
class ReconStatusState:
    """What was true when the last report went out."""

    phase: Phase = Phase.IDLE
    last_sent_at: datetime | None = None
    # The joined set and the archive size at the moment a task began, so the
    # completion report can say what the task added.
    baseline_joined: tuple[str, ...] = ()
    baseline_messages: int = 0
    started_at: datetime | None = None
    # Chats named as awaiting an admin in the last report; a new one is news.
    reported_requested: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        data = asdict(self)
        data["phase"] = self.phase.value
        data["last_sent_at"] = self.last_sent_at.isoformat() if self.last_sent_at else None
        data["started_at"] = self.started_at.isoformat() if self.started_at else None
        data["baseline_joined"] = list(self.baseline_joined)
        data["reported_requested"] = list(self.reported_requested)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReconStatusState:
        """Read a stored state, tolerating a missing or partial file."""
        raw_phase = str(data.get("phase") or Phase.IDLE.value)
        return cls(
            # An unknown phase reads as idle: the worst that follows is one
            # extra "new task" message, where a crash would mean silence.
            phase=Phase(raw_phase) if raw_phase in Phase.__members__.values() else Phase.IDLE,
            last_sent_at=_parse(data.get("last_sent_at")),
            baseline_joined=tuple(str(x) for x in data.get("baseline_joined") or ()),
            baseline_messages=int(data.get("baseline_messages") or 0),
            started_at=_parse(data.get("started_at")),
            reported_requested=tuple(str(x) for x in data.get("reported_requested") or ()),
        )


@dataclass(frozen=True, slots=True)
class Decision:
    """What to send, if anything, and the state to store afterwards."""

    message: str | None
    state: ReconStatusState
    reason: str


def decide(snapshot: ReconSnapshot, state: ReconStatusState, *, now: datetime) -> Decision:
    """The state machine: idle <-> working, with reports on the transitions."""
    working = snapshot.active_work > 0

    if state.phase is Phase.IDLE and working:
        started = ReconStatusState(
            phase=Phase.WORKING,
            last_sent_at=now,
            baseline_joined=snapshot.joined,
            baseline_messages=snapshot.messages_stored,
            started_at=now,
            reported_requested=snapshot.requested,
        )
        return Decision(_render_started(snapshot), started, "task started")

    if state.phase is Phase.WORKING and not working:
        finished = ReconStatusState(
            phase=Phase.IDLE,
            last_sent_at=now,
            reported_requested=snapshot.requested,
        )
        return Decision(_render_finished(snapshot, state, now=now), finished, "task finished")

    if state.phase is Phase.WORKING:
        due = (
            state.last_sent_at is None
            or (now - state.last_sent_at).total_seconds() >= PROGRESS_INTERVAL_HOURS * 3600
        )
        if not due:
            return Decision(None, state, "in progress, reported recently")
        progressed = ReconStatusState(
            phase=Phase.WORKING,
            last_sent_at=now,
            baseline_joined=state.baseline_joined,
            baseline_messages=state.baseline_messages,
            started_at=state.started_at,
            reported_requested=snapshot.requested,
        )
        return Decision(_render_progress(snapshot, state), progressed, "progress report due")

    return Decision(None, state, "idle, nothing to report")


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _plural(count: int, one: str, few: str, many: str) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return one
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


def _messages(count: int) -> str:
    return (
        f"{count:,}".replace(",", " ") + " " + _plural(count, "сообщение", "сообщения", "сообщений")
    )


def _chat_list(refs: tuple[str, ...], *, limit: int = 8) -> list[str]:
    lines = [f"• @{ref}" for ref in refs[:limit]]
    if len(refs) > limit:
        lines.append(f"• …и ещё {len(refs) - limit}")
    return lines


def _render_started(snapshot: ReconSnapshot) -> str:
    lines = ["🛰 <b>Разведка: новое задание</b>", ""]
    if snapshot.pending:
        lines.append(
            f"<b>В очереди на вступление ({len(snapshot.pending)}):</b>",
        )
        lines += _chat_list(snapshot.pending)
    if snapshot.backfill_pending:
        lines.append(
            f"<b>Докачивается история:</b> {snapshot.backfill_pending} "
            f"{_plural(snapshot.backfill_pending, 'чат', 'чата', 'чатов')}"
        )
    if snapshot.requested:
        lines.append(f"<b>Ждут одобрения админа:</b> {len(snapshot.requested)}")
    lines += ["", "Отчитаюсь, когда закончу.", "", "#разведка #статус"]
    return "\n".join(lines)


def _render_progress(snapshot: ReconSnapshot, state: ReconStatusState) -> str:
    newly = tuple(ref for ref in snapshot.joined if ref not in state.baseline_joined)
    lines = ["🛰 <b>Разведка идёт</b>", ""]
    lines.append(
        f"<b>Вступил:</b> {len(newly)} из "
        f"{len(newly) + len(snapshot.pending)}, в очереди {len(snapshot.pending)}"
    )
    added = snapshot.messages_stored - state.baseline_messages
    lines.append(
        f"<b>Скачано:</b> {_messages(max(0, added))}, докачивается {snapshot.backfill_pending}"
    )
    if snapshot.failed:
        lines += ["", "<b>Не получилось:</b>"]
        lines += [f"• @{ref} — {error or '?'}" for ref, error in snapshot.failed[:6]]
    lines += ["", "#разведка #статус"]
    return "\n".join(lines)


def _render_finished(snapshot: ReconSnapshot, state: ReconStatusState, *, now: datetime) -> str:
    newly = tuple(ref for ref in snapshot.joined if ref not in state.baseline_joined)
    added = snapshot.messages_stored - state.baseline_messages
    lines = ["🛰 <b>Разведка завершена</b>", ""]
    if state.started_at is not None:
        hours = max(0, int((now - state.started_at).total_seconds() // 3600))
        lines.append(f"<b>Заняло:</b> ~{hours} {_plural(hours, 'час', 'часа', 'часов')}")
    lines.append(f"<b>Вступил в {len(newly)} {_plural(len(newly), 'чат', 'чата', 'чатов')}</b>")
    lines += _chat_list(newly)
    lines.append(f"<b>Скачано:</b> {_messages(max(0, added))}")
    if snapshot.oldest and snapshot.newest:
        lines.append(f"<b>Глубина архива:</b> {snapshot.oldest[:10]} .. {snapshot.newest[:10]}")
    if snapshot.requested:
        lines += ["", f"<b>Ждут одобрения админа ({len(snapshot.requested)}):</b>"]
        lines += _chat_list(snapshot.requested)
    if snapshot.failed:
        lines += ["", "<b>Не получилось:</b>"]
        lines += [f"• @{ref} — {error or '?'}" for ref, error in snapshot.failed[:6]]
    lines += [
        "",
        f"Всего читаю {len(snapshot.joined)} чатов, {_messages(snapshot.messages_stored)}.",
    ]
    lines += ["", "#разведка #статус"]
    return "\n".join(lines)


def _parse(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
