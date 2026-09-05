"""Re-sending every listing the owner has been told about, in the current format.

The first weeks of alerts carried a t.me link the owner could not open; the
format changed to a report followed by the forwarded original. This module
queues those same listings again in that format — one report per listing,
dated twice: when the advertisement was posted and when the first alert about
it went out — so the owner reads the batch as a ledger rather than as news.

Two things are deliberately NOT done here. Nothing is sent: rows are queued in
the housing outbox and the daemon's delivery loop, with its pacing and its
two-step checkpoint, sends them. And nothing is re-extracted: the stored facts
are judged again against the ACTIVE requirements (a pure function, no model
call), so a listing that no longer fits is left out and listed in the plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pipeline.housing.requirements import match_requirements
from pipeline.housing.worker import format_when, render_alert
from storage.housing import AlertKind, HousingStore, UnitOrigin, Verdict

logger = logging.getLogger(__name__)

# Seconds between consecutive replayed listings in the outbox. Two messages
# per listing, so this is ten seconds per message — measured 2026-09-05: the
# six-second version, roughly three seconds per message, earned PEER_FLOOD on
# the 17th message to a DM the owner had never written to. It also keeps the
# reports in posted order when two delivery cycles overlap, because the outbox
# claims by next_attempt_at.
REPLAY_SPACING_SECONDS = 20


@dataclass(frozen=True, slots=True)
class ReplayItem:
    """One listing in the plan, with everything the report needs."""

    unit_key: str
    chat_id: int
    telegram_msg_id: int
    verdict: Verdict
    origin: UnitOrigin
    first_alert_at: datetime | None
    members: int
    body_html: str


@dataclass(slots=True)
class ReplayPlan:
    """What a replay would send, and what it would leave out."""

    items: list[ReplayItem] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # unit_key, reason
    already_queued: list[str] = field(default_factory=list)
    revision: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serializable form for the CLI."""
        return {
            "revision": self.revision,
            "items": [
                {
                    "unit_key": item.unit_key,
                    "chat": item.origin.chat_title,
                    "posted_at": item.origin.posted_at.isoformat()
                    if item.origin.posted_at
                    else None,
                    "first_alert_at": item.first_alert_at.isoformat()
                    if item.first_alert_at
                    else None,
                    "verdict": item.verdict.value,
                    "members": item.members,
                }
                for item in self.items
            ],
            "rejected": [{"unit_key": key, "reason": reason} for key, reason in self.rejected],
            "already_queued": list(self.already_queued),
        }


async def plan_replay(store: HousingStore, *, now: datetime | None = None) -> ReplayPlan:
    """Decide which alerted listings to send again and render their reports."""
    plan = ReplayPlan()
    active = await store.active_requirements()
    if active is None:
        return plan
    revision, requirements = active
    plan.revision = revision

    candidates = await store.alerted_units()
    for row in candidates:
        unit_key = str(row["unit_key"])
        if int(row["replays"] or 0) > 0:
            plan.already_queued.append(unit_key)
            continue
        unit = await store.get_unit(unit_key)
        facts = await store.get_facts(unit_key)
        if unit is None or facts is None:
            plan.rejected.append((unit_key, "unit or facts aged out"))
            continue
        if facts.get("is_rental_offer") != 1:
            plan.rejected.append((unit_key, "not a rental offer"))
            continue
        result = match_requirements(facts, requirements)
        if result.verdict is Verdict.HARD_MISS:
            violated = ", ".join(
                f"{verdict.field}: {verdict.detail}"
                for verdict in result.fields
                if verdict.state.value == "violated"
            )
            plan.rejected.append((unit_key, f"hard miss under revision {revision}: {violated}"))
            continue
        origin = await store.unit_origin(unit_key)
        first_alert_at = _parse(row.get("first_alert_at"))
        photos_read = 0
        if str(facts.get("vision_status") or "") == "done":
            photos_read = len(await store.downloaded_media(unit_key))
        plan.items.append(
            ReplayItem(
                unit_key=unit_key,
                chat_id=unit.chat_id,
                telegram_msg_id=unit.members[0].telegram_msg_id if unit.members else 0,
                verdict=result.verdict,
                origin=origin,
                first_alert_at=first_alert_at,
                members=len(unit.members),
                body_html=render_alert(
                    unit,
                    facts,
                    result,
                    photos_read=photos_read,
                    origin=origin,
                    replayed_from=first_alert_at,
                    now=now,
                ),
            )
        )
    # Oldest posting first: read top to bottom, the batch is a timeline.
    plan.items.sort(key=lambda item: item.origin.posted_at or datetime.min.replace(tzinfo=UTC))
    return plan


async def queue_replay(
    store: HousingStore, plan: ReplayPlan, *, now: datetime | None = None
) -> int:
    """Put the plan into the outbox. Returns how many listing rows were queued.

    A header row goes first so the owner knows the burst is a replay, not
    fifteen new houses. Rows are spaced out and deduplicated on
    (unit, verdict, 'replay'): running this twice queues nothing new.
    """
    if not plan.items:
        return 0
    moment = now or datetime.now(UTC)
    stamp = moment.strftime("%Y%m%d")
    await store.enqueue_alert(
        unit_key=f"replay-header:{stamp}",
        chat_id=0,
        chat_title=None,
        telegram_msg_id=0,
        requirements_revision=plan.revision,
        verdict=Verdict.POSSIBLE,
        kind=AlertKind.DIGEST,
        body_html=render_replay_header(plan, now=moment),
    )
    queued = 0
    for index, item in enumerate(plan.items, start=1):
        alert_id = await store.enqueue_alert(
            unit_key=item.unit_key,
            chat_id=item.chat_id,
            chat_title=item.origin.chat_title,
            telegram_msg_id=item.telegram_msg_id,
            requirements_revision=plan.revision,
            verdict=item.verdict,
            kind=AlertKind.REPLAY,
            body_html=item.body_html,
            delay_seconds=index * REPLAY_SPACING_SECONDS,
        )
        if alert_id is not None:
            queued += 1
    logger.info("Queued %d replayed listings (%d rejected)", queued, len(plan.rejected))
    return queued


def render_replay_header(plan: ReplayPlan, *, now: datetime | None = None) -> str:
    """The one message that explains the burst that follows it."""
    dated = [item.origin.posted_at for item in plan.items if item.origin.posted_at is not None]
    span = ""
    if dated:
        first, last = min(dated), max(dated)
        span = f" за {_day(first)} – {_day(last)}"
    lines = [
        "<b>👁 Это Эйдолон</b> — аккаунт, который читает чаты. Housing-алерты теперь "
        "приходят сюда, а не от бота: бот не состоит в этих чатах и не может "
        "переслать оригинал.",
        "",
        f"<b>🔁 Пересылаю {len(plan.items)} объявлений{span} в новом формате</b>",
        "К каждому — сначала отчёт, затем оригинал из чата (весь текст, фото, автор).",
        "В отчёте: когда объявление опубликовано и когда пришёл первый алерт — "
        "чтобы было видно, насколько оно устарело.",
        f"Все перепроверены по текущим требованиям (ревизия {plan.revision}).",
    ]
    if plan.rejected:
        lines.append(f"Не пересылаю {len(plan.rejected)}: по текущим требованиям уже не подходят.")
    if now is not None:
        lines.append(f"<i>{format_when(now, now=now)}</i>")
    return "\n".join(lines)


def _day(moment: datetime) -> str:
    return format_when(moment).split(" (")[0].rsplit(" ", 1)[0]


def _parse(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    try:
        parsed = datetime.fromisoformat(text.replace(" ", "T", 1) if " " in text else text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
