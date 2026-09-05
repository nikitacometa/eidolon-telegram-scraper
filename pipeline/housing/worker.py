"""The housing loop: settled units become facts, verdicts and alerts.

Everything expensive lives here rather than on the ingestion path. A message
arriving from Telegram does one database write and returns; this worker picks
the advertisement up once its album siblings have had time to land, reads it,
judges it, and queues an alert. A slow model call delays one listing instead of
stalling every watcher in the daemon.

The loop is restart-safe by construction: each step is a state transition
written to the database before the next begins, so a daemon killed mid-way
resumes from the last state rather than from nothing.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from pipeline.housing.extractor import HousingTextExtractor
from pipeline.housing.gate import could_be_housing
from pipeline.housing.owner_transport import OwnerTransport, SendOutcome, SendStatus
from pipeline.housing.requirements import (
    DEFAULT_REQUIREMENTS,
    FieldState,
    MatchResult,
    format_thb,
    match_requirements,
)
from pipeline.housing.vision import MAX_IMAGES as VISION_MAX_IMAGES
from pipeline.models import DeliveryResult
from storage.housing import (
    AlertKind,
    ContentUnit,
    ForwardStatus,
    HousingStore,
    UnitOrigin,
    UnitState,
    Verdict,
)

logger = logging.getLogger(__name__)

POLL_SECONDS = 1.0
# How long the requirements poller waits between scalar reads, matching the
# agent-watcher sync's cadence: an edit takes effect within half a minute
# without a restart.
REQUIREMENTS_POLL_SECONDS = 30.0

# The owner reads dates in island time, not the server's UTC.
OWNER_TZ = ZoneInfo("Asia/Bangkok")
# The last line of a report whose original is forwarded right after it.
ORIGINAL_FOLLOWS_LINE = "⬇️ Оригинал ниже"
_MONTHS_RU = ("янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")


class HousingWorker:
    """Finalizes content units and turns them into alerts."""

    def __init__(
        self,
        *,
        store: HousingStore,
        extractor: HousingTextExtractor | None = None,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self._store = store
        self._extractor = extractor or HousingTextExtractor()
        self._poll_seconds = poll_seconds
        self._requirements_generation: int | None = None

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        """Work settled units until asked to stop."""
        logger.info("Housing worker started")
        while not shutdown.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A housing problem must never take the monitor down with it.
                logger.exception("Housing cycle failed")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                continue
        logger.info("Housing worker stopped")

    async def run_once(self) -> int:
        """Process every unit whose quiet window has closed. Returns the count."""
        units = await self._store.claim_settled_units()
        for unit in units:
            try:
                await self._process(unit)
            except Exception as error:
                logger.exception("Housing unit %s failed", unit.unit_key)
                await self._store.set_unit_state(
                    unit.unit_key, UnitState.ERROR, error=type(error).__name__
                )
        try:
            await self._rematch_if_edited()
        except Exception:
            logger.exception("Requirements re-match failed")
        return len(units)

    async def _rematch_if_edited(self) -> None:
        """Re-judge the retained archive after the owner edits requirements.

        Judging is a pure function over facts already paid for, so the sweep
        is free of model calls. The marker generation is persisted: an edit
        made while the daemon was down is still noticed on the next boot,
        and a completed sweep is never repeated. Listings that a LOOSENED
        requirement newly admits are reported once, as one digest message —
        not as one alert each, which for a broad edit would be a hundred
        messages teaching the owner to mute the bot.
        """
        generation = await self._store.pending_rematch_generation()
        if generation is None:
            return
        revision, requirements = await self._active_requirements()
        upgraded: list[tuple[ContentUnit, dict[str, Any], MatchResult]] = []
        matches: list[tuple[str, Verdict, dict[str, Any]]] = []
        for unit_key in await self._store.units_for_rematch():
            facts = await self._store.get_facts(unit_key)
            unit = await self._store.get_unit(unit_key)
            if facts is None or unit is None:
                continue
            previous = await self._store.latest_match(unit_key)
            previously_rejected = previous is None or (
                str(previous["verdict"]) == Verdict.HARD_MISS.value
            )
            result = match_requirements(facts, requirements)
            matches.append((unit_key, result.verdict, result.as_dict()))
            if result.verdict is not Verdict.HARD_MISS and previously_rejected:
                upgraded.append((unit, facts, result))

        alert = None
        if upgraded:
            origins = {
                unit.unit_key: await self._store.unit_origin(unit.unit_key)
                for unit, _facts, _result in upgraded
            }
            alert = {
                # A synthetic key scoped to the revision: re-running the same
                # sweep after a crash deduplicates instead of re-sending.
                "unit_key": f"rematch:{revision}",
                "chat_id": 0,
                "chat_title": None,
                "telegram_msg_id": 0,
                "verdict": Verdict.POSSIBLE,
                "kind": AlertKind.DIGEST,
                "body_html": render_rematch_digest(revision, upgraded, origins=origins),
            }
        # One transaction: the verdicts, the digest, and the generation
        # marker land together, so a crash mid-sweep repeats the whole sweep
        # against the still-previous verdicts instead of losing listings.
        await self._store.record_rematch(
            revision=revision,
            generation=generation,
            matches=matches,
            alert=alert,
        )
        logger.info(
            "Re-matched archive at generation %d: %d newly admitted",
            generation,
            len(upgraded),
        )

    async def _process(self, unit: ContentUnit) -> None:
        """One advertisement, from assembled text to a queued alert."""
        if not await self._worth_reading(unit):
            await self._store.set_unit_state(unit.unit_key, UnitState.DONE)
            return

        await self._store.set_unit_state(unit.unit_key, UnitState.EXTRACTING)
        facts = await self._extractor.extract(unit.assembled_text or "")
        row = facts.as_row(unit_version=unit.unit_version, source_text=unit.assembled_text)
        await self._store.record_facts(unit.unit_key, row)
        await self._store.set_unit_state(unit.unit_key, UnitState.EXTRACTED)

        if facts.is_rental_offer is not True or facts.is_vehicle_ad is True:
            # Not an offer of a place to live: a wanted-ad, a sale, a scooter.
            # An extraction that failed outright lands here too — is_rental_offer
            # is None — which is the one place unknown does mean "no alert",
            # because there is no evidence any advertisement exists at all.
            await self._store.set_unit_state(unit.unit_key, UnitState.DONE)
            return

        await self._store.set_unit_state(unit.unit_key, UnitState.MATCHING)
        revision, requirements = await self._active_requirements()
        result = match_requirements(row, requirements)
        alert = None
        if result.verdict is not Verdict.HARD_MISS:
            origin = await self._store.unit_origin(unit.unit_key)
            alert = {
                "chat_id": unit.chat_id,
                "chat_title": origin.chat_title,
                "telegram_msg_id": unit.members[0].telegram_msg_id if unit.members else 0,
                "verdict": result.verdict,
                "kind": AlertKind.LIVE,
                "body_html": render_alert(unit, row, result, origin=origin),
            }
        # One transaction: a verdict without its alert must not survive a
        # death between two separate commits.
        await self._store.record_match_with_alert(
            unit_key=unit.unit_key,
            requirements_revision=revision,
            verdict=result.verdict,
            field_verdicts=result.as_dict(),
            alert=alert,
        )

        if result.verdict is not Verdict.HARD_MISS:
            # Photographs are asked for AFTER the alert is queued, never
            # before. A listing whose bathrooms are unstated is worth telling
            # the owner about immediately; waiting for a download would delay
            # every listing on the island for a fact that may not be in the
            # pictures anyway.
            await self._request_photographs(unit, result)
        await self._store.set_unit_state(unit.unit_key, UnitState.DONE)

    async def _request_photographs(self, unit: ContentUnit, result: MatchResult) -> None:
        """Queue this unit's photographs when they could answer an open question.

        Photographs can answer a hard question (a standalone house is
        recognisable, bathrooms can be counted as a lower bound) and two of
        the preferences (a television, a terrace). Anything else in the
        pictures is not worth the download budget.
        """
        answerable = ({"bathrooms", "property_type"} & set(result.unknown_fields)) | (
            {"tv", "terrace"} & set(result.unknown_preferences)
        )
        if not answerable:
            return
        photos = [
            (member.telegram_msg_id, member.telegram_photo_id)
            for member in unit.members
            if member.telegram_photo_id is not None
        ]
        if not photos:
            return
        # Vision reads at most VISION_MAX_IMAGES frames; downloading a
        # fifteen-photo album spends download budget on frames nobody will
        # ever look at.
        photos = photos[:VISION_MAX_IMAGES]
        queued = await self._store.enqueue_media(
            unit_key=unit.unit_key,
            chat_id=unit.chat_id,
            photos=photos,
        )
        if queued:
            logger.info(
                "Queued %d photo(s) for %s to answer %s",
                queued,
                unit.unit_key,
                ", ".join(sorted(answerable)),
            )

    async def _worth_reading(self, unit: ContentUnit) -> bool:
        """Decide whether this message is worth a model call at all.

        On a dedicated rentals board the answer is always yes: everything
        posted there is a candidate. On a general chat the lexical gate runs,
        because those chats are large and mostly conversation — see
        pipeline/housing/gate.py for what it drops and what that was measured
        against.
        """
        kind = await self._store.chat_kind(unit.chat_id)
        if kind == "dedicated_housing":
            return True
        return could_be_housing(unit.assembled_text)

    async def _active_requirements(self) -> tuple[int, dict[str, Any]]:
        """The active revision, seeding the owner's stated criteria on first use."""
        active = await self._store.active_requirements()
        if active is not None:
            return active
        revision = await self._store.save_requirements(
            definition=DEFAULT_REQUIREMENTS,
            created_by="config-default",
        )
        return revision, DEFAULT_REQUIREMENTS


def format_when(moment: datetime, *, now: datetime | None = None) -> str:
    """'4 сен 17:04 (вчера)' in the owner's time zone.

    The relative part is what makes a replayed listing readable as a ledger
    entry: the owner's question is "how stale is this", and a bare date makes
    him do the arithmetic.
    """
    reference = now or datetime.now(UTC)
    local = moment.astimezone(OWNER_TZ)
    today = reference.astimezone(OWNER_TZ).date()
    stamp = f"{local.day} {_MONTHS_RU[local.month - 1]}"
    if local.year != today.year:
        stamp += f" {local.year}"
    stamp += f" {local:%H:%M}"
    days = (today - local.date()).days
    if days <= 0:
        relative = "сегодня"
    elif days == 1:
        relative = "вчера"
    else:
        relative = f"{days} {_plural(days, 'день', 'дня', 'дней')} назад"
    return f"{stamp} ({relative})"


def _plural(count: int, one: str, few: str, many: str) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return one
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


def render_origin_line(origin: UnitOrigin | None, *, now: datetime | None = None) -> str | None:
    """'📅 Опубликовано 4 сен 17:04 (вчера) · Chat · @author' — the ledger line."""
    if origin is None:
        return None
    parts: list[str] = []
    if origin.posted_at is not None:
        parts.append(f"Опубликовано {format_when(origin.posted_at, now=now)}")
    if origin.chat_title:
        parts.append(html.escape(origin.chat_title))
    if origin.sender_name and origin.sender_name != "Unknown":
        parts.append(html.escape(origin.sender_name))
    if not parts:
        return None
    return "📅 " + " · ".join(parts)


def render_alert(
    unit: ContentUnit,
    facts: dict[str, Any],
    result: MatchResult,
    *,
    photos_read: int = 0,
    origin: UnitOrigin | None = None,
    replayed_from: datetime | None = None,
    original_follows: bool = True,
    now: datetime | None = None,
) -> str:
    """Format the report the owner actually reads.

    Every criterion is named with its state, including the ones nobody could
    answer. A listing reaching him as "possible" with bathrooms and TV unknown
    is the normal case on this island — the point of the message is to tell him
    precisely what is known, so that the one question he has to ask the poster
    is short.

    The report carries no quotation and no t.me link: the original message is
    forwarded right after it, in full, and a link into a chat the owner has not
    joined is a link he cannot open. The date line says when the advertisement
    was posted, so a listing read days later is read as days old.
    """
    icons = {
        FieldState.SATISFIED: "✅",
        FieldState.VIOLATED: "❌",
        FieldState.UNKNOWN: "❔",
    }
    headline = "🏠 Совпадение" if result.verdict is Verdict.CONFIRMED else "🏠 Возможно подходит"
    lines = [f"<b>{headline}</b>"]
    if photos_read:
        lines.append(f"<i>Уточнено по {photos_read} фото</i>")

    price = facts.get("monthly_price_thb")
    if isinstance(price, int):
        lines.append(f"<b>{format_thb(price)} THB/мес</b>")

    for field in result.fields:
        lines.append(f"{icons[field.state]} {_label(field.field)}: {html.escape(field.detail)}")

    if result.preferences:
        marks = []
        for pref in result.preferences:
            mark = icons[pref.state]
            marks.append(f"{mark} {_label(pref.field).lower()}")
        lines.append(f"🎯 Хотелки {result.preference_score}%: " + ", ".join(marks))

    area = facts.get("area_raw")
    if area:
        lines.append(f"📍 {html.escape(str(area))}")

    origin_line = render_origin_line(origin, now=now)
    if origin_line:
        lines.append(origin_line)
    if replayed_from is not None:
        lines.append(f"🔁 Повтор: первый алерт был {format_when(replayed_from, now=now)}")
    if original_follows and unit.members:
        lines.append(ORIGINAL_FOLLOWS_LINE)
    return "\n".join(lines)


def render_rejection(
    result: MatchResult,
    *,
    origin: UnitOrigin | None = None,
    now: datetime | None = None,
) -> str:
    """The follow-up when photographs show a listing does not fit after all."""
    lines = ["<b>🏠 Отбой по объявлению</b>", "По фото видно, что не подходит:"]
    lines += [
        f"❌ {_label(field.field)}: {html.escape(field.detail)}"
        for field in result.fields
        if field.state is FieldState.VIOLATED
    ]
    origin_line = render_origin_line(origin, now=now)
    if origin_line:
        lines.append(origin_line)
    return "\n".join(lines)


def render_rematch_digest(
    revision: int,
    upgraded: list[tuple[ContentUnit, dict[str, Any], MatchResult]],
    *,
    origins: dict[str, UnitOrigin] | None = None,
    shown: int = 10,
    now: datetime | None = None,
) -> str:
    """One message for everything a loosened requirement newly admits.

    Each line names the listing by what the owner can act on — price,
    bedrooms, area, score, when and where it was posted — rather than by a
    link into a chat he cannot open.
    """
    lines = [
        f"<b>🏠 Требования обновлены (ревизия {revision})</b>",
        f"Теперь подходят ещё {len(upgraded)} объявлений из архива:",
    ]
    for unit, facts, result in upgraded[:shown]:
        bits = []
        price = facts.get("monthly_price_thb")
        if isinstance(price, int):
            bits.append(f"{format_thb(price)} THB")
        bedrooms = facts.get("bedrooms")
        if isinstance(bedrooms, int):
            bits.append(f"{bedrooms}BR")
        area = facts.get("area_raw")
        if area:
            bits.append(html.escape(str(area)))
        bits.append(f"🎯{result.preference_score}%")
        origin = (origins or {}).get(unit.unit_key)
        if origin is not None:
            if origin.posted_at is not None:
                bits.append(format_when(origin.posted_at, now=now))
            if origin.chat_title:
                bits.append(html.escape(origin.chat_title))
        lines.append("• " + " · ".join(bits))
    if len(upgraded) > shown:
        lines.append(f"…и ещё {len(upgraded) - shown}.")
    return "\n".join(lines)


def _label(field: str) -> str:
    return {
        "bedrooms": "Спальни",
        "bathrooms": "Ванные",
        "tv": "Телевизор",
        "monthly_rent_thb": "Цена",
        "property_type": "Тип жилья",
        "terrace": "Терраса",
        "private_setting": "Приватность",
        "nature_setting": "Природа",
    }.get(field, field)


def message_link(chat_id: int, telegram_msg_id: int) -> str | None:
    """Build a t.me link to the message, when the chat id allows one.

    Supergroup ids are the channel id with a -100 prefix; private groups have
    no public link form at all, and inventing one would hand the owner a dead
    link, so those get none. Used only on the bot fallback path: the owner
    can open it from the account that is a member, nobody else can.
    """
    raw = str(chat_id)
    if not raw.startswith("-100") or telegram_msg_id <= 0:
        return None
    return f"https://t.me/c/{raw[4:]}/{telegram_msg_id}"


def _message_link(unit: ContentUnit) -> str | None:
    if not unit.members:
        return None
    return message_link(unit.chat_id, unit.members[0].telegram_msg_id)


# Governor answers that mean "come back later" rather than "this failed": the
# pace, the budget, a FloodWait Telegram lifts by itself, or an account-wide
# halt a person has to clear. None of them says anything about THIS alert, so
# none of them spends one of its bounded attempts — a listing must not be
# dropped because Telegram asked the account to wait six times.
NOT_YET_CODES = frozenset(
    {
        "paced",
        "budget",
        "flood_wait",
        "halted",
        "channels_too_much",
        "PeerFloodError",
        "UserBannedInChannelError",
    }
)

# Why a forward was refused, in the owner's language.
_FORWARD_REFUSALS_RU = {
    "ChatForwardsRestrictedError": "чат запрещает пересылку",
    "MessageIdInvalidError": "сообщение удалено",
    "MediaEmptyError": "медиа больше недоступно",
    "nothing_to_forward": "исходное сообщение не сохранилось",
}


class HousingAlertDelivery:
    """Drains the housing outbox: a report to the owner, then the original.

    A separate outbox from the generic one, because the generic claim query
    inner-joins an alert to exactly one message row and a listing alert has no
    single message — an album is several, and a follow-up after a photograph is
    read belongs to the advertisement rather than to any one of them.

    Delivery is two Telegram calls with a durable checkpoint between them. The
    report's message id is written the moment Telegram accepts it; a crash or
    a FloodWait after that point resumes at the forward, so the owner never
    receives the same report twice and never receives an original without its
    report above it.
    """

    def __init__(
        self,
        *,
        store: HousingStore,
        dispatcher: Any,
        lease_owner: str,
        owner: OwnerTransport | None = None,
        poll_seconds: float = 5.0,
        max_attempts: int = 6,
    ) -> None:
        self._store = store
        self._dispatcher = dispatcher
        self._owner = owner
        self._lease_owner = lease_owner
        self._poll_seconds = poll_seconds
        self._max_attempts = max_attempts

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        """Deliver queued housing alerts until asked to stop."""
        logger.info("Housing alert delivery started")
        while not shutdown.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Housing delivery cycle failed")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                continue
        logger.info("Housing alert delivery stopped")

    async def run_once(self) -> int:
        """Deliver whatever is due. Returns how many were sent."""
        due = await self._store.claim_due_alerts(lease_owner=self._lease_owner)
        sent = 0
        for alert in due:
            alert_id = int(alert["id"])
            try:
                result = await self._deliver(alert)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # One alert's surprise must not hold the rest of the batch
                # behind its lease. It is settled as a transient failure and
                # retried with backoff like any other.
                logger.exception("Housing alert %d raised during delivery", alert_id)
                result = DeliveryResult(
                    sent=False, retryable=True, error_code=type(error).__name__, retry_after=60
                )
            if result.sent:
                await self._store.settle_alert(alert_id, delivered=True)
                sent += 1
                continue
            if result.error_code in NOT_YET_CODES:
                # Pacing and budget are "not yet", never "failed": the row
                # comes back when the governor says so, and the attempt
                # counter — which bounds real failures — stays where it is.
                await self._store.settle_alert(
                    alert_id,
                    delivered=False,
                    error=result.error_code,
                    retry_in_seconds=max(1, int(result.retry_after or 2)),
                    count_attempt=False,
                )
                continue
            # The stored counter holds completed attempts (settle_alert
            # increments it); the delivery that just failed makes one more.
            attempts = int(alert["attempts"]) + 1
            if not result.retryable or attempts >= self._max_attempts:
                if alert.get("report_message_id") is not None:
                    # The owner has the report. Giving up on the original is
                    # a fact to tell him, not a failed row to hide: the report
                    # is amended and the alert closes as delivered.
                    await self._abandon_original(alert, reason=result.error_code)
                    await self._store.settle_alert(alert_id, delivered=True)
                    sent += 1
                    continue
                # Stop retrying, but keep the row: a failed alert is a listing
                # the owner never saw, and that has to stay visible rather than
                # disappear into a log line.
                await self._store.settle_alert(alert_id, delivered=False, error=result.error_code)
                logger.error(
                    "Housing alert %d abandoned after %d attempts: %s",
                    alert_id,
                    attempts,
                    result.error_code,
                )
                continue
            await self._store.settle_alert(
                alert_id,
                delivered=False,
                error=result.error_code,
                retry_in_seconds=max(int(result.retry_after or 0), 2**attempts),
            )
        return sent

    async def _deliver(self, alert: dict[str, Any]) -> DeliveryResult:
        """One alert, both steps, resuming wherever the previous attempt stopped."""
        if self._owner is None or not self._owner.configured:
            return await self._deliver_via_bot(alert)

        alert_id = int(alert["id"])
        unit_key = str(alert["unit_key"])
        kind = AlertKind(str(alert["kind"]))
        chat_id = int(alert["chat_id"])
        # A follow-up replies to the report it follows up on. Whether one
        # exists is read here, once, and decides both the reply target and
        # whether the original still has to be shown.
        prior_report = None
        carries_original = chat_id != 0 and kind in (AlertKind.LIVE, AlertKind.REPLAY)
        if kind is AlertKind.UPDATE:
            prior_report = await self._store.report_message_id(unit_key, exclude_alert_id=alert_id)
            if chat_id != 0 and not await self._store.unit_reported(
                unit_key, exclude_alert_id=alert_id
            ):
                # The owner never got a report for this listing (the first
                # alert failed, or predates this format): show him the
                # original now. Judged on whether ANY report went out, not on
                # whether one is reply-able — a report whose id came back
                # unknown was still read.
                carries_original = True

        report_id = alert.get("report_message_id")
        if report_id is None:
            body = str(alert["body_html"])
            if carries_original and ORIGINAL_FOLLOWS_LINE not in body:
                # The body was rendered as a follow-up (no original expected);
                # delivery decided otherwise, and the report has to agree.
                body = body.rstrip() + "\n" + ORIGINAL_FOLLOWS_LINE
            outcome = await self._owner.send_report(body, reply_to=prior_report)
            if outcome.status is SendStatus.UNREACHABLE:
                return await self._deliver_via_bot(alert)
            if not outcome.sent:
                return _retry(outcome)
            # 0 stands for "sent, id unknown": the checkpoint is the fact of
            # sending, not the id, and a retry must not send again.
            report_id = outcome.message_id or 0
            await self._store.mark_report_sent(alert_id, message_id=report_id)
            # run_once keeps this same dict for its bookkeeping; it has to
            # see that a report now exists, or a forward that fails on the
            # last attempt is closed as "nothing was ever sent".
            alert["report_message_id"] = report_id

        if str(alert.get("forward_status") or ForwardStatus.PENDING) != ForwardStatus.PENDING:
            return DeliveryResult.success()

        if not carries_original:
            await self._store.set_forward_status(alert_id, ForwardStatus.SKIPPED)
            return DeliveryResult.success()

        unit = await self._store.get_unit(unit_key)
        message_ids = (
            [member.telegram_msg_id for member in unit.members]
            if unit is not None and unit.members
            else [int(alert["telegram_msg_id"])]
        )
        forwarded = await self._owner.forward(chat_id=chat_id, message_ids=message_ids)
        if forwarded.sent:
            error = None
            if forwarded.missing:
                # Part of the album is gone. The owner sees what arrived; the
                # report says how much did not, instead of a quietly shorter
                # album read as the whole advertisement.
                error = f"partial:{forwarded.missing}/{len(message_ids)}"
                await self._amend_report(
                    alert,
                    int(report_id),
                    f"⚠️ Часть оригинала недоступна: {forwarded.missing} из "
                    f"{len(message_ids)} сообщений удалены",
                )
            await self._store.set_forward_status(alert_id, ForwardStatus.FORWARDED, error=error)
            return DeliveryResult.success()
        if forwarded.status is SendStatus.RETRY:
            return _retry(forwarded)
        if forwarded.status is SendStatus.UNREACHABLE:
            # The report went through moments ago; an unreachable owner now
            # is a transient contradiction, not a verdict. Try again later.
            return DeliveryResult(
                sent=False,
                retryable=True,
                error_code=forwarded.error_code or "owner_unreachable",
                retry_after=300,
            )

        # Telegram refused THIS message: the chat forbids forwarding, or the
        # advertisement was deleted. Send what the daemon read instead.
        reason = _FORWARD_REFUSALS_RU.get(
            forwarded.error_code or "", forwarded.error_code or "неизвестная причина"
        )
        author = html.escape(str(alert.get("chat_title") or ""))
        header = f"📎 <b>Копия объявления</b> — оригинал переслать нельзя: {html.escape(reason)}"
        if author:
            header += f"\n💬 {author}"
        copied = await self._owner.send_copy(
            text=unit.assembled_text if unit is not None else None,
            photo_paths=await self._store.downloaded_media(unit_key),
            header_html=header,
            reply_to=int(report_id) if report_id else None,
        )
        if copied.sent:
            await self._store.set_forward_status(
                alert_id, ForwardStatus.COPIED, error=forwarded.error_code
            )
            return DeliveryResult.success()
        if copied.status is SendStatus.RETRY:
            return _retry(copied)
        await self._abandon_original(alert, reason=forwarded.error_code, report_id=int(report_id))
        return DeliveryResult.success()

    async def _abandon_original(
        self,
        alert: dict[str, Any],
        *,
        reason: str | None,
        report_id: int | None = None,
    ) -> None:
        """Close the forward step without the original, and say so on the report."""
        alert_id = int(alert["id"])
        await self._store.set_forward_status(alert_id, ForwardStatus.UNAVAILABLE, error=reason)
        message_id = report_id if report_id is not None else alert.get("report_message_id")
        if self._owner is None or not message_id:
            return
        why = _FORWARD_REFUSALS_RU.get(reason or "", reason or "не удалось переслать")
        await self._amend_report(
            alert, int(message_id), f"⚠️ Оригинал недоступен: {html.escape(why)}"
        )

    async def _amend_report(self, alert: dict[str, Any], message_id: int, note: str) -> None:
        """Rewrite the report in the owner's DM with a note about the original."""
        if self._owner is None or not message_id:
            return
        body = str(alert["body_html"]).replace(ORIGINAL_FOLLOWS_LINE, "").rstrip()
        # The edit follows a copy attempt of the same kind, so the governor's
        # per-kind pace can say "not yet" once; that is waited out rather than
        # dropped, because the note IS the owner's only sign the original is
        # missing. Anything beyond a couple of waits is logged and let go.
        for _ in range(3):
            edited = await self._owner.edit_report(message_id, body + "\n" + note)
            if edited.sent:
                return
            if edited.status is not SendStatus.RETRY:
                break
            await asyncio.sleep(max(1, int(edited.retry_after or 2)))
        logger.warning(
            "Could not amend report %s for alert %d: %s",
            message_id,
            int(alert["id"]),
            edited.error_code,
        )

    async def _deliver_via_bot(self, alert: dict[str, Any]) -> DeliveryResult:
        """The pre-forward path: the report through the bot, with a link.

        Used when no owner is configured, and as the fallback when the owner's
        DM cannot be written to. The link only works from an account that is
        a member of the chat, which is why this is the fallback and not the
        design — but a report with a dead link beats no report.
        """
        body = str(alert["body_html"]).replace(ORIGINAL_FOLLOWS_LINE, "").rstrip()
        link = message_link(int(alert["chat_id"]), int(alert["telegram_msg_id"]))
        if link:
            body += f'\n<a href="{link}">Открыть в Telegram</a>'
        photos = _photo_paths(alert.get("photo_paths_json"))
        result: DeliveryResult
        if photos:
            # The photograph the claim was read from travels with the
            # claim: it is the only check available on a visual answer.
            result = await self._dispatcher.deliver_photo(body, photos[0])
        else:
            result = await self._dispatcher.deliver_html(body)
        if result.sent:
            status = (
                ForwardStatus.BOT_FALLBACK
                if self._owner is not None and self._owner.configured
                else ForwardStatus.SKIPPED
            )
            await self._store.set_forward_status(int(alert["id"]), status)
        return result


def _retry(outcome: SendOutcome) -> DeliveryResult:
    """A transient refusal from the owner transport, as the outbox reads it."""
    return DeliveryResult(
        sent=False,
        retryable=True,
        error_code=outcome.error_code or "owner_retry",
        retry_after=max(1, int(outcome.retry_after or 30)),
    )


class HousingVisionWorker:
    """Reads downloaded photographs and upgrades a verdict when they answer it.

    The second half of a two-timeline subsystem: the text was judged minutes
    ago and the owner already has an alert saying what was unknown. This loop
    closes those unknowns when the pictures can, and tells him again only when
    the answer actually changed the verdict.
    """

    def __init__(
        self,
        *,
        store: HousingStore,
        extractor: Any,
        poll_seconds: float = 20.0,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._poll_seconds = poll_seconds

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        """Read photographs until asked to stop."""
        logger.info("Housing vision worker started")
        while not shutdown.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Housing vision cycle failed")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                continue
        logger.info("Housing vision worker stopped")

    async def run_once(self) -> int:
        """Read every unit whose photographs are in. Returns how many were read."""
        keys = await self._store.units_awaiting_vision()
        for unit_key in keys:
            try:
                await self._read(unit_key)
            except Exception:
                logger.exception("Vision read failed for %s", unit_key)
                facts = await self._store.get_facts(unit_key)
                if facts is not None:
                    facts["vision_status"] = "error"
                    await self._store.record_facts(unit_key, facts)
        return len(keys)

    async def _read(self, unit_key: str) -> None:
        """One unit: look at its photographs, re-judge, tell the owner if it changed."""
        unit = await self._store.get_unit(unit_key)
        facts = await self._store.get_facts(unit_key)
        if unit is None or facts is None:
            return
        paths = await self._store.downloaded_media(unit_key)
        if not paths:
            facts["vision_status"] = "unavailable"
            await self._store.record_facts(unit_key, facts)
            return

        previous = await self._store.latest_match(unit_key)
        previous_verdict = str(previous["verdict"]) if previous else None

        reading = await self._extractor.read(paths, listing_text=unit.assembled_text)
        merged = reading.merged_into(facts)
        # The final status ('done'/'error') is committed only after the
        # re-match and any alert are safely recorded. Committing it first
        # would remove the unit from units_awaiting_vision, so a crash
        # between the writes would strand it with a stale verdict and no
        # alert, forever. Until then the row stays 'pending': a crash
        # anywhere below merely repeats the (idempotent) read next cycle.
        final_status = merged["vision_status"]
        merged["vision_status"] = "pending"
        await self._store.record_facts(unit_key, merged)

        async def finalize() -> None:
            merged["vision_status"] = final_status
            await self._store.record_facts(unit_key, merged)

        active = await self._store.active_requirements()
        if active is None:
            await finalize()
            return
        revision, requirements = active
        result = match_requirements(merged, requirements)

        alert = None
        if result.verdict.value != previous_verdict:
            # An unchanged verdict sends nothing: the photographs added detail
            # but changed nothing the owner has to act on, and repeating a
            # verdict trains him to ignore these messages. The comparison is
            # against the verdict read BEFORE this pass wrote anything, and
            # the verdict and its alert are committed together below — so an
            # interrupted pass either recorded both or neither, and the retry
            # compares against the right thing either way.
            origin = await self._store.unit_origin(unit_key)
            if result.verdict is Verdict.HARD_MISS:
                body = render_rejection(result, origin=origin)
                verdict_for_alert = Verdict.POSSIBLE
            else:
                # A reply to the report already in the owner's DM, so the
                # original is not shown twice.
                body = render_alert(
                    unit,
                    merged,
                    result,
                    photos_read=len(paths),
                    origin=origin,
                    original_follows=False,
                )
                verdict_for_alert = result.verdict
            alert = {
                "chat_id": unit.chat_id,
                "chat_title": origin.chat_title,
                "telegram_msg_id": unit.members[0].telegram_msg_id if unit.members else 0,
                "verdict": verdict_for_alert,
                "kind": AlertKind.UPDATE,
                "body_html": body,
                "photo_paths": paths[:3] if result.verdict is not Verdict.HARD_MISS else None,
            }

        await self._store.record_match_with_alert(
            unit_key=unit_key,
            requirements_revision=revision,
            verdict=result.verdict,
            field_verdicts=result.as_dict(),
            alert=alert,
        )
        await finalize()


def _photo_paths(raw: Any) -> list[str]:
    """Read the stored photo list, tolerating a row that has none."""
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return [str(path) for path in parsed] if isinstance(parsed, list) else []
