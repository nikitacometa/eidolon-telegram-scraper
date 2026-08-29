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
from typing import Any

from pipeline.housing.extractor import HousingTextExtractor
from pipeline.housing.gate import could_be_housing
from pipeline.housing.requirements import (
    DEFAULT_REQUIREMENTS,
    FieldState,
    MatchResult,
    match_requirements,
)
from storage.housing import AlertKind, ContentUnit, HousingStore, UnitState, Verdict

logger = logging.getLogger(__name__)

POLL_SECONDS = 1.0
# How long the requirements poller waits between scalar reads, matching the
# agent-watcher sync's cadence: an edit takes effect within half a minute
# without a restart.
REQUIREMENTS_POLL_SECONDS = 30.0


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
        return len(units)

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
        await self._store.record_match(
            unit_key=unit.unit_key,
            requirements_revision=revision,
            verdict=result.verdict,
            field_verdicts=result.as_dict(),
        )

        if result.verdict is not Verdict.HARD_MISS:
            await self._store.enqueue_alert(
                unit_key=unit.unit_key,
                chat_id=unit.chat_id,
                chat_title=None,
                telegram_msg_id=unit.members[0].telegram_msg_id if unit.members else 0,
                requirements_revision=revision,
                verdict=result.verdict,
                kind=AlertKind.LIVE,
                body_html=render_alert(unit, row, result),
            )
            # Photographs are asked for AFTER the alert is queued, never
            # before. A listing whose bathrooms are unstated is worth telling
            # the owner about immediately; waiting for a download would delay
            # every listing on the island for a fact that may not be in the
            # pictures anyway.
            await self._request_photographs(unit, result)
        await self._store.set_unit_state(unit.unit_key, UnitState.DONE)

    async def _request_photographs(self, unit: ContentUnit, result: MatchResult) -> None:
        """Queue this unit's photographs when they could answer an open question."""
        answerable = {"bathrooms", "tv"} & set(result.unknown_fields)
        if not answerable:
            return
        photos = [
            (member.telegram_msg_id, member.telegram_photo_id)
            for member in unit.members
            if member.telegram_photo_id is not None
        ]
        if not photos:
            return
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


def render_alert(
    unit: ContentUnit,
    facts: dict[str, Any],
    result: MatchResult,
    *,
    photos_read: int = 0,
) -> str:
    """Format the alert the owner actually reads.

    Every criterion is named with its state, including the ones nobody could
    answer. A listing reaching him as "possible" with bathrooms and TV unknown
    is the normal case on this island — the point of the message is to tell him
    precisely what is known, so that the one question he has to ask the poster
    is short.
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
        lines.append(f"<b>{price:,} THB/мес</b>".replace(",", " "))

    for field in result.fields:
        lines.append(f"{icons[field.state]} {_label(field.field)}: {html.escape(field.detail)}")

    area = facts.get("area_raw")
    if area:
        lines.append(f"📍 {html.escape(str(area))}")

    quote = facts.get("evidence_quote")
    if quote:
        lines.append(f"\n<blockquote>{html.escape(str(quote)[:400])}</blockquote>")

    link = _message_link(unit)
    if link:
        lines.append(f'\n<a href="{link}">Открыть в Telegram</a>')
    return "\n".join(lines)


def _label(field: str) -> str:
    return {
        "bedrooms": "Спальни",
        "bathrooms": "Ванные",
        "tv": "Телевизор",
        "monthly_rent_thb": "Цена",
    }.get(field, field)


def _message_link(unit: ContentUnit) -> str | None:
    """Build a t.me link to the message, when the chat id allows one.

    Supergroup ids are the channel id with a -100 prefix; private groups have
    no public link form at all, and inventing one would hand the owner a dead
    link, so those get none.
    """
    if not unit.members:
        return None
    raw = str(unit.chat_id)
    if not raw.startswith("-100"):
        return None
    return f"https://t.me/c/{raw[4:]}/{unit.members[0].telegram_msg_id}"


class HousingAlertDelivery:
    """Drains the housing outbox through the shared bot transport.

    A separate outbox from the generic one, because the generic claim query
    inner-joins an alert to exactly one message row and a listing alert has no
    single message — an album is several, and a follow-up after a photograph is
    read belongs to the advertisement rather than to any one of them.
    """

    def __init__(
        self,
        *,
        store: HousingStore,
        dispatcher: Any,
        lease_owner: str,
        poll_seconds: float = 5.0,
        max_attempts: int = 6,
    ) -> None:
        self._store = store
        self._dispatcher = dispatcher
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
            photos = _photo_paths(alert.get("photo_paths_json"))
            if photos:
                # The photograph the claim was read from travels with the
                # claim: it is the only check available on a visual answer.
                result = await self._dispatcher.deliver_photo(str(alert["body_html"]), photos[0])
            else:
                result = await self._dispatcher.deliver_html(str(alert["body_html"]))
            alert_id = int(alert["id"])
            if result.sent:
                await self._store.settle_alert(alert_id, delivered=True)
                sent += 1
                continue
            attempts = int(alert["attempts"])
            if not result.retryable or attempts >= self._max_attempts:
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
        await self._store.record_facts(unit_key, merged)

        active = await self._store.active_requirements()
        if active is None:
            return
        revision, requirements = active
        result = match_requirements(merged, requirements)
        await self._store.record_match(
            unit_key=unit_key,
            requirements_revision=revision,
            verdict=result.verdict,
            field_verdicts=result.as_dict(),
        )

        if result.verdict.value == previous_verdict:
            # The photographs added detail but changed nothing the owner has
            # to act on. Sending the same verdict twice trains him to ignore
            # these messages, which is how a working filter becomes useless.
            return

        if result.verdict is Verdict.HARD_MISS:
            body = "<b>🏠 Отбой по объявлению</b>\nПо фото видно, что не подходит:\n" + "\n".join(
                f"❌ {_label(field.field)}: {html.escape(field.detail)}"
                for field in result.fields
                if field.state is FieldState.VIOLATED
            )
            verdict_for_alert = Verdict.POSSIBLE
        else:
            body = render_alert(unit, merged, result, photos_read=len(paths))
            verdict_for_alert = result.verdict

        await self._store.enqueue_alert(
            unit_key=unit_key,
            chat_id=unit.chat_id,
            chat_title=None,
            telegram_msg_id=unit.members[0].telegram_msg_id if unit.members else 0,
            requirements_revision=revision,
            verdict=verdict_for_alert,
            kind=AlertKind.UPDATE,
            body_html=body,
            photo_paths=paths[:3] if result.verdict is not Verdict.HARD_MISS else None,
        )


def _photo_paths(raw: Any) -> list[str]:
    """Read the stored photo list, tolerating a row that has none."""
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return [str(path) for path in parsed] if isinstance(parsed, list) else []
