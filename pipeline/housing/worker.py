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
import logging
from typing import Any

from pipeline.housing.extractor import HousingTextExtractor
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
        await self._store.set_unit_state(unit.unit_key, UnitState.EXTRACTING)
        facts = await self._extractor.extract(unit.assembled_text or "")
        row = facts.as_row(unit_version=unit.unit_version)
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
        await self._store.set_unit_state(unit.unit_key, UnitState.DONE)

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


def render_alert(unit: ContentUnit, facts: dict[str, Any], result: MatchResult) -> str:
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
