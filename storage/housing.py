"""Durable state for the housing search: content units, facts, matches, alerts.

The unit of housing is not a Telegram message. An advertisement posted as an
album arrives as several messages milliseconds apart, each carrying one photo
and at most one of them carrying the text, and the thing the owner wants to be
told about is the advertisement. So a unit is opened by the first message that
belongs to it, held open for a short quiet window in case siblings follow, and
finalized by a sweep that reads this table.

The sweep, rather than an in-process timer, is the whole point: the row is the
state. A daemon restarted between the first album member and the second loses
nothing, because the unfinished unit is still sitting here with a deadline in
the past, waiting to be picked up.

This module shares the live database's connection rather than opening its own:
two connections to one SQLite file serialize against each other at the file
lock, and the daemon already funnels its writes through one lock it owns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# How long a unit stays open waiting for album siblings. Telegram delivers the
# members of one album as separate updates a few hundred milliseconds apart;
# this is deliberately several times that, because the cost of waiting is a
# short delay on an alert and the cost of not waiting is an advertisement
# reported without its photographs.
DEFAULT_QUIET_WINDOW_SECONDS = 2.5


class UnitState(StrEnum):
    """Where a content unit is in its life."""

    ASSEMBLING = "assembling"
    READY = "ready"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    MATCHING = "matching"
    DONE = "done"
    ERROR = "error"


class Verdict(StrEnum):
    """How a unit stands against the owner's requirements."""

    CONFIRMED = "confirmed"
    POSSIBLE = "possible"
    HARD_MISS = "hard_miss"


class AlertKind(StrEnum):
    """Why an alert is being sent."""

    LIVE = "live"
    UPDATE = "update"
    DIGEST = "digest"


@dataclass(frozen=True, slots=True)
class UnitMember:
    """One Telegram message belonging to a content unit."""

    message_id: int
    telegram_msg_id: int
    ordinal: int
    has_media: bool
    telegram_photo_id: int | None
    has_text: bool


@dataclass(frozen=True, slots=True)
class ContentUnit:
    """One advertisement, however many messages carried it."""

    unit_key: str
    chat_id: int
    grouped_id: int | None
    unit_version: int
    representative_message_id: int
    assembled_text: str | None
    media_count: int
    state: UnitState
    members: tuple[UnitMember, ...] = ()

    @property
    def photo_ids(self) -> tuple[int, ...]:
        """Telegram photo ids carried by this unit, in arrival order."""
        return tuple(
            member.telegram_photo_id
            for member in self.members
            if member.telegram_photo_id is not None
        )


def unit_key_for(chat_id: int, *, grouped_id: int | None, telegram_msg_id: int) -> str:
    """Name the advertisement a message belongs to.

    A message with no ``grouped_id`` is its own unit. This is deliberately not
    Telethon's album event: that builder refuses to dispatch a group that only
    ever receives one member, so a single-photo post carrying a ``grouped_id``
    would be dropped entirely. Here the two cases differ only in the key.
    """
    if grouped_id is not None:
        return f"g:{chat_id}:{grouped_id}"
    return f"m:{chat_id}:{telegram_msg_id}"


class HousingStore:
    """Housing state, on the live database's connection."""

    def __init__(
        self,
        conn: aiosqlite.Connection,
        write_lock: Any,
        *,
        quiet_window_seconds: float = DEFAULT_QUIET_WINDOW_SECONDS,
    ) -> None:
        self._conn = conn
        self._write_lock = write_lock
        self._quiet_window = quiet_window_seconds

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    async def record_message(
        self,
        *,
        unit_key: str,
        chat_id: int,
        grouped_id: int | None,
        message_id: int,
        telegram_msg_id: int,
        text: str | None,
        has_media: bool,
        telegram_photo_id: int | None,
    ) -> None:
        """Attach one message to its unit, extending the quiet window.

        Called on the live path, so it does exactly this and nothing else: no
        model call, no download, no network. Everything expensive happens later,
        from the sweep, where a slow step delays one alert instead of blocking
        ingestion for every watcher.
        """
        has_text = bool(text and text.strip())
        settle_after = datetime.now(UTC) + timedelta(seconds=self._quiet_window)
        async with self._write_lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                await self._conn.execute(
                    """
                    INSERT INTO housing_live_units (
                        unit_key, chat_id, grouped_id, representative_message_id,
                        assembled_text, media_count, state, settle_after
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'assembling', ?)
                    ON CONFLICT(unit_key) DO UPDATE SET
                        -- A later member never overwrites text with emptiness:
                        -- in an album exactly one member carries the caption
                        -- and it is not reliably the first to arrive.
                        assembled_text = COALESCE(
                            NULLIF(excluded.assembled_text, ''),
                            housing_live_units.assembled_text
                        ),
                        -- media_count is recomputed from the members below
                        -- rather than incremented here: recovery replays a
                        -- message whose durable job was interrupted, and an
                        -- increment would count that message's photograph
                        -- twice while the member insert correctly ignores it.
                        media_count = housing_live_units.media_count,
                        -- Only a unit still assembling may have its deadline
                        -- pushed out. A late duplicate of a message whose unit
                        -- already went to extraction must not reopen it.
                        settle_after = CASE
                            WHEN housing_live_units.state = 'assembling'
                            THEN excluded.settle_after
                            ELSE housing_live_units.settle_after
                        END,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        unit_key,
                        chat_id,
                        grouped_id,
                        message_id,
                        text if has_text else None,
                        1 if has_media else 0,
                        settle_after.isoformat(),
                    ),
                )
                cursor = await self._conn.execute(
                    "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM housing_live_unit_messages"
                    " WHERE unit_key = ?",
                    (unit_key,),
                )
                row = await cursor.fetchone()
                ordinal = int(row[0]) if row is not None else 0
                await self._conn.execute(
                    """
                    INSERT INTO housing_live_unit_messages (
                        unit_key, message_id, telegram_msg_id, ordinal,
                        has_media, telegram_photo_id, has_text
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(unit_key, message_id) DO NOTHING
                    """,
                    (
                        unit_key,
                        message_id,
                        telegram_msg_id,
                        ordinal,
                        1 if has_media else 0,
                        telegram_photo_id,
                        1 if has_text else 0,
                    ),
                )
                # Derived from the members, in the same transaction that just
                # changed them, so a replay converges on the same number
                # instead of drifting upward on every retry.
                await self._conn.execute(
                    """
                    UPDATE housing_live_units
                    SET media_count = (
                        SELECT COUNT(*) FROM housing_live_unit_messages
                        WHERE unit_key = ? AND has_media = 1
                    )
                    WHERE unit_key = ?
                    """,
                    (unit_key, unit_key),
                )
                await self._conn.commit()
            except BaseException:
                await self._conn.rollback()
                raise

    # How long a unit may sit in a mid-processing state before a sweep
    # concludes the worker that owned it died. Processing a unit takes
    # seconds; fifteen minutes is not a race with a slow LLM call.
    STALE_UNIT_MINUTES = 15
    # A unit that keeps failing is retried this many times and then left
    # alone, so one poisoned advertisement cannot burn an LLM call per sweep
    # forever.
    MAX_SWEEP_ATTEMPTS = 5

    async def claim_settled_units(self, *, limit: int = 20) -> list[ContentUnit]:
        """Take ownership of work: settled units, plus abandoned ones.

        Claiming moves the row out of ``assembling`` in the same transaction as
        the read, so two sweeps cannot both work the same advertisement.

        A unit stuck in any non-terminal state (``ready``/``extracting``/
        ``extracted``/``matching``/``error``) is work a previous cycle started
        and never finished — the daemon was killed mid-flight, or the cycle
        errored. Re-processing from the top is safe: facts and matches are
        upserts and alerts deduplicate, so the only cost of a retry is the
        repeated model call.
        """
        async with self._write_lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self._conn.execute(
                    """
                    SELECT unit_key FROM housing_live_units
                    WHERE (state = 'assembling'
                           AND datetime(settle_after) <= CURRENT_TIMESTAMP)
                       OR (state IN ('ready', 'extracting', 'extracted',
                                     'matching', 'error')
                           AND datetime(updated_at)
                               <= datetime('now', ?)
                           AND sweep_attempts < ?)
                    ORDER BY settle_after
                    LIMIT ?
                    """,
                    (
                        f"-{self.STALE_UNIT_MINUTES} minutes",
                        self.MAX_SWEEP_ATTEMPTS,
                        limit,
                    ),
                )
                keys = [str(row[0]) for row in await cursor.fetchall()]
                if keys:
                    placeholders = ",".join("?" * len(keys))
                    await self._conn.execute(
                        f"UPDATE housing_live_units SET state = 'ready',"  # noqa: S608 - fixed placeholders
                        f" sweep_attempts = sweep_attempts + 1,"
                        f" updated_at = CURRENT_TIMESTAMP WHERE unit_key IN ({placeholders})",
                        keys,
                    )
                await self._conn.commit()
            except BaseException:
                await self._conn.rollback()
                raise
        return [unit for key in keys if (unit := await self.get_unit(key)) is not None]

    async def get_unit(self, unit_key: str) -> ContentUnit | None:
        """Read one unit with its members."""
        cursor = await self._conn.execute(
            """
            SELECT unit_key, chat_id, grouped_id, unit_version, representative_message_id,
                   assembled_text, media_count, state
            FROM housing_live_units WHERE unit_key = ?
            """,
            (unit_key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        cursor = await self._conn.execute(
            """
            SELECT message_id, telegram_msg_id, ordinal, has_media, telegram_photo_id, has_text
            FROM housing_live_unit_messages WHERE unit_key = ? ORDER BY ordinal
            """,
            (unit_key,),
        )
        members = tuple(
            UnitMember(
                message_id=int(member["message_id"]),
                telegram_msg_id=int(member["telegram_msg_id"]),
                ordinal=int(member["ordinal"]),
                has_media=bool(member["has_media"]),
                telegram_photo_id=(
                    int(member["telegram_photo_id"])
                    if member["telegram_photo_id"] is not None
                    else None
                ),
                has_text=bool(member["has_text"]),
            )
            for member in await cursor.fetchall()
        )
        return ContentUnit(
            unit_key=str(row["unit_key"]),
            chat_id=int(row["chat_id"]),
            grouped_id=int(row["grouped_id"]) if row["grouped_id"] is not None else None,
            unit_version=int(row["unit_version"]),
            representative_message_id=int(row["representative_message_id"]),
            assembled_text=row["assembled_text"],
            media_count=int(row["media_count"]),
            state=UnitState(str(row["state"])),
            members=members,
        )

    async def set_unit_state(
        self,
        unit_key: str,
        state: UnitState,
        *,
        error: str | None = None,
    ) -> None:
        """Move a unit along, recording why if it stopped badly."""
        async with self._write_lock:
            await self._conn.execute(
                "UPDATE housing_live_units SET state = ?, last_error = ?,"
                " updated_at = CURRENT_TIMESTAMP WHERE unit_key = ?",
                (state.value, error, unit_key),
            )
            await self._conn.commit()

    # ------------------------------------------------------------------
    # Facts and verdicts
    # ------------------------------------------------------------------

    async def record_facts(self, unit_key: str, facts: dict[str, Any]) -> None:
        """Store what is known about a unit, including what is not known.

        Every absent value is written as NULL with its source set to
        ``unknown`` rather than being left out, so a later reader can tell
        "the advertisement says nothing about bathrooms" from "nobody has
        looked at the advertisement yet".
        """
        columns = (
            "unit_version",
            "is_rental_offer",
            "is_vehicle_ad",
            "bedrooms",
            "bedrooms_source",
            "bathrooms",
            "bathrooms_source",
            "monthly_price_thb",
            "price_source",
            "tv_present",
            "tv_size_class",
            "tv_source",
            "property_type",
            "property_type_source",
            "terrace",
            "terrace_source",
            "private_setting",
            "nature_setting",
            "amenities_json",
            "area_raw",
            "evidence_quote",
            "vision_status",
            "extractor_version",
        )
        values = [facts.get(column) for column in columns]
        assignments = ", ".join(f"{column} = excluded.{column}" for column in columns)
        async with self._write_lock:
            await self._conn.execute(
                f"""
                INSERT INTO housing_live_facts (unit_key, {", ".join(columns)}, extracted_at)
                VALUES (?, {", ".join("?" * len(columns))}, CURRENT_TIMESTAMP)
                ON CONFLICT(unit_key) DO UPDATE SET
                    {assignments}, extracted_at = CURRENT_TIMESTAMP
                """,  # noqa: S608 - column names are a fixed literal tuple
                [unit_key, *values],
            )
            await self._conn.commit()

    async def get_facts(self, unit_key: str) -> dict[str, Any] | None:
        """Read stored facts for a unit."""
        cursor = await self._conn.execute(
            "SELECT * FROM housing_live_facts WHERE unit_key = ?", (unit_key,)
        )
        row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def record_match(
        self,
        *,
        unit_key: str,
        requirements_revision: int,
        verdict: Verdict,
        field_verdicts: dict[str, Any],
    ) -> None:
        """Store how a unit stood against one revision of the requirements."""
        async with self._write_lock:
            await self._conn.execute(
                """
                INSERT INTO housing_matches (
                    unit_key, requirements_revision, verdict, field_verdicts_json
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(unit_key, requirements_revision) DO UPDATE SET
                    verdict = excluded.verdict,
                    field_verdicts_json = excluded.field_verdicts_json,
                    computed_at = CURRENT_TIMESTAMP
                """,
                (
                    unit_key,
                    requirements_revision,
                    verdict.value,
                    json.dumps(field_verdicts, ensure_ascii=False),
                ),
            )
            await self._conn.commit()

    async def record_match_with_alert(
        self,
        *,
        unit_key: str,
        requirements_revision: int,
        verdict: Verdict,
        field_verdicts: dict[str, Any],
        alert: dict[str, Any] | None,
    ) -> int | None:
        """Store a verdict and its alert in one transaction.

        The verdict and the alert it justifies must land together: committed
        separately, a death between the two writes leaves a recorded verdict
        whose alert never existed — and a retry that compares against the
        recorded verdict concludes nothing changed and never sends it.

        ``alert`` carries enqueue_alert's keyword arguments (minus the unit
        key and revision); None records the verdict alone. Returns the alert
        id, or None when there was no alert to queue or it deduplicated.
        """
        async with self._write_lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                await self._conn.execute(
                    """
                    INSERT INTO housing_matches (
                        unit_key, requirements_revision, verdict, field_verdicts_json
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(unit_key, requirements_revision) DO UPDATE SET
                        verdict = excluded.verdict,
                        field_verdicts_json = excluded.field_verdicts_json,
                        computed_at = CURRENT_TIMESTAMP
                    """,
                    (
                        unit_key,
                        requirements_revision,
                        verdict.value,
                        json.dumps(field_verdicts, ensure_ascii=False),
                    ),
                )
                alert_id: int | None = None
                if alert is not None:
                    photo_paths = alert.get("photo_paths")
                    cursor = await self._conn.execute(
                        """
                        INSERT INTO housing_alerts (
                            unit_key, chat_id, chat_title, telegram_msg_id,
                            requirements_revision, verdict, kind, body_html,
                            photo_paths_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(unit_key, verdict, kind) DO NOTHING
                        RETURNING id
                        """,
                        (
                            unit_key,
                            alert["chat_id"],
                            alert.get("chat_title"),
                            alert["telegram_msg_id"],
                            requirements_revision,
                            Verdict(alert["verdict"]).value,
                            AlertKind(alert["kind"]).value,
                            alert["body_html"],
                            json.dumps(photo_paths, ensure_ascii=False) if photo_paths else None,
                        ),
                    )
                    row = await cursor.fetchone()
                    alert_id = int(row[0]) if row is not None else None
                await self._conn.commit()
            except BaseException:
                await self._conn.rollback()
                raise
        return alert_id

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    async def enqueue_alert(
        self,
        *,
        unit_key: str,
        chat_id: int,
        chat_title: str | None,
        telegram_msg_id: int,
        requirements_revision: int,
        verdict: Verdict,
        kind: AlertKind,
        body_html: str,
        photo_paths: list[str] | None = None,
    ) -> int | None:
        """Queue an alert, or return None when this verdict was already sent.

        Deduplication is on (unit, verdict, kind): re-running the matcher over
        the same facts must not re-notify, while a genuine upgrade from
        ``possible`` to ``confirmed`` is a different verdict and therefore a
        new alert.
        """
        async with self._write_lock:
            cursor = await self._conn.execute(
                """
                INSERT INTO housing_alerts (
                    unit_key, chat_id, chat_title, telegram_msg_id,
                    requirements_revision, verdict, kind, body_html, photo_paths_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(unit_key, verdict, kind) DO NOTHING
                RETURNING id
                """,
                (
                    unit_key,
                    chat_id,
                    chat_title,
                    telegram_msg_id,
                    requirements_revision,
                    verdict.value,
                    kind.value,
                    body_html,
                    json.dumps(photo_paths, ensure_ascii=False) if photo_paths else None,
                ),
            )
            row = await cursor.fetchone()
            await self._conn.commit()
        return int(row[0]) if row is not None else None

    async def claim_due_alerts(
        self,
        *,
        lease_owner: str,
        lease_seconds: int = 120,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Lease the alerts that are due, so two deliveries cannot overlap."""
        async with self._write_lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self._conn.execute(
                    """
                    SELECT id FROM housing_alerts
                    WHERE delivery_status = 'pending'
                      AND datetime(next_attempt_at) <= CURRENT_TIMESTAMP
                      AND (claimed_until IS NULL OR datetime(claimed_until) <= CURRENT_TIMESTAMP)
                    ORDER BY next_attempt_at, id
                    LIMIT ?
                    """,
                    (limit,),
                )
                ids = [int(row[0]) for row in await cursor.fetchall()]
                if ids:
                    placeholders = ",".join("?" * len(ids))
                    # attempts is NOT incremented here: a claim is a lease,
                    # not a delivery attempt. A crash between claiming and
                    # calling Telegram would otherwise burn retry budget on
                    # sends that never happened; settle_alert counts the
                    # attempts that actually completed.
                    await self._conn.execute(
                        f"""
                        UPDATE housing_alerts
                        SET claimed_until = datetime(CURRENT_TIMESTAMP, ?),
                            lease_owner = ?
                        WHERE id IN ({placeholders})
                        """,  # noqa: S608 - placeholders are generated, values are bound
                        [f"+{lease_seconds} seconds", lease_owner, *ids],
                    )
                await self._conn.commit()
            except BaseException:
                await self._conn.rollback()
                raise
            if not ids:
                return []
            placeholders = ",".join("?" * len(ids))
            cursor = await self._conn.execute(
                f"SELECT * FROM housing_alerts WHERE id IN ({placeholders})",  # noqa: S608
                ids,
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def settle_alert(
        self,
        alert_id: int,
        *,
        delivered: bool,
        error: str | None = None,
        retry_in_seconds: int | None = None,
    ) -> None:
        """Record a delivery outcome, scheduling a retry when one is wanted."""
        async with self._write_lock:
            if delivered:
                await self._conn.execute(
                    "UPDATE housing_alerts SET delivery_status = 'delivered',"
                    " attempts = attempts + 1,"
                    " delivered_at = CURRENT_TIMESTAMP, claimed_until = NULL,"
                    " lease_owner = NULL, last_error = NULL WHERE id = ?",
                    (alert_id,),
                )
            elif retry_in_seconds is None:
                await self._conn.execute(
                    "UPDATE housing_alerts SET delivery_status = 'failed',"
                    " attempts = attempts + 1,"
                    " claimed_until = NULL, lease_owner = NULL, last_error = ? WHERE id = ?",
                    (error, alert_id),
                )
            else:
                await self._conn.execute(
                    "UPDATE housing_alerts SET delivery_status = 'pending',"
                    " attempts = attempts + 1,"
                    " next_attempt_at = datetime(CURRENT_TIMESTAMP, ?),"
                    " claimed_until = NULL, lease_owner = NULL, last_error = ? WHERE id = ?",
                    (f"+{retry_in_seconds} seconds", error, alert_id),
                )
            await self._conn.commit()

    # ------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------

    async def active_requirements(self) -> tuple[int, dict[str, Any]] | None:
        """The revision the daemon is currently judging listings against."""
        cursor = await self._conn.execute(
            """
            SELECT r.revision, r.definition_json
            FROM housing_requirements_active a
            JOIN housing_requirements_revisions r ON r.revision = a.active_revision
            WHERE a.id = 1
            """
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return int(row["revision"]), json.loads(str(row["definition_json"]))

    async def requirements_generation(self) -> int:
        """One scalar read that tells the daemon whether the owner edited anything."""
        cursor = await self._conn.execute(
            "SELECT generation FROM housing_requirements_active WHERE id = 1"
        )
        row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def save_requirements(
        self,
        *,
        definition: dict[str, Any],
        created_by: str,
        expected_revision: int | None = None,
    ) -> int:
        """Append a revision and make it active.

        ``expected_revision`` is optimistic concurrency: two callers editing
        from the same starting point must not have one silently overwrite the
        other, because the loser would believe an edit landed that did not.
        """
        payload = json.dumps(definition, ensure_ascii=False, sort_keys=True)
        async with self._write_lock:
            await self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self._conn.execute(
                    "SELECT active_revision FROM housing_requirements_active WHERE id = 1"
                )
                row = await cursor.fetchone()
                current = int(row[0]) if row is not None else 0
                if expected_revision is not None and expected_revision != current:
                    raise ValueError(
                        f"requirements moved on: expected revision {expected_revision}, "
                        f"active revision is {current}"
                    )
                revision = current + 1
                await self._conn.execute(
                    "INSERT INTO housing_requirements_revisions"
                    " (revision, definition_json, created_by) VALUES (?, ?, ?)",
                    (revision, payload, created_by),
                )
                await self._conn.execute(
                    """
                    INSERT INTO housing_requirements_active (id, active_revision, generation)
                    VALUES (1, ?, 1)
                    ON CONFLICT(id) DO UPDATE SET
                        active_revision = excluded.active_revision,
                        generation = housing_requirements_active.generation + 1
                    """,
                    (revision,),
                )
                await self._conn.commit()
            except BaseException:
                await self._conn.rollback()
                raise
        return revision

    async def units_for_rematch(self, *, limit: int = 500) -> list[str]:
        """Units whose stored facts can be judged again after an edit.

        Everything extracted is eligible, with no time window. A requirement
        the owner loosens has to be able to surface a listing from any point in
        the retained history: re-judging is a pure function over facts that are
        already paid for, so there is nothing to save by looking at less.
        """
        cursor = await self._conn.execute(
            """
            SELECT unit_key FROM housing_live_facts
            WHERE is_rental_offer = 1
            ORDER BY extracted_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [str(row[0]) for row in await cursor.fetchall()]

    # ------------------------------------------------------------------
    # Photographs
    # ------------------------------------------------------------------

    async def enqueue_media(
        self,
        *,
        unit_key: str,
        chat_id: int,
        photos: list[tuple[int, int]],
        priority: str = "live",
    ) -> int:
        """Ask for a unit's photographs, skipping any already on disk.

        ``photos`` is (telegram_msg_id, telegram_photo_id) pairs. A photograph
        already downloaded under another unit — the same advertisement
        crossposted into a second chat — is recorded as done immediately with
        the existing file, so the same bytes are never fetched twice.
        """
        queued = 0
        async with self._write_lock:
            for telegram_msg_id, photo_id in photos:
                cursor = await self._conn.execute(
                    "SELECT local_path, byte_size FROM housing_media"
                    " WHERE telegram_photo_id = ? AND download_status = 'downloaded'"
                    " LIMIT 1",
                    (photo_id,),
                )
                existing = await cursor.fetchone()
                if existing is not None:
                    await self._conn.execute(
                        """
                        INSERT INTO housing_media (
                            unit_key, chat_id, telegram_msg_id, telegram_photo_id,
                            priority, download_status, local_path, byte_size, downloaded_at
                        )
                        VALUES (?, ?, ?, ?, ?, 'downloaded', ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(unit_key, telegram_msg_id) DO NOTHING
                        """,
                        (
                            unit_key,
                            chat_id,
                            telegram_msg_id,
                            photo_id,
                            priority,
                            existing["local_path"],
                            existing["byte_size"],
                        ),
                    )
                    continue
                cursor = await self._conn.execute(
                    """
                    INSERT INTO housing_media (
                        unit_key, chat_id, telegram_msg_id, telegram_photo_id, priority
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(unit_key, telegram_msg_id) DO NOTHING
                    RETURNING unit_key
                    """,
                    (unit_key, chat_id, telegram_msg_id, photo_id, priority),
                )
                if await cursor.fetchone() is not None:
                    queued += 1
            await self._conn.commit()
        return queued

    async def next_media_download(self, *, priority: str) -> dict[str, Any] | None:
        """The next photograph due for fetching at this priority."""
        cursor = await self._conn.execute(
            """
            SELECT * FROM housing_media
            WHERE download_status = 'pending'
              AND priority = ?
              AND (not_before IS NULL OR datetime(not_before) <= CURRENT_TIMESTAMP)
            ORDER BY requested_at
            LIMIT 1
            """,
            (priority,),
        )
        row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def settle_media(
        self,
        *,
        unit_key: str,
        telegram_msg_id: int,
        status: str,
        local_path: str | None = None,
        byte_size: int | None = None,
        error: str | None = None,
        retry_in_seconds: int | None = None,
    ) -> None:
        """Record what happened to one download attempt."""
        async with self._write_lock:
            await self._conn.execute(
                """
                UPDATE housing_media
                SET download_status = ?,
                    local_path = COALESCE(?, local_path),
                    byte_size = COALESCE(?, byte_size),
                    attempts = attempts + 1,
                    last_error = ?,
                    not_before = CASE
                        WHEN ? IS NULL THEN NULL
                        ELSE datetime(CURRENT_TIMESTAMP, ?)
                    END,
                    downloaded_at = CASE WHEN ? = 'downloaded' THEN CURRENT_TIMESTAMP END
                WHERE unit_key = ? AND telegram_msg_id = ?
                """,
                (
                    status,
                    local_path,
                    byte_size,
                    error,
                    retry_in_seconds,
                    f"+{retry_in_seconds} seconds" if retry_in_seconds else None,
                    status,
                    unit_key,
                    telegram_msg_id,
                ),
            )
            await self._conn.commit()

    async def downloaded_media(self, unit_key: str) -> list[str]:
        """Local paths of this unit's photographs, in message order."""
        cursor = await self._conn.execute(
            "SELECT local_path FROM housing_media"
            " WHERE unit_key = ? AND download_status = 'downloaded' AND local_path IS NOT NULL"
            " ORDER BY telegram_msg_id",
            (unit_key,),
        )
        return [str(row[0]) for row in await cursor.fetchall()]

    async def units_awaiting_vision(self, *, limit: int = 10) -> list[str]:
        """Units whose photographs are in and whose unknowns vision could answer.

        Only units already judged worth alerting on are eligible: reading the
        photographs of an advertisement that failed on its stated price would
        spend the budget on a listing nobody will be shown.
        """
        cursor = await self._conn.execute(
            """
            SELECT f.unit_key
            FROM housing_live_facts f
            JOIN housing_matches m ON m.unit_key = f.unit_key
            WHERE f.is_rental_offer = 1
              AND f.vision_status IN ('not_attempted', 'pending')
              AND m.verdict IN ('possible', 'confirmed')
              AND EXISTS (
                  SELECT 1 FROM housing_media
                  WHERE housing_media.unit_key = f.unit_key
                    AND download_status = 'downloaded'
              )
              AND NOT EXISTS (
                  SELECT 1 FROM housing_media
                  WHERE housing_media.unit_key = f.unit_key
                    AND download_status = 'pending'
              )
            ORDER BY f.extracted_at
            LIMIT ?
            """,
            (limit,),
        )
        return [str(row[0]) for row in await cursor.fetchall()]

    async def latest_match(self, unit_key: str) -> dict[str, Any] | None:
        """The most recent verdict recorded for a unit."""
        cursor = await self._conn.execute(
            "SELECT * FROM housing_matches WHERE unit_key = ?"
            " ORDER BY requirements_revision DESC LIMIT 1",
            (unit_key,),
        )
        row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def chat_kind(self, chat_id: int) -> str:
        """Whether this chat is a rentals board or the island's talk chat."""
        cursor = await self._conn.execute(
            "SELECT kind FROM housing_chat_kinds WHERE chat_id = ?", (chat_id,)
        )
        row = await cursor.fetchone()
        return str(row[0]) if row is not None else "general_island"

    async def set_chat_kind(self, chat_id: int, kind: str) -> None:
        """Record what a chat is, so the gate knows how much to trust it."""
        async with self._write_lock:
            await self._conn.execute(
                """
                INSERT INTO housing_chat_kinds (chat_id, kind) VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    kind = excluded.kind, updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, kind),
            )
            await self._conn.commit()
