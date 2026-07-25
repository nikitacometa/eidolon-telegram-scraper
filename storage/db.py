"""SQLite database wrapper with async access and auto-migration."""

import asyncio
import json
import logging
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import aiosqlite

from config.watchers import Watcher
from pipeline.models import (
    AlertDeliveryStatus,
    AlertDraft,
    AlertOutboxItem,
    DeliveryResult,
    ObservationMode,
    ObservationSource,
    ObservedChat,
    PipelineOutcome,
    StageStatus,
    StoredPipelineJob,
)

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class DailyMessage(TypedDict):
    """Message shape consumed by the daily summarizer."""

    message_id: int
    chat_id: int
    chat_title: str
    sender_name: str
    text: str
    date: str


def _normalize_delivery_error_code(error_code: str | None) -> str:
    """Return a bounded non-PII code suitable for durable storage."""
    if error_code is not None and _SAFE_ERROR_CODE.fullmatch(error_code):
        return error_code
    return "delivery_error"


class Database:
    """Async SQLite wrapper for Eidolon message and alert storage."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        # One aiosqlite connection cannot safely host overlapping transactions
        # or interleaved reads inside another task's transaction.
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open the database connection and run migrations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.db_path.parent, 0o700)
        self._conn = await aiosqlite.connect(self.db_path)
        os.chmod(self.db_path, 0o600)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._migrate()
        logger.info("Database connected: %s", self.db_path)

    async def _migrate(self) -> None:
        """Run schema.sql to create tables if they don't exist, then apply incremental migrations."""
        schema = SCHEMA_PATH.read_text()
        await self.conn.executescript(schema)
        await self.conn.commit()
        await self._apply_migrations()

    async def _apply_migrations(self) -> None:
        """Apply incremental schema changes for existing databases."""
        # Existing production databases predate the explicit migration fields.
        cursor = await self.conn.execute("PRAGMA table_info(messages)")
        message_columns = {row[1] for row in await cursor.fetchall()}
        if "chat_title" not in message_columns:
            await self.conn.execute("ALTER TABLE messages ADD COLUMN chat_title TEXT")
            logger.info("Migration: added chat_title column to messages")

        cursor = await self.conn.execute("PRAGMA table_info(alerts)")
        alert_columns = {row[1] for row in await cursor.fetchall()}
        alert_migrations = {
            "delivery_status": (
                "ALTER TABLE alerts ADD COLUMN delivery_status TEXT NOT NULL DEFAULT 'pending'"
            ),
            "delivery_attempts": (
                "ALTER TABLE alerts ADD COLUMN delivery_attempts INTEGER NOT NULL DEFAULT 0"
            ),
            "next_attempt_at": "ALTER TABLE alerts ADD COLUMN next_attempt_at TIMESTAMP",
            "last_error": "ALTER TABLE alerts ADD COLUMN last_error TEXT",
            "claimed_until": "ALTER TABLE alerts ADD COLUMN claimed_until TIMESTAMP",
            "lease_owner": "ALTER TABLE alerts ADD COLUMN lease_owner TEXT",
            "matched_keyword": "ALTER TABLE alerts ADD COLUMN matched_keyword TEXT",
        }
        for column, statement in alert_migrations.items():
            if column not in alert_columns:
                await self.conn.execute(statement)
                logger.info("Migration: added alerts.%s", column)

        # Preserve successful legacy deliveries; all other legacy rows become
        # immediately recoverable pending outbox work.
        await self.conn.execute(
            """
            UPDATE alerts
            SET delivery_status = 'sent',
                next_attempt_at = NULL,
                claimed_until = NULL,
                lease_owner = NULL
            WHERE sent_at IS NOT NULL
            """
        )
        # Legacy schemas allowed nullable/orphaned foreign keys. Such rows
        # cannot be delivered because there is no immutable source payload.
        await self.conn.execute(
            """
            UPDATE alerts
            SET delivery_status = 'failed',
                next_attempt_at = NULL,
                claimed_until = NULL,
                lease_owner = NULL,
                last_error = 'orphaned_message'
            WHERE message_id IS NULL
               OR NOT EXISTS (
                   SELECT 1 FROM messages WHERE messages.id = alerts.message_id
               )
            """
        )
        await self.conn.execute(
            """
            UPDATE alerts
            SET next_attempt_at = COALESCE(next_attempt_at, created_at, CURRENT_TIMESTAMP)
            WHERE delivery_status = 'pending'
            """
        )

        # Old schemas did not enforce the idempotency key. Keep the newest
        # successfully sent row (or newest row) before creating the unique index.
        await self.conn.execute(
            """
            DELETE FROM alerts
            WHERE id IN (
                SELECT id
                FROM (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY watcher_name, message_id
                            ORDER BY
                                CASE WHEN sent_at IS NOT NULL THEN 0 ELSE 1 END,
                                id DESC
                        ) AS duplicate_rank
                    FROM alerts
                    WHERE message_id IS NOT NULL
                )
                WHERE duplicate_rank > 1
            )
            """
        )
        await self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_alerts_watcher_message
            ON alerts(watcher_name, message_id)
            """
        )
        await self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_alerts_due
            ON alerts(delivery_status, next_attempt_at, claimed_until, id)
            """
        )

        cursor = await self.conn.execute("PRAGMA table_info(pipeline_runs)")
        pipeline_columns = {row[1] for row in await cursor.fetchall()}
        provenance_migrations = {
            "embedding_negative_score": (
                "ALTER TABLE pipeline_runs ADD COLUMN embedding_negative_score REAL"
            ),
            "embedding_threshold": (
                "ALTER TABLE pipeline_runs ADD COLUMN embedding_threshold REAL"
            ),
            "embedding_model": "ALTER TABLE pipeline_runs ADD COLUMN embedding_model TEXT",
            "embedding_latency_ms": (
                "ALTER TABLE pipeline_runs ADD COLUMN embedding_latency_ms REAL"
            ),
            "embedding_input_tokens": (
                "ALTER TABLE pipeline_runs ADD COLUMN "
                "embedding_input_tokens INTEGER NOT NULL DEFAULT 0"
            ),
            "llm_model": "ALTER TABLE pipeline_runs ADD COLUMN llm_model TEXT",
            "llm_prompt_version": ("ALTER TABLE pipeline_runs ADD COLUMN llm_prompt_version TEXT"),
            "llm_latency_ms": "ALTER TABLE pipeline_runs ADD COLUMN llm_latency_ms REAL",
            "llm_input_tokens": (
                "ALTER TABLE pipeline_runs ADD COLUMN llm_input_tokens INTEGER NOT NULL DEFAULT 0"
            ),
            "llm_output_tokens": (
                "ALTER TABLE pipeline_runs ADD COLUMN llm_output_tokens INTEGER NOT NULL DEFAULT 0"
            ),
            "llm_passed": (
                "ALTER TABLE pipeline_runs ADD COLUMN llm_passed INTEGER NOT NULL DEFAULT 0"
            ),
        }
        for column, statement in provenance_migrations.items():
            if column not in pipeline_columns:
                await self.conn.execute(statement)
                logger.info("Migration: added pipeline_runs.%s", column)
        if "llm_passed" not in pipeline_columns:
            await self.conn.execute("UPDATE pipeline_runs SET llm_passed = llm_relevant")

        lifecycle_migrations = {
            "watcher_config_fingerprint": (
                "ALTER TABLE pipeline_runs ADD COLUMN watcher_config_fingerprint "
                "TEXT NOT NULL DEFAULT 'legacy-unknown'"
            ),
            "processing_status": (
                "ALTER TABLE pipeline_runs ADD COLUMN processing_status "
                "TEXT NOT NULL DEFAULT 'completed'"
            ),
            "accepted": (
                "ALTER TABLE pipeline_runs ADD COLUMN accepted INTEGER NOT NULL DEFAULT 0"
            ),
            "completed_at": "ALTER TABLE pipeline_runs ADD COLUMN completed_at TIMESTAMP",
        }
        added_processing_status = "processing_status" not in pipeline_columns
        added_accepted = "accepted" not in pipeline_columns
        for column, statement in lifecycle_migrations.items():
            if column not in pipeline_columns:
                await self.conn.execute(statement)
                logger.info("Migration: added pipeline_runs.%s", column)
        if added_processing_status:
            await self.conn.execute(
                """
                UPDATE pipeline_runs
                SET processing_status = 'completed',
                    completed_at = COALESCE(completed_at, created_at)
                """
            )
        if added_accepted:
            await self.conn.execute(
                """
                UPDATE pipeline_runs
                SET accepted = CASE
                    WHEN llm_status = 'ok' THEN llm_passed
                    WHEN embedding_status = 'ok' THEN embedding_passed
                    ELSE rule_passed
                END
                """
            )
        await self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pipeline_runs_pending
            ON pipeline_runs(processing_status, created_at, id)
            """
        )

        cursor = await self.conn.execute("PRAGMA table_info(filter_stats)")
        filter_stats_columns = {row[1] for row in await cursor.fetchall()}
        if "accepted" not in filter_stats_columns:
            await self.conn.execute(
                "ALTER TABLE filter_stats ADD COLUMN accepted INTEGER NOT NULL DEFAULT 0"
            )
            await self.conn.execute(
                """
                UPDATE filter_stats
                SET accepted = COALESCE(
                    (
                        SELECT SUM(p.accepted)
                        FROM pipeline_runs AS p
                        WHERE p.watcher_name = filter_stats.watcher_name
                          AND DATE(p.created_at) = filter_stats.date
                    ),
                    0
                )
                """
            )
            logger.info("Migration: added filter_stats.accepted")

        await self.conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database closed")

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    async def store_message(
        self,
        *,
        telegram_msg_id: int,
        chat_id: int,
        chat_title: str | None = None,
        chat_type: str | None = None,
        sender_id: int | None,
        sender_name: str | None,
        text: str | None,
        date: str,
        raw_json: str | None = None,
        watcher_names: Sequence[str] = (),
        watcher_fingerprints: Mapping[str, str] | None = None,
    ) -> int | None:
        """Atomically store a message and its expected watcher jobs.

        The pending rows close the crash window between Telegram ingestion and
        pipeline execution. They can be reconstructed from SQLite on restart.
        Returns ``None`` when Telegram replays an existing message.
        """
        async with self._write_lock:
            try:
                cursor = await self.conn.execute(
                    """
                    INSERT INTO messages (
                        telegram_msg_id, chat_id, chat_title, sender_id,
                        sender_name, text, date, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id, telegram_msg_id) DO NOTHING
                    RETURNING id
                    """,
                    (
                        telegram_msg_id,
                        chat_id,
                        chat_title,
                        sender_id,
                        sender_name,
                        text,
                        date,
                        raw_json,
                    ),
                )
                inserted = await cursor.fetchone()
                if inserted is None:
                    await self.conn.rollback()
                    logger.debug(
                        "Duplicate message %d in chat %d",
                        telegram_msg_id,
                        chat_id,
                    )
                    return None
                row_id = int(inserted[0])
                unique_watchers = tuple(dict.fromkeys(watcher_names))
                if unique_watchers:
                    if watcher_fingerprints is None or any(
                        watcher_name not in watcher_fingerprints for watcher_name in unique_watchers
                    ):
                        raise ValueError("every durable watcher job requires a policy fingerprint")
                    await self.conn.executemany(
                        """
                        INSERT INTO pipeline_runs (
                            message_id, watcher_name,
                            watcher_config_fingerprint, processing_status
                        )
                        VALUES (?, ?, ?, 'pending')
                        ON CONFLICT(message_id, watcher_name) DO NOTHING
                        """,
                        [
                            (
                                row_id,
                                watcher_name,
                                watcher_fingerprints[watcher_name],
                            )
                            for watcher_name in unique_watchers
                        ],
                    )
                await self.conn.execute(
                    """
                    INSERT INTO chats (
                        chat_id, title, type, last_message_at, message_count
                    )
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, 1)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        title = COALESCE(excluded.title, chats.title),
                        type = COALESCE(excluded.type, chats.type),
                        last_message_at = CURRENT_TIMESTAMP,
                        message_count = chats.message_count + 1
                    """,
                    (chat_id, chat_title, chat_type),
                )
                await self.conn.commit()
                return row_id
            except BaseException:
                await self.conn.rollback()
                raise

    async def store_alert(
        self,
        *,
        watcher_name: str,
        message_id: int,
        filter_level: int,
        score: float | None = None,
        llm_response: str | None = None,
        matched_keyword: str | None = None,
    ) -> int:
        """Idempotently enqueue an alert and return its durable outbox row ID.

        Re-evaluating the same watcher/message may refresh an unclaimed pending
        payload. Claimed and terminal rows remain immutable audit records.
        """
        async with self._write_lock:
            try:
                alert_id = await self._enqueue_alert_locked(
                    watcher_name=watcher_name,
                    message_id=message_id,
                    alert=AlertDraft(
                        filter_level=filter_level,
                        score=score,
                        llm_response=llm_response,
                        matched_keyword=matched_keyword,
                    ),
                )
                await self.conn.commit()
                return alert_id
            except BaseException:
                await self.conn.rollback()
                raise

    async def _enqueue_alert_locked(
        self,
        *,
        watcher_name: str,
        message_id: int,
        alert: AlertDraft,
    ) -> int:
        """Upsert an alert payload inside the caller's transaction."""
        cursor = await self.conn.execute(
            """
            INSERT INTO alerts (
                watcher_name, message_id, filter_level, score,
                llm_response, matched_keyword
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(watcher_name, message_id) DO UPDATE SET
                filter_level = excluded.filter_level,
                score = excluded.score,
                llm_response = excluded.llm_response,
                matched_keyword = excluded.matched_keyword
            WHERE alerts.delivery_status = 'pending'
              AND alerts.claimed_until IS NULL
            RETURNING id
            """,
            (
                watcher_name,
                message_id,
                alert.filter_level,
                alert.score,
                alert.llm_response,
                alert.matched_keyword,
            ),
        )
        row = await cursor.fetchone()
        if row is None:
            cursor = await self.conn.execute(
                """
                SELECT id
                FROM alerts
                WHERE watcher_name = ? AND message_id = ?
                """,
                (watcher_name, message_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("SQLite did not return an alert row ID")
        return int(row[0])

    async def mark_alert_sent(self, alert_id: int, *, claim_token: str) -> None:
        """Compatibility wrapper for the existing immediate-delivery path."""
        await self.mark_alert_delivery_result(
            alert_id,
            DeliveryResult.success(),
            claim_token=claim_token,
        )

    async def claim_due_alerts(
        self,
        *,
        limit: int = 50,
        lease_seconds: int = 60,
    ) -> list[AlertOutboxItem]:
        """Atomically claim due pending alerts for one bounded worker batch.

        The lease makes an interrupted claim recoverable after restart. SQLite's
        ``BEGIN IMMEDIATE`` serializes claimers, so no two workers on this
        database can receive the same live lease.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be at least 1")

        async with self._write_lock:
            return await self._claim_due_alerts_locked(
                limit=limit,
                lease_seconds=lease_seconds,
            )

    async def _claim_due_alerts_locked(
        self,
        *,
        limit: int,
        lease_seconds: int,
    ) -> list[AlertOutboxItem]:
        """Claim one batch while the process-local outbox lock is held."""
        lease_modifier = f"+{lease_seconds} seconds"
        claim_token = secrets.token_urlsafe(24)
        await self.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self.conn.execute(
                """
                WITH due AS (
                    SELECT a.id
                    FROM alerts AS a
                    JOIN messages AS m ON m.id = a.message_id
                    WHERE a.delivery_status = 'pending'
                      AND (
                          a.next_attempt_at IS NULL
                          OR datetime(a.next_attempt_at) <= CURRENT_TIMESTAMP
                      )
                      AND (
                          a.claimed_until IS NULL
                          OR datetime(a.claimed_until) <= CURRENT_TIMESTAMP
                      )
                    ORDER BY datetime(a.created_at), a.id
                    LIMIT ?
                )
                UPDATE alerts
                SET claimed_until = datetime('now', ?),
                    lease_owner = ?
                WHERE id IN (SELECT id FROM due)
                RETURNING id
                """,
                (limit, lease_modifier, claim_token),
            )
            claimed_ids = [int(row[0]) for row in await cursor.fetchall()]
            if not claimed_ids:
                await self.conn.commit()
                return []

            claimed_ids_json = json.dumps(claimed_ids, separators=(",", ":"))
            cursor = await self.conn.execute(
                """
                SELECT
                    a.id,
                    a.watcher_name,
                    a.message_id,
                    a.filter_level,
                    a.delivery_attempts,
                    a.lease_owner,
                    m.chat_title,
                    m.sender_name,
                    m.text,
                    a.matched_keyword
                FROM alerts AS a
                JOIN messages AS m ON m.id = a.message_id
                WHERE a.id IN (
                    SELECT CAST(value AS INTEGER)
                    FROM json_each(?)
                )
                ORDER BY datetime(a.created_at), a.id
                """,
                (claimed_ids_json,),
            )
            rows = await cursor.fetchall()
            await self.conn.commit()
        except BaseException:
            await self.conn.rollback()
            raise

        return [
            AlertOutboxItem(
                alert_id=int(row["id"]),
                watcher_name=str(row["watcher_name"]),
                message_id=int(row["message_id"]),
                chat_title=str(row["chat_title"] or "Unknown"),
                sender_name=str(row["sender_name"] or "Unknown"),
                text=str(row["text"] or ""),
                matched_keyword=(
                    str(row["matched_keyword"]) if row["matched_keyword"] is not None else None
                ),
                filter_level=int(row["filter_level"]),
                delivery_attempts=int(row["delivery_attempts"]),
                claim_token=str(row["lease_owner"]),
            )
            for row in rows
        ]

    async def mark_alert_delivery_result(
        self,
        alert_id: int,
        result: DeliveryResult,
        *,
        claim_token: str,
        max_attempts: int = 5,
        base_delay_seconds: int = 5,
    ) -> AlertDeliveryStatus:
        """Persist one outbox delivery cycle and return the resulting state."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if base_delay_seconds < 1:
            raise ValueError("base_delay_seconds must be at least 1")

        async with self._write_lock:
            return await self._mark_alert_delivery_result_locked(
                alert_id,
                result,
                claim_token=claim_token,
                max_attempts=max_attempts,
                base_delay_seconds=base_delay_seconds,
            )

    async def _mark_alert_delivery_result_locked(
        self,
        alert_id: int,
        result: DeliveryResult,
        *,
        claim_token: str,
        max_attempts: int,
        base_delay_seconds: int,
    ) -> AlertDeliveryStatus:
        """Update one delivery while the process-local outbox lock is held."""
        await self.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self.conn.execute(
                """
                SELECT
                    delivery_status,
                    delivery_attempts,
                    message_id,
                    watcher_name,
                    lease_owner
                FROM alerts
                WHERE id = ?
                """,
                (alert_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise KeyError(f"alert not found: {alert_id}")

            current_status = AlertDeliveryStatus(str(row["delivery_status"]))
            if current_status is AlertDeliveryStatus.SENT:
                await self.conn.commit()
                return current_status
            if current_status is AlertDeliveryStatus.FAILED and not result.sent:
                await self.conn.commit()
                return current_status
            stored_claim_token = row["lease_owner"]
            if not isinstance(stored_claim_token, str) or not secrets.compare_digest(
                stored_claim_token,
                claim_token,
            ):
                raise RuntimeError("alert delivery claim is no longer owned")

            attempts = int(row["delivery_attempts"]) + 1
            if result.sent:
                status = AlertDeliveryStatus.SENT
                await self.conn.execute(
                    """
                    UPDATE alerts
                    SET delivery_status = ?,
                        delivery_attempts = ?,
                        next_attempt_at = NULL,
                        last_error = NULL,
                        claimed_until = NULL,
                        lease_owner = NULL,
                        sent_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (status.value, attempts, alert_id),
                )
                await self._mark_pipeline_alert_sent_locked(
                    message_id=int(row["message_id"]),
                    watcher_name=str(row["watcher_name"]),
                )
            else:
                error_code = _normalize_delivery_error_code(result.error_code)
                if result.retryable and attempts < max_attempts:
                    status = AlertDeliveryStatus.PENDING
                    delay = result.retry_after or min(
                        base_delay_seconds * (2 ** (attempts - 1)),
                        3600,
                    )
                    await self.conn.execute(
                        """
                        UPDATE alerts
                        SET delivery_status = ?,
                            delivery_attempts = ?,
                            next_attempt_at = datetime('now', ?),
                            last_error = ?,
                            claimed_until = NULL,
                            lease_owner = NULL
                        WHERE id = ?
                        """,
                        (
                            status.value,
                            attempts,
                            f"+{min(delay, 3600)} seconds",
                            error_code,
                            alert_id,
                        ),
                    )
                else:
                    status = AlertDeliveryStatus.FAILED
                    await self.conn.execute(
                        """
                        UPDATE alerts
                        SET delivery_status = ?,
                            delivery_attempts = ?,
                            next_attempt_at = NULL,
                            last_error = ?,
                            claimed_until = NULL,
                            lease_owner = NULL
                        WHERE id = ?
                        """,
                        (status.value, attempts, error_code, alert_id),
                    )

            await self.conn.commit()
            return status
        except BaseException:
            await self.conn.rollback()
            raise

    async def update_chat(
        self,
        *,
        chat_id: int,
        title: str | None = None,
        chat_type: str | None = None,
    ) -> None:
        """Upsert chat metadata and increment message count."""
        async with self._write_lock:
            try:
                await self.conn.execute(
                    """
                    INSERT INTO chats (chat_id, title, type, last_message_at, message_count)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, 1)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        title = COALESCE(excluded.title, chats.title),
                        type = COALESCE(excluded.type, chats.type),
                        last_message_at = CURRENT_TIMESTAMP,
                        message_count = chats.message_count + 1
                    """,
                    (chat_id, title, chat_type),
                )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise

    async def sync_config_bindings(self, watchers: Sequence[Watcher]) -> None:
        """Reconcile config-sourced observation with the policy file.

        The YAML file stays authoritative for what it declares: a chat dropped
        from a policy loses its config binding here too. Bindings created by
        reconnaissance are left alone, so a promoted chat is not silently
        un-promoted by the next deploy.
        """
        declared = {(chat_id, watcher.name) for watcher in watchers for chat_id in watcher.chats}
        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                for chat_id, watcher_name in sorted(declared):
                    await self.conn.execute(
                        """
                        INSERT INTO observed_chats (chat_id, mode, source)
                        VALUES (?, 'monitor', 'config')
                        ON CONFLICT(chat_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                        """,
                        (chat_id,),
                    )
                    await self.conn.execute(
                        """
                        INSERT INTO chat_policy_bindings (chat_id, watcher_name, source)
                        VALUES (?, ?, 'config')
                        ON CONFLICT(chat_id, watcher_name) DO NOTHING
                        """,
                        (chat_id, watcher_name),
                    )

                declared_json = json.dumps(
                    [f"{chat_id}:{name}" for chat_id, name in sorted(declared)],
                    separators=(",", ":"),
                )
                await self.conn.execute(
                    """
                    DELETE FROM chat_policy_bindings
                    WHERE source = 'config'
                      AND (chat_id || ':' || watcher_name) NOT IN (
                          SELECT value FROM json_each(?)
                      )
                    """,
                    (declared_json,),
                )
                # A config chat that lost every policy is no longer observed.
                # Reconnaissance-owned rows are kept: they are not the file's.
                await self.conn.execute(
                    """
                    DELETE FROM observed_chats
                    WHERE source = 'config'
                      AND chat_id NOT IN (SELECT chat_id FROM chat_policy_bindings)
                    """
                )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise

    async def observation_snapshot(self) -> dict[int, ObservedChat]:
        """Return the full chat registry keyed by chat id."""
        async with self.conn.execute(
            """
            SELECT o.chat_id, o.mode, o.source, o.title, o.job_id,
                   GROUP_CONCAT(b.watcher_name) AS watcher_names
            FROM observed_chats AS o
            LEFT JOIN chat_policy_bindings AS b ON b.chat_id = o.chat_id
            GROUP BY o.chat_id
            """
        ) as cursor:
            rows = await cursor.fetchall()

        snapshot: dict[int, ObservedChat] = {}
        for row in rows:
            names = str(row["watcher_names"]) if row["watcher_names"] is not None else ""
            snapshot[int(row["chat_id"])] = ObservedChat(
                chat_id=int(row["chat_id"]),
                mode=ObservationMode(str(row["mode"])),
                source=ObservationSource(str(row["source"])),
                watcher_names=tuple(sorted(name for name in names.split(",") if name)),
                title=str(row["title"]) if row["title"] is not None else None,
                job_id=str(row["job_id"]) if row["job_id"] is not None else None,
            )
        return snapshot

    async def observe_chat(
        self,
        *,
        chat_id: int,
        mode: ObservationMode,
        source: ObservationSource,
        title: str | None = None,
        job_id: str | None = None,
    ) -> None:
        """Add or update one chat in the registry."""
        async with self._write_lock:
            await self.conn.execute(
                """
                INSERT INTO observed_chats (chat_id, mode, title, source, job_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    mode = excluded.mode,
                    title = COALESCE(excluded.title, observed_chats.title),
                    job_id = COALESCE(excluded.job_id, observed_chats.job_id),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, mode.value, title, source.value, job_id),
            )
            await self.conn.commit()

    async def bind_policy(
        self,
        *,
        chat_id: int,
        watcher_name: str,
        source: ObservationSource = ObservationSource.RECON,
        job_id: str | None = None,
    ) -> bool:
        """Attach a policy to an observed chat.

        Returns ``False`` when the chat is not observed: a binding without a
        registry entry would be invisible to ingestion.
        """
        async with self._write_lock:
            async with self.conn.execute(
                "SELECT 1 FROM observed_chats WHERE chat_id = ?",
                (chat_id,),
            ) as cursor:
                if await cursor.fetchone() is None:
                    return False
            await self.conn.execute(
                """
                INSERT INTO chat_policy_bindings (chat_id, watcher_name, source, job_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id, watcher_name) DO NOTHING
                """,
                (chat_id, watcher_name, source.value, job_id),
            )
            await self.conn.commit()
        return True

    async def record_pipeline_outcome(
        self,
        outcome: PipelineOutcome,
        *,
        alert: AlertDraft | None = None,
    ) -> bool:
        """Atomically finish one pipeline job, optional alert, and its stats.

        A pre-enqueued ``pending`` job is completed in place. Direct callers
        may still insert an already-completed outcome. Returns ``False`` for a
        terminal duplicate; aggregate counters change exactly once.
        """
        if alert is not None and not outcome.alert_created:
            raise ValueError("an alert draft requires outcome.alert_created")
        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self.conn.execute(
                    """
                    INSERT INTO pipeline_runs (
                        message_id, watcher_name, processing_status, rule_passed,
                        embedding_status, embedding_passed, embedding_score,
                        embedding_negative_score, embedding_threshold,
                        embedding_model, embedding_latency_ms, embedding_input_tokens,
                        llm_status, llm_relevant, llm_passed, llm_verdict,
                        llm_confidence, llm_model, llm_prompt_version, llm_latency_ms,
                        llm_input_tokens, llm_output_tokens,
                        accepted, alert_created, alert_sent, error_code,
                        completed_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        CURRENT_TIMESTAMP
                    )
                    ON CONFLICT(message_id, watcher_name) DO UPDATE SET
                        processing_status = excluded.processing_status,
                        rule_passed = excluded.rule_passed,
                        embedding_status = excluded.embedding_status,
                        embedding_passed = excluded.embedding_passed,
                        embedding_score = excluded.embedding_score,
                        embedding_negative_score = excluded.embedding_negative_score,
                        embedding_threshold = excluded.embedding_threshold,
                        embedding_model = excluded.embedding_model,
                        embedding_latency_ms = excluded.embedding_latency_ms,
                        embedding_input_tokens = excluded.embedding_input_tokens,
                        llm_status = excluded.llm_status,
                        llm_relevant = excluded.llm_relevant,
                        llm_passed = excluded.llm_passed,
                        llm_verdict = excluded.llm_verdict,
                        llm_confidence = excluded.llm_confidence,
                        llm_model = excluded.llm_model,
                        llm_prompt_version = excluded.llm_prompt_version,
                        llm_latency_ms = excluded.llm_latency_ms,
                        llm_input_tokens = excluded.llm_input_tokens,
                        llm_output_tokens = excluded.llm_output_tokens,
                        accepted = excluded.accepted,
                        alert_created = excluded.alert_created,
                        alert_sent = excluded.alert_sent,
                        error_code = excluded.error_code,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE pipeline_runs.processing_status = 'pending'
                    RETURNING DATE(created_at)
                    """,
                    (
                        outcome.message_id,
                        outcome.watcher_name,
                        outcome.processing_status.value,
                        outcome.rule_passed,
                        outcome.embedding_status.value,
                        outcome.embedding_passed,
                        outcome.embedding_score,
                        outcome.embedding_negative_score,
                        outcome.embedding_threshold,
                        outcome.embedding_model,
                        outcome.embedding_latency_ms,
                        outcome.embedding_input_tokens,
                        outcome.llm_status.value,
                        outcome.llm_relevant,
                        outcome.llm_passed,
                        outcome.llm_verdict,
                        outcome.llm_confidence,
                        outcome.llm_model,
                        outcome.llm_prompt_version,
                        outcome.llm_latency_ms,
                        outcome.llm_input_tokens,
                        outcome.llm_output_tokens,
                        outcome.accepted,
                        outcome.alert_created,
                        outcome.alert_sent,
                        outcome.error_code,
                    ),
                )
                terminal_row = await cursor.fetchone()
                if terminal_row is None:
                    await self.conn.commit()
                    return False

                if alert is not None:
                    await self._enqueue_alert_locked(
                        watcher_name=outcome.watcher_name,
                        message_id=outcome.message_id,
                        alert=alert,
                    )

                outcome_date = str(terminal_row[0])
                await self.conn.execute(
                    """
                    INSERT INTO filter_stats (
                        watcher_name, date, messages_total,
                        passed_level1, passed_level2, passed_level3,
                        accepted, alerts_sent
                    )
                    VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                    ON CONFLICT(watcher_name, date) DO UPDATE SET
                        messages_total = filter_stats.messages_total + 1,
                        passed_level1 = filter_stats.passed_level1 + excluded.passed_level1,
                        passed_level2 = filter_stats.passed_level2 + excluded.passed_level2,
                        passed_level3 = filter_stats.passed_level3 + excluded.passed_level3,
                        accepted = filter_stats.accepted + excluded.accepted,
                        alerts_sent = filter_stats.alerts_sent + excluded.alerts_sent
                    """,
                    (
                        outcome.watcher_name,
                        outcome_date,
                        int(outcome.rule_passed),
                        int(
                            outcome.embedding_status is StageStatus.OK and outcome.embedding_passed
                        ),
                        int(outcome.llm_status is StageStatus.OK and outcome.llm_passed),
                        int(outcome.accepted),
                        int(outcome.alert_sent),
                    ),
                )
                await self.conn.commit()
                return True
            except BaseException:
                await self.conn.rollback()
                raise

    async def get_pending_pipeline_jobs(self, limit: int = 500) -> list[StoredPipelineJob]:
        """Return durable jobs left unfinished by an interrupted worker."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        async with self._write_lock:
            cursor = await self.conn.execute(
                """
                SELECT
                    p.message_id,
                    p.watcher_name,
                    p.watcher_config_fingerprint,
                    m.chat_id,
                    m.chat_title,
                    m.sender_name,
                    m.text
                FROM pipeline_runs AS p
                JOIN messages AS m ON m.id = p.message_id
                WHERE p.processing_status = 'pending'
                ORDER BY datetime(p.created_at), p.id
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
        return [
            StoredPipelineJob(
                message_id=int(row["message_id"]),
                watcher_name=str(row["watcher_name"]),
                chat_id=int(row["chat_id"]),
                chat_title=str(row["chat_title"] or "Unknown"),
                sender_name=str(row["sender_name"] or "Unknown"),
                text=str(row["text"] or ""),
                watcher_config_fingerprint=str(row["watcher_config_fingerprint"]),
            )
            for row in rows
        ]

    async def mark_pipeline_alert_sent(
        self,
        *,
        message_id: int,
        watcher_name: str,
    ) -> bool:
        """Reconcile a replayed outbox success into provenance and daily stats."""
        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                updated = await self._mark_pipeline_alert_sent_locked(
                    message_id=message_id,
                    watcher_name=watcher_name,
                )
                await self.conn.commit()
                return updated
            except BaseException:
                await self.conn.rollback()
                raise

    async def _mark_pipeline_alert_sent_locked(
        self,
        *,
        message_id: int,
        watcher_name: str,
    ) -> bool:
        """Reconcile delivery provenance inside the caller's transaction."""
        cursor = await self.conn.execute(
            """
            UPDATE pipeline_runs
            SET alert_sent = 1
            WHERE message_id = ?
              AND watcher_name = ?
              AND processing_status = 'completed'
              AND alert_sent = 0
            RETURNING DATE(created_at)
            """,
            (message_id, watcher_name),
        )
        updated = await cursor.fetchone()
        if updated is None:
            return False
        await self.conn.execute(
            """
            INSERT INTO filter_stats (
                watcher_name, date, messages_total, passed_level1,
                passed_level2, passed_level3, accepted, alerts_sent
            )
            VALUES (?, ?, 0, 0, 0, 0, 0, 1)
            ON CONFLICT(watcher_name, date) DO UPDATE SET
                alerts_sent = filter_stats.alerts_sent + 1
            """,
            (watcher_name, str(updated[0])),
        )
        return True

    async def get_daily_messages(
        self,
        chat_ids: list[int],
        date: str,
    ) -> list[DailyMessage]:
        """Fetch all messages for given chats on a specific date (YYYY-MM-DD).

        Returns list of dicts with chat_title, sender_name, text, date.
        """
        if not chat_ids:
            return []
        chat_ids_json = json.dumps(chat_ids, separators=(",", ":"))
        async with self._write_lock:
            cursor = await self.conn.execute(
                """
                SELECT id, chat_id, chat_title, sender_name, text, date
                FROM messages
                WHERE chat_id IN (
                    SELECT CAST(value AS INTEGER)
                    FROM json_each(?)
                )
                  AND DATE(date) = ?
                  AND text IS NOT NULL
                ORDER BY date ASC
                """,
                (chat_ids_json, date),
            )
            rows = await cursor.fetchall()
        return [
            DailyMessage(
                message_id=int(row[0]),
                chat_id=int(row[1]),
                chat_title=str(row[2] or "Unknown"),
                sender_name=str(row[3] or "Unknown"),
                text=str(row[4]),
                date=str(row[5]),
            )
            for row in rows
        ]

    async def get_daily_accepted_messages(
        self,
        watcher_name: str,
        date: str,
    ) -> list[DailyMessage]:
        """Fetch only messages accepted by one watcher for a grounded digest."""
        async with self._write_lock:
            cursor = await self.conn.execute(
                """
                SELECT
                    m.id AS message_id,
                    m.chat_id,
                    m.chat_title,
                    m.sender_name,
                    m.text,
                    m.date
                FROM pipeline_runs AS p
                JOIN messages AS m ON m.id = p.message_id
                WHERE p.watcher_name = ?
                  AND p.processing_status = 'completed'
                  AND p.accepted = 1
                  AND DATE(m.date) = ?
                  AND m.text IS NOT NULL
                ORDER BY datetime(m.date), m.id
                """,
                (watcher_name, date),
            )
            rows = await cursor.fetchall()
        return [
            DailyMessage(
                message_id=int(row["message_id"]),
                chat_id=int(row["chat_id"]),
                chat_title=str(row["chat_title"] or "Unknown"),
                sender_name=str(row["sender_name"] or "Unknown"),
                text=str(row["text"]),
                date=str(row["date"]),
            )
            for row in rows
        ]

    async def get_accepted_messages_between(
        self,
        watcher_name: str,
        window_start: str,
        window_end: str,
    ) -> list[DailyMessage]:
        """Fetch accepted messages in the half-open digest window ``(start, end]``."""
        async with self._write_lock:
            cursor = await self.conn.execute(
                """
                SELECT
                    m.id AS message_id,
                    m.chat_id,
                    m.chat_title,
                    m.sender_name,
                    m.text,
                    m.date
                FROM pipeline_runs AS p
                JOIN messages AS m ON m.id = p.message_id
                WHERE p.watcher_name = ?
                  AND p.processing_status = 'completed'
                  AND p.accepted = 1
                  AND datetime(m.date) > datetime(?)
                  AND datetime(m.date) <= datetime(?)
                  AND m.text IS NOT NULL
                ORDER BY datetime(m.date), m.id
                """,
                (watcher_name, window_start, window_end),
            )
            rows = await cursor.fetchall()
        return [
            DailyMessage(
                message_id=int(row["message_id"]),
                chat_id=int(row["chat_id"]),
                chat_title=str(row["chat_title"] or "Unknown"),
                sender_name=str(row["sender_name"] or "Unknown"),
                text=str(row["text"]),
                date=str(row["date"]),
            )
            for row in rows
        ]

    async def purge_expired_data(self, retention_days: int) -> int:
        """Delete message content older than the configured retention window."""
        async with self._write_lock:
            return await self._purge_expired_data_locked(retention_days)

    async def _purge_expired_data_locked(self, retention_days: int) -> int:
        """Purge terminal content without deleting recoverable work."""
        cutoff = f"-{retention_days} days"
        await self.conn.execute("BEGIN IMMEDIATE")
        try:
            await self.conn.execute(
                """
                CREATE TEMP TABLE IF NOT EXISTS retention_candidates (
                    message_id INTEGER PRIMARY KEY
                )
                """
            )
            await self.conn.execute("DELETE FROM retention_candidates")
            await self.conn.execute(
                """
                INSERT INTO retention_candidates (message_id)
                SELECT m.id
                FROM messages AS m
                WHERE datetime(m.date) < datetime('now', ?)
                  AND NOT EXISTS (
                      SELECT 1
                      FROM pipeline_runs AS p
                      WHERE p.message_id = m.id
                        AND p.processing_status = 'pending'
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM alerts AS a
                      WHERE a.message_id = m.id
                        AND a.delivery_status = 'pending'
                  )
                """,
                (cutoff,),
            )
            cursor = await self.conn.execute("SELECT COUNT(*) FROM retention_candidates")
            row = await cursor.fetchone()
            candidate_count = int(row[0]) if row is not None else 0
            if candidate_count == 0:
                await self.conn.commit()
                return 0

            await self.conn.execute(
                """
                DELETE FROM pipeline_runs
                WHERE message_id IN (
                    SELECT message_id FROM retention_candidates
                )
                """
            )
            await self.conn.execute(
                """
                DELETE FROM alerts
                WHERE message_id IN (
                    SELECT message_id FROM retention_candidates
                )
                """
            )
            await self.conn.execute(
                """
                DELETE FROM messages
                WHERE id IN (
                    SELECT message_id FROM retention_candidates
                )
                """
            )
            await self.conn.execute("DELETE FROM retention_candidates")
            await self.conn.commit()
            return candidate_count
        except BaseException:
            await self.conn.rollback()
            raise
