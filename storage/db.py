"""SQLite database wrapper with async access and auto-migration."""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import TypedDict

import aiosqlite

from pipeline.models import (
    AlertDeliveryStatus,
    AlertOutboxItem,
    DeliveryResult,
    PipelineOutcome,
    StageStatus,
)

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class DailyMessage(TypedDict):
    """Message shape consumed by the daily summarizer."""

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
        self._outbox_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open the database connection and run migrations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.db_path.parent, 0o700)
        self._conn = await aiosqlite.connect(self.db_path)
        os.chmod(self.db_path, 0o600)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
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
                claimed_until = NULL
            WHERE sent_at IS NOT NULL
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
            "embedding_model": "ALTER TABLE pipeline_runs ADD COLUMN embedding_model TEXT",
            "embedding_latency_ms": (
                "ALTER TABLE pipeline_runs ADD COLUMN embedding_latency_ms REAL"
            ),
            "llm_model": "ALTER TABLE pipeline_runs ADD COLUMN llm_model TEXT",
            "llm_prompt_version": ("ALTER TABLE pipeline_runs ADD COLUMN llm_prompt_version TEXT"),
            "llm_latency_ms": "ALTER TABLE pipeline_runs ADD COLUMN llm_latency_ms REAL",
        }
        for column, statement in provenance_migrations.items():
            if column not in pipeline_columns:
                await self.conn.execute(statement)
                logger.info("Migration: added pipeline_runs.%s", column)

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
        sender_id: int | None,
        sender_name: str | None,
        text: str | None,
        date: str,
        raw_json: str | None = None,
    ) -> int | None:
        """Store a message and return its row ID. Returns None if duplicate."""
        try:
            cursor = await self.conn.execute(
                """
                INSERT INTO messages
                    (telegram_msg_id, chat_id, chat_title, sender_id, sender_name, text, date, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            await self.conn.commit()
            return cursor.lastrowid
        except aiosqlite.IntegrityError:
            logger.debug("Duplicate message %d in chat %d", telegram_msg_id, chat_id)
            return None

    async def store_alert(
        self,
        *,
        watcher_name: str,
        message_id: int,
        filter_level: int,
        score: float | None = None,
        llm_response: str | None = None,
    ) -> int:
        """Idempotently enqueue an alert and return its durable outbox row ID.

        Re-evaluating the same watcher/message refreshes the payload but never
        resets terminal delivery state or attempt history.
        """
        async with self._outbox_lock:
            cursor = await self.conn.execute(
                """
                INSERT INTO alerts (watcher_name, message_id, filter_level, score, llm_response)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(watcher_name, message_id) DO UPDATE SET
                    filter_level = excluded.filter_level,
                    score = excluded.score,
                    llm_response = excluded.llm_response
                RETURNING id
                """,
                (watcher_name, message_id, filter_level, score, llm_response),
            )
            row = await cursor.fetchone()
            await self.conn.commit()
            if row is None:
                raise RuntimeError("SQLite did not return an alert row ID")
            return int(row[0])

    async def mark_alert_sent(self, alert_id: int) -> None:
        """Compatibility wrapper for the existing immediate-delivery path."""
        await self.mark_alert_delivery_result(
            alert_id,
            DeliveryResult.success(),
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

        async with self._outbox_lock:
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
        await self.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self.conn.execute(
                """
                WITH due AS (
                    SELECT id
                    FROM alerts
                    WHERE delivery_status = 'pending'
                      AND (
                          next_attempt_at IS NULL
                          OR datetime(next_attempt_at) <= CURRENT_TIMESTAMP
                      )
                      AND (
                          claimed_until IS NULL
                          OR datetime(claimed_until) <= CURRENT_TIMESTAMP
                      )
                    ORDER BY datetime(created_at), id
                    LIMIT ?
                )
                UPDATE alerts
                SET claimed_until = datetime('now', ?)
                WHERE id IN (SELECT id FROM due)
                RETURNING id
                """,
                (limit, lease_modifier),
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
                    m.chat_title,
                    m.sender_name,
                    m.text
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
                filter_level=int(row["filter_level"]),
                delivery_attempts=int(row["delivery_attempts"]),
            )
            for row in rows
        ]

    async def mark_alert_delivery_result(
        self,
        alert_id: int,
        result: DeliveryResult,
        *,
        max_attempts: int = 5,
        base_delay_seconds: int = 5,
    ) -> AlertDeliveryStatus:
        """Persist one outbox delivery cycle and return the resulting state."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if base_delay_seconds < 1:
            raise ValueError("base_delay_seconds must be at least 1")

        async with self._outbox_lock:
            return await self._mark_alert_delivery_result_locked(
                alert_id,
                result,
                max_attempts=max_attempts,
                base_delay_seconds=base_delay_seconds,
            )

    async def _mark_alert_delivery_result_locked(
        self,
        alert_id: int,
        result: DeliveryResult,
        *,
        max_attempts: int,
        base_delay_seconds: int,
    ) -> AlertDeliveryStatus:
        """Update one delivery while the process-local outbox lock is held."""
        await self.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self.conn.execute(
                """
                SELECT delivery_status, delivery_attempts
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
                        sent_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (status.value, attempts, alert_id),
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
                            claimed_until = NULL
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
                            claimed_until = NULL
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

    async def record_pipeline_outcome(self, outcome: PipelineOutcome) -> bool:
        """Persist one message/watcher outcome and update aggregate stats once.

        Returns True when a new run was inserted and False for an idempotent
        duplicate. Aggregate counters only change for a new run.
        """
        async with self.conn.execute(
            """
            INSERT INTO pipeline_runs (
                message_id, watcher_name, rule_passed,
                embedding_status, embedding_passed, embedding_score,
                embedding_model, embedding_latency_ms,
                llm_status, llm_relevant, llm_verdict, llm_confidence,
                llm_model, llm_prompt_version, llm_latency_ms,
                alert_created, alert_sent, error_code
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id, watcher_name) DO NOTHING
            RETURNING id
            """,
            (
                outcome.message_id,
                outcome.watcher_name,
                outcome.rule_passed,
                outcome.embedding_status.value,
                outcome.embedding_passed,
                outcome.embedding_score,
                outcome.embedding_model,
                outcome.embedding_latency_ms,
                outcome.llm_status.value,
                outcome.llm_relevant,
                outcome.llm_verdict,
                outcome.llm_confidence,
                outcome.llm_model,
                outcome.llm_prompt_version,
                outcome.llm_latency_ms,
                outcome.alert_created,
                outcome.alert_sent,
                outcome.error_code,
            ),
        ) as cursor:
            inserted = await cursor.fetchone()

        if inserted is None:
            await self.conn.commit()
            return False

        await self.conn.execute(
            """
            INSERT INTO filter_stats (
                watcher_name, date, messages_total,
                passed_level1, passed_level2, passed_level3, alerts_sent
            )
            VALUES (?, DATE('now'), 1, ?, ?, ?, ?)
            ON CONFLICT(watcher_name, date) DO UPDATE SET
                messages_total = filter_stats.messages_total + 1,
                passed_level1 = filter_stats.passed_level1 + excluded.passed_level1,
                passed_level2 = filter_stats.passed_level2 + excluded.passed_level2,
                passed_level3 = filter_stats.passed_level3 + excluded.passed_level3,
                alerts_sent = filter_stats.alerts_sent + excluded.alerts_sent
            """,
            (
                outcome.watcher_name,
                int(outcome.rule_passed),
                int(outcome.embedding_status is StageStatus.OK and outcome.embedding_passed),
                int(outcome.llm_status is StageStatus.OK and outcome.llm_relevant),
                int(outcome.alert_sent),
            ),
        )
        await self.conn.commit()
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
        cursor = await self.conn.execute(
            """
            SELECT chat_id, chat_title, sender_name, text, date
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
                chat_id=row[0],
                chat_title=row[1] or "Unknown",
                sender_name=row[2] or "Unknown",
                text=row[3],
                date=row[4],
            )
            for row in rows
        ]

    async def purge_expired_data(self, retention_days: int) -> int:
        """Delete message content older than the configured retention window."""
        async with self._outbox_lock:
            return await self._purge_expired_data_locked(retention_days)

    async def _purge_expired_data_locked(self, retention_days: int) -> int:
        """Purge content without racing an active outbox claim."""
        cutoff = f"-{retention_days} days"
        await self.conn.execute(
            """
            DELETE FROM pipeline_runs
            WHERE message_id IN (
                SELECT id FROM messages
                WHERE datetime(date) < datetime('now', ?)
            )
            """,
            (cutoff,),
        )
        await self.conn.execute(
            """
            DELETE FROM alerts
            WHERE message_id IN (
                SELECT id FROM messages
                WHERE datetime(date) < datetime('now', ?)
            )
            """,
            (cutoff,),
        )
        cursor = await self.conn.execute(
            "DELETE FROM messages WHERE datetime(date) < datetime('now', ?)",
            (cutoff,),
        )
        await self.conn.commit()
        return max(cursor.rowcount, 0)
