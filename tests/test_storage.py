"""Tests for storage/db.py — SQLite wrapper."""

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pipeline.models import (
    AlertDeliveryStatus,
    DeliveryResult,
    PipelineOutcome,
    StageStatus,
)
from storage.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Create an in-memory-like database in tmp_path for testing."""
    database = Database(tmp_path / "test.db")
    await database.connect()
    yield database
    await database.close()


async def test_connect_creates_tables(db: Database) -> None:
    """Database.connect() should run schema.sql and create all tables."""
    cursor = await db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in await cursor.fetchall()]
    assert "messages" in tables
    assert "alerts" in tables
    assert "chats" in tables
    assert "filter_stats" in tables
    assert "pipeline_runs" in tables


async def test_connect_migrates_legacy_alerts_without_losing_sent_state(
    tmp_path: Path,
) -> None:
    """Incremental migration should deduplicate legacy rows and recover delivery state."""
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_msg_id INTEGER NOT NULL,
            chat_id INTEGER NOT NULL,
            chat_title TEXT,
            sender_id INTEGER,
            sender_name TEXT,
            text TEXT,
            date TIMESTAMP NOT NULL,
            raw_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(chat_id, telegram_msg_id)
        );
        CREATE TABLE alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            watcher_name TEXT NOT NULL,
            message_id INTEGER REFERENCES messages(id),
            filter_level INTEGER NOT NULL,
            score REAL,
            llm_response TEXT,
            sent_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO messages (
            telegram_msg_id, chat_id, sender_name, text, date
        ) VALUES (
            1, -100900, 'Legacy', 'Legacy alert', CURRENT_TIMESTAMP
        );
        INSERT INTO alerts (
            watcher_name, message_id, filter_level, sent_at
        ) VALUES (
            'legacy-watcher', 1, 1, CURRENT_TIMESTAMP
        );
        INSERT INTO alerts (
            watcher_name, message_id, filter_level, sent_at
        ) VALUES (
            'legacy-watcher', 1, 3, NULL
        );
        """
    )
    legacy.commit()
    legacy.close()

    migrated = Database(db_path)
    await migrated.connect()
    try:
        cursor = await migrated.conn.execute(
            """
            SELECT
                COUNT(*),
                MIN(delivery_status),
                MIN(delivery_attempts),
                MIN(next_attempt_at)
            FROM alerts
            WHERE watcher_name = 'legacy-watcher' AND message_id = 1
            """
        )
        row = await cursor.fetchone()
        assert row is not None
        assert tuple(row) == (1, "sent", 0, None)
        assert await migrated.claim_due_alerts() == []

        cursor = await migrated.conn.execute("PRAGMA table_info(pipeline_runs)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert {
            "embedding_model",
            "embedding_latency_ms",
            "llm_model",
            "llm_prompt_version",
            "llm_latency_ms",
        } <= columns
    finally:
        await migrated.close()


async def test_store_message(db: Database) -> None:
    """store_message should insert a row and return its ID."""
    row_id = await db.store_message(
        telegram_msg_id=42,
        chat_id=-100123,
        sender_id=999,
        sender_name="Alice",
        text="Hello world",
        date="2026-03-04 12:00:00",
    )
    assert row_id is not None
    assert row_id > 0

    cursor = await db.conn.execute("SELECT text FROM messages WHERE id = ?", (row_id,))
    row = await cursor.fetchone()
    assert row[0] == "Hello world"


async def test_store_message_duplicate_returns_none(db: Database) -> None:
    """Duplicate (chat_id + telegram_msg_id) should return None."""
    await db.store_message(
        telegram_msg_id=1,
        chat_id=-100,
        sender_id=1,
        sender_name="Bob",
        text="First",
        date="2026-03-04 12:00:00",
    )
    result = await db.store_message(
        telegram_msg_id=1,
        chat_id=-100,
        sender_id=1,
        sender_name="Bob",
        text="Duplicate",
        date="2026-03-04 12:00:01",
    )
    assert result is None


async def test_store_alert(db: Database) -> None:
    """store_alert should insert an alert linked to a message."""
    msg_id = await db.store_message(
        telegram_msg_id=10,
        chat_id=-200,
        sender_id=5,
        sender_name="Eve",
        text="Villa for rent",
        date="2026-03-04 13:00:00",
    )
    alert_id = await db.store_alert(
        watcher_name="phangan-housing",
        message_id=msg_id,
        filter_level=1,
        score=0.95,
    )
    assert alert_id > 0


async def test_store_alert_upserts_unique_watcher_message_pair(db: Database) -> None:
    """Reprocessing one watcher/message pair should update, not duplicate, its alert."""
    msg_id = await db.store_message(
        telegram_msg_id=12,
        chat_id=-200,
        sender_id=5,
        sender_name="Eve",
        text="Villa for rent",
        date="2026-03-04 13:00:00",
    )
    assert msg_id is not None

    first_id = await db.store_alert(
        watcher_name="phangan-housing",
        message_id=msg_id,
        filter_level=1,
        score=0.70,
        llm_response="first",
    )
    second_id = await db.store_alert(
        watcher_name="phangan-housing",
        message_id=msg_id,
        filter_level=3,
        score=0.91,
        llm_response="updated",
    )

    assert second_id == first_id
    cursor = await db.conn.execute(
        """
        SELECT COUNT(*), MIN(filter_level), MIN(score), MIN(llm_response)
        FROM alerts
        WHERE watcher_name = ? AND message_id = ?
        """,
        ("phangan-housing", msg_id),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == (1, 3, 0.91, "updated")

    await db.mark_alert_sent(first_id)
    third_id = await db.store_alert(
        watcher_name="phangan-housing",
        message_id=msg_id,
        filter_level=3,
        score=0.93,
        llm_response="reprocessed",
    )
    assert third_id == first_id
    cursor = await db.conn.execute(
        """
        SELECT delivery_status, delivery_attempts, last_error
        FROM alerts
        WHERE id = ?
        """,
        (first_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == ("sent", 1, None)


async def test_mark_alert_sent(db: Database) -> None:
    """mark_alert_sent should set sent_at timestamp."""
    msg_id = await db.store_message(
        telegram_msg_id=11,
        chat_id=-200,
        sender_id=5,
        sender_name="Eve",
        text="Another villa",
        date="2026-03-04 14:00:00",
    )
    alert_id = await db.store_alert(
        watcher_name="test",
        message_id=msg_id,
        filter_level=1,
    )
    await db.mark_alert_sent(alert_id)
    await db.mark_alert_sent(alert_id)

    cursor = await db.conn.execute(
        """
        SELECT sent_at, delivery_status, delivery_attempts
        FROM alerts
        WHERE id = ?
        """,
        (alert_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] is not None
    assert tuple(row[1:]) == ("sent", 1)


async def test_pending_alert_survives_restart_and_claim_is_leased(
    tmp_path: Path,
) -> None:
    """A committed pending alert should be claimable after process restart."""
    db_path = tmp_path / "restart.db"
    initial = Database(db_path)
    await initial.connect()
    message_id = await initial.store_message(
        telegram_msg_id=201,
        chat_id=-100800,
        chat_title="Recovery Chat",
        sender_id=8,
        sender_name="Alice",
        text="Recover this alert",
        date="2026-07-17 12:00:00",
    )
    assert message_id is not None
    alert_id = await initial.store_alert(
        watcher_name="recovery-watcher",
        message_id=message_id,
        filter_level=2,
    )
    assert len(await initial.claim_due_alerts(limit=10, lease_seconds=60)) == 1
    # Simulate process downtime lasting beyond the persisted delivery lease.
    await initial.conn.execute(
        "UPDATE alerts SET claimed_until = datetime('now', '-1 second') WHERE id = ?",
        (alert_id,),
    )
    await initial.conn.commit()
    await initial.close()

    recovered = Database(db_path)
    await recovered.connect()
    try:
        claimed = await recovered.claim_due_alerts(limit=10, lease_seconds=60)
        assert len(claimed) == 1
        assert claimed[0].alert_id == alert_id
        assert claimed[0].watcher_name == "recovery-watcher"
        assert claimed[0].chat_title == "Recovery Chat"
        assert claimed[0].sender_name == "Alice"
        assert claimed[0].text == "Recover this alert"
        assert claimed[0].delivery_attempts == 0

        assert await recovered.claim_due_alerts(limit=10, lease_seconds=60) == []
    finally:
        await recovered.close()


async def test_concurrent_claimers_receive_each_alert_once(db: Database) -> None:
    """Process-local workers must not claim the same pending outbox row."""
    message_id = await db.store_message(
        telegram_msg_id=211,
        chat_id=-100810,
        sender_id=8,
        sender_name="Alice",
        text="Claim exactly once",
        date="2026-07-17 12:00:00",
    )
    assert message_id is not None
    alert_id = await db.store_alert(
        watcher_name="claim-watcher",
        message_id=message_id,
        filter_level=1,
    )

    batches = await asyncio.gather(
        db.claim_due_alerts(limit=1, lease_seconds=60),
        db.claim_due_alerts(limit=1, lease_seconds=60),
    )

    claimed_ids = [item.alert_id for batch in batches for item in batch]
    assert claimed_ids == [alert_id]


async def test_retryable_delivery_reschedules_without_persisting_error_text(
    db: Database,
) -> None:
    """Retryable failure should release its lease and schedule another attempt."""
    message_id = await db.store_message(
        telegram_msg_id=202,
        chat_id=-100801,
        sender_id=8,
        sender_name="Alice",
        text="Retry this alert",
        date="2026-07-17 12:00:00",
    )
    assert message_id is not None
    alert_id = await db.store_alert(
        watcher_name="retry-watcher",
        message_id=message_id,
        filter_level=1,
    )
    assert len(await db.claim_due_alerts(lease_seconds=60)) == 1

    status = await db.mark_alert_delivery_result(
        alert_id,
        DeliveryResult(
            sent=False,
            retryable=True,
            error_code="timeout for Alice: Retry this alert",
            retry_after=30,
        ),
    )

    assert status is AlertDeliveryStatus.PENDING
    cursor = await db.conn.execute(
        """
        SELECT
            delivery_status,
            delivery_attempts,
            last_error,
            claimed_until,
            datetime(next_attempt_at) > CURRENT_TIMESTAMP
        FROM alerts
        WHERE id = ?
        """,
        (alert_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == ("pending", 1, "delivery_error", None, 1)
    assert await db.claim_due_alerts() == []


@pytest.mark.parametrize(
    ("result", "max_attempts"),
    [
        (
            DeliveryResult(
                sent=False,
                retryable=False,
                error_code="telegram_http_403",
            ),
            5,
        ),
        (
            DeliveryResult(
                sent=False,
                retryable=True,
                error_code="telegram_timeout",
                retry_after=1,
            ),
            1,
        ),
    ],
)
async def test_terminal_or_exhausted_delivery_is_marked_failed(
    db: Database,
    result: DeliveryResult,
    max_attempts: int,
) -> None:
    """Terminal provider errors and exhausted retries should leave the due queue."""
    message_id = await db.store_message(
        telegram_msg_id=203 + max_attempts,
        chat_id=-100802,
        sender_id=9,
        sender_name="Bob",
        text="Do not retry forever",
        date="2026-07-17 12:00:00",
    )
    assert message_id is not None
    alert_id = await db.store_alert(
        watcher_name=f"terminal-{max_attempts}",
        message_id=message_id,
        filter_level=1,
    )

    status = await db.mark_alert_delivery_result(
        alert_id,
        result,
        max_attempts=max_attempts,
    )

    assert status is AlertDeliveryStatus.FAILED
    cursor = await db.conn.execute(
        """
        SELECT delivery_status, delivery_attempts, next_attempt_at, last_error
        FROM alerts
        WHERE id = ?
        """,
        (alert_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == ("failed", 1, None, result.error_code)
    assert await db.claim_due_alerts() == []


async def test_update_chat_upsert(db: Database) -> None:
    """update_chat should insert on first call, increment count on second."""
    await db.update_chat(chat_id=-300, title="Test Group", chat_type="supergroup")
    await db.update_chat(chat_id=-300)

    cursor = await db.conn.execute(
        "SELECT title, message_count FROM chats WHERE chat_id = ?", (-300,)
    )
    row = await cursor.fetchone()
    assert row[0] == "Test Group"
    assert row[1] == 2


async def test_record_pipeline_outcome_counts_once_and_is_idempotent(
    db: Database,
) -> None:
    """One message/watcher outcome should affect aggregate counters exactly once."""
    msg_id = await db.store_message(
        telegram_msg_id=13,
        chat_id=-300,
        sender_id=7,
        sender_name="Alice",
        text="Villa for rent",
        date="2026-03-04 15:00:00",
    )
    assert msg_id is not None
    outcome = PipelineOutcome(
        message_id=msg_id,
        watcher_name="test-watcher",
        rule_passed=True,
        embedding_status=StageStatus.OK,
        embedding_passed=True,
        embedding_score=0.88,
        embedding_model="text-embedding-test",
        embedding_latency_ms=12.5,
        llm_status=StageStatus.OK,
        llm_relevant=True,
        llm_verdict="offer",
        llm_confidence=0.95,
        llm_model="test-model",
        llm_prompt_version="classifier-v2",
        llm_latency_ms=42.0,
        alert_created=True,
        alert_sent=True,
    )

    assert await db.record_pipeline_outcome(outcome) is True
    assert await db.record_pipeline_outcome(outcome) is False

    cursor = await db.conn.execute(
        """
        SELECT messages_total, passed_level1, passed_level2, passed_level3, alerts_sent
        FROM filter_stats
        WHERE watcher_name = ?
        """,
        ("test-watcher",),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == (1, 1, 1, 1, 1)

    cursor = await db.conn.execute(
        "SELECT COUNT(*) FROM pipeline_runs WHERE message_id = ? AND watcher_name = ?",
        (msg_id, "test-watcher"),
    )
    assert (await cursor.fetchone())[0] == 1

    cursor = await db.conn.execute(
        """
        SELECT
            embedding_model,
            embedding_latency_ms,
            llm_model,
            llm_prompt_version,
            llm_latency_ms
        FROM pipeline_runs
        WHERE message_id = ? AND watcher_name = ?
        """,
        (msg_id, "test-watcher"),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == (
        "text-embedding-test",
        12.5,
        "test-model",
        "classifier-v2",
        42.0,
    )


async def test_store_message_with_chat_title(db: Database) -> None:
    """store_message should store chat_title alongside other fields."""
    row_id = await db.store_message(
        telegram_msg_id=50,
        chat_id=-100500,
        chat_title="Phangan Expats",
        sender_id=1,
        sender_name="Alice",
        text="Hello group",
        date="2026-03-05 10:00:00",
    )
    cursor = await db.conn.execute("SELECT chat_title FROM messages WHERE id = ?", (row_id,))
    row = await cursor.fetchone()
    assert row[0] == "Phangan Expats"


async def test_get_daily_messages(db: Database) -> None:
    """get_daily_messages should return messages for given chats on a specific date."""
    chat_id = -100600
    # Insert messages on target date
    await db.store_message(
        telegram_msg_id=1,
        chat_id=chat_id,
        chat_title="Test Chat",
        sender_id=1,
        sender_name="Alice",
        text="Morning msg",
        date="2026-03-05 08:00:00",
    )
    await db.store_message(
        telegram_msg_id=2,
        chat_id=chat_id,
        chat_title="Test Chat",
        sender_id=2,
        sender_name="Bob",
        text="Afternoon msg",
        date="2026-03-05 14:00:00",
    )
    # Insert message on different date (should be excluded)
    await db.store_message(
        telegram_msg_id=3,
        chat_id=chat_id,
        chat_title="Test Chat",
        sender_id=1,
        sender_name="Alice",
        text="Yesterday msg",
        date="2026-03-04 12:00:00",
    )
    # Insert message in different chat (should be excluded)
    await db.store_message(
        telegram_msg_id=4,
        chat_id=-100999,
        chat_title="Other Chat",
        sender_id=3,
        sender_name="Eve",
        text="Other chat msg",
        date="2026-03-05 09:00:00",
    )

    messages = await db.get_daily_messages([chat_id], "2026-03-05")
    assert len(messages) == 2
    assert messages[0]["text"] == "Morning msg"
    assert messages[1]["text"] == "Afternoon msg"
    assert messages[0]["chat_title"] == "Test Chat"


async def test_get_daily_messages_empty(db: Database) -> None:
    """get_daily_messages with no matching data should return empty list."""
    messages = await db.get_daily_messages([-100999], "2026-01-01")
    assert messages == []

    messages = await db.get_daily_messages([], "2026-03-05")
    assert messages == []


async def test_purge_expired_data_removes_related_rows_and_keeps_recent_data(
    db: Database,
) -> None:
    """Retention should purge expired message content and its dependent records only."""
    now = datetime.now(UTC)
    old_id = await db.store_message(
        telegram_msg_id=101,
        chat_id=-100700,
        sender_id=1,
        sender_name="Old",
        text="Expired message",
        date=(now - timedelta(days=31)).isoformat(),
    )
    recent_id = await db.store_message(
        telegram_msg_id=102,
        chat_id=-100700,
        sender_id=2,
        sender_name="Recent",
        text="Recent message",
        date=(now - timedelta(days=1)).isoformat(),
    )
    assert old_id is not None
    assert recent_id is not None

    for message_id in (old_id, recent_id):
        await db.store_alert(
            watcher_name="retention-watcher",
            message_id=message_id,
            filter_level=1,
        )
        await db.record_pipeline_outcome(
            PipelineOutcome(
                message_id=message_id,
                watcher_name="retention-watcher",
                rule_passed=True,
                alert_created=True,
            )
        )

    assert await db.purge_expired_data(retention_days=30) == 1

    retention_queries = {
        "messages": "SELECT id FROM messages",
        "alerts": "SELECT message_id FROM alerts",
        "pipeline_runs": "SELECT message_id FROM pipeline_runs",
    }
    for query in retention_queries.values():
        cursor = await db.conn.execute(query)
        remaining_ids = {row[0] for row in await cursor.fetchall()}
        assert remaining_ids == {recent_id}


async def test_conn_raises_when_not_connected(tmp_path: Path) -> None:
    """Accessing conn before connect() should raise RuntimeError."""
    db = Database(tmp_path / "nonexistent.db")
    with pytest.raises(RuntimeError, match="not connected"):
        _ = db.conn
