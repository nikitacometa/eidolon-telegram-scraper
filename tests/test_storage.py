"""Tests for storage/db.py — SQLite wrapper."""

from pathlib import Path

import pytest

from storage.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
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

    cursor = await db.conn.execute("SELECT sent_at FROM alerts WHERE id = ?", (alert_id,))
    row = await cursor.fetchone()
    assert row[0] is not None


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


async def test_update_filter_stats(db: Database) -> None:
    """update_filter_stats should track message counts per watcher per day."""
    await db.update_filter_stats(watcher_name="test-watcher", level_passed=1)
    await db.update_filter_stats(watcher_name="test-watcher")
    await db.update_filter_stats(watcher_name="test-watcher", alert_sent=True)

    cursor = await db.conn.execute(
        "SELECT messages_total, passed_level1, alerts_sent FROM filter_stats WHERE watcher_name = ?",
        ("test-watcher",),
    )
    row = await cursor.fetchone()
    assert row[0] == 3  # 3 total messages
    assert row[1] == 1  # 1 passed level 1
    assert row[2] == 1  # 1 alert sent


async def test_conn_raises_when_not_connected() -> None:
    """Accessing conn before connect() should raise RuntimeError."""
    db = Database(Path("/tmp/nonexistent.db"))
    with pytest.raises(RuntimeError, match="not connected"):
        _ = db.conn
