"""Integration tests for idempotent Telegram message ingestion."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pipeline.ingestion import ingest_message
from storage.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "ingestion.db")
    await database.connect()
    yield database
    await database.close()


def _telegram_event() -> SimpleNamespace:
    message = SimpleNamespace(
        id=42,
        chat_id=-100123,
        text="Villa available near the beach",
        date=datetime(2026, 3, 4, 12, 0, tzinfo=UTC),
        to_dict=lambda: {"id": 42, "message": "Villa available near the beach"},
    )
    return SimpleNamespace(
        message=message,
        chat_id=-100123,
        get_sender=AsyncMock(return_value=SimpleNamespace(id=7, username="alice")),
        get_chat=AsyncMock(return_value=SimpleNamespace(title="Test Group")),
    )


async def test_duplicate_ingestion_does_not_increment_chat_count(
    db: Database,
) -> None:
    """A replayed Telegram update must not inflate denormalized chat metrics."""
    event = _telegram_event()

    first_id = await ingest_message(event, db)
    duplicate_id = await ingest_message(event, db)

    assert first_id is not None
    assert duplicate_id is None

    cursor = await db.conn.execute(
        "SELECT message_count FROM chats WHERE chat_id = ?",
        (-100123,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 1

    cursor = await db.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE chat_id = ?",
        (-100123,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 1


async def test_ingestion_enqueues_expected_watcher_jobs(db: Database) -> None:
    event = _telegram_event()

    message_id = await ingest_message(
        event,
        db,
        watcher_names=("housing-watch", "audit-watch"),
        watcher_fingerprints={
            "housing-watch": "housing-policy-v1",
            "audit-watch": "audit-policy-v1",
        },
    )

    assert message_id is not None
    jobs = await db.get_pending_pipeline_jobs()
    assert {(job.message_id, job.watcher_name, job.text) for job in jobs} == {
        (message_id, "housing-watch", "Villa available near the beach"),
        (message_id, "audit-watch", "Villa available near the beach"),
    }


async def test_ingestion_prefers_sender_handle_when_available(db: Database) -> None:
    message_id = await ingest_message(_telegram_event(), db)

    assert message_id is not None
    cursor = await db.conn.execute("SELECT sender_name FROM messages WHERE id = ?", (message_id,))
    assert (await cursor.fetchone())[0] == "@alice"


async def test_metadata_lookup_failure_does_not_drop_update(db: Database) -> None:
    event = _telegram_event()
    event.get_sender.side_effect = RuntimeError("telegram lookup failed")
    event.get_chat.side_effect = TimeoutError("telegram lookup timed out")

    message_id = await ingest_message(event, db)

    assert message_id is not None
    cursor = await db.conn.execute(
        "SELECT sender_name, chat_title FROM messages WHERE id = ?",
        (message_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == ("Unknown", "Unknown")


async def test_optional_raw_payload_failure_does_not_drop_update(db: Database) -> None:
    event = _telegram_event()

    def fail_to_dict() -> dict[str, object]:
        raise RuntimeError("serialization failed")

    event.message.to_dict = fail_to_dict

    message_id = await ingest_message(event, db, store_raw_json=True)

    assert message_id is not None
    cursor = await db.conn.execute("SELECT raw_json FROM messages WHERE id = ?", (message_id,))
    assert (await cursor.fetchone())[0] is None
