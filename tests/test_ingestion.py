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
