"""Offline coverage for Telegram reply linkage."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from pipeline.crawler import TelegramCrawler
from pipeline.recon_models import ScoutMessage
from storage.db import Database
from storage.scout import ScoutDatabase
from storage.search import SearchDatabase, content_digest


async def test_reply_field_survives_crawler_store_and_corpus_sync(tmp_path: Path) -> None:
    """The nested Telethon header must reach the derived corpus unchanged."""
    live = Database(tmp_path / "live.db")
    scout = ScoutDatabase(tmp_path / "scout.db")
    await live.connect()
    await scout.connect()
    try:
        crawler = TelegramCrawler(client=object(), governor=SimpleNamespace())
        page = crawler._read_page(
            SimpleNamespace(
                messages=[
                    SimpleNamespace(
                        id=42,
                        message="citi dental",
                        date=datetime(2026, 8, 12, tzinfo=UTC),
                        entities=None,
                        from_id=None,
                        fwd_from=None,
                        reply_to=SimpleNamespace(reply_to_msg_id=17),
                    )
                ],
                users=[],
            ),
            chat_id=-100123,
        )
        assert page.messages[0].reply_to_message_id == 17
        assert await scout.store_messages(page.messages) == 1

        search = SearchDatabase(tmp_path / "search.db")
        search.connect()
        try:
            search.sync(live_db=live.db_path, scout_db=scout.db_path)
            row = search.conn.execute(
                "SELECT reply_to_message_id FROM corpus_messages WHERE chat_id = -100123"
            ).fetchone()
            assert row is not None
            assert row["reply_to_message_id"] == 17
        finally:
            search.close()
    finally:
        await scout.close()
        await live.close()


async def test_raw_json_reply_backfill_is_idempotent(tmp_path: Path) -> None:
    """A second pass must neither rewrite rows nor recount settled non-replies."""
    database = Database(tmp_path / "live.db")
    await database.connect()
    try:
        await database.conn.executemany(
            """
            INSERT INTO messages (telegram_msg_id, chat_id, text, date, raw_json)
            VALUES (?, -100123, 'legacy', '2026-08-12', ?)
            """,
            [
                (1, json.dumps({"reply_to": {"reply_to_msg_id": 77}})),
                (2, json.dumps({"reply_to": None})),
            ],
        )
        await database.conn.commit()

        first = await database.backfill_reply_to_message_ids(limit=10)
        second = await database.backfill_reply_to_message_ids(limit=10)

        assert first.as_dict() == {
            "scanned": 2,
            "updated": 1,
            "no_reply": 1,
            "invalid_json": 0,
            "remaining": 0,
        }
        assert second.as_dict() == {
            "scanned": 0,
            "updated": 0,
            "no_reply": 0,
            "invalid_json": 0,
            "remaining": 0,
        }
        rows = await (
            await database.conn.execute(
                "SELECT telegram_msg_id, reply_to_message_id, reply_backfill_checked "
                "FROM messages ORDER BY telegram_msg_id"
            )
        ).fetchall()
        assert [tuple(row) for row in rows] == [(1, 77, 1), (2, None, 1)]
    finally:
        await database.close()


async def test_rescrape_enriches_existing_scout_reply_without_recounting(tmp_path: Path) -> None:
    scout = ScoutDatabase(tmp_path / "scout.db")
    live = Database(tmp_path / "live.db")
    await scout.connect()
    await live.connect()
    try:
        original = ScoutMessage(
            chat_id=-100123,
            telegram_msg_id=42,
            date="2026-08-12",
            text="citi dental",
            source="backfill",
        )
        enriched = ScoutMessage(
            chat_id=-100123,
            telegram_msg_id=42,
            date="2026-08-12",
            text="citi dental",
            reply_to_message_id=17,
            source="backfill",
        )

        assert await scout.store_message(original)
        search = SearchDatabase(tmp_path / "search.db")
        search.connect()
        search.sync(live_db=live.db_path, scout_db=scout.db_path)
        assert not await scout.store_message(enriched)
        row = await (
            await scout.conn.execute(
                "SELECT reply_to_message_id FROM scout_messages WHERE chat_id = -100123"
            )
        ).fetchone()
        assert row is not None
        assert row["reply_to_message_id"] == 17
        report = search.sync(live_db=live.db_path, scout_db=scout.db_path)
        assert report["reply_links"] == 1
        corpus = search.conn.execute(
            "SELECT reply_to_message_id FROM corpus_messages WHERE chat_id = -100123"
        ).fetchone()
        assert corpus is not None
        assert corpus["reply_to_message_id"] == 17
        search.close()
    finally:
        await scout.close()
        await live.close()


def _insert_corpus_message(
    search: SearchDatabase,
    *,
    telegram_msg_id: int,
    text: str,
    reply_to_message_id: int | None = None,
    chat_id: int = -100123,
) -> None:
    search.conn.execute(
        """
        INSERT INTO corpus_messages (
            source, chat_id, telegram_msg_id, reply_to_message_id, text, date, content_hash
        ) VALUES ('live', ?, ?, ?, ?, '2026-08-12', ?)
        """,
        (chat_id, telegram_msg_id, reply_to_message_id, text, content_digest(text)),
    )
    search.conn.commit()


def test_reply_with_absent_parent_returns_none(tmp_path: Path) -> None:
    search = SearchDatabase(tmp_path / "search.db")
    search.connect()
    try:
        _insert_corpus_message(
            search,
            telegram_msg_id=42,
            text="citi dental",
            reply_to_message_id=17,
        )
        answer = search.recent(limit=1)[0]

        assert answer.reply_to_message_id == 17
        assert search.parent_text_for_reply(answer) is None
    finally:
        search.close()


def test_non_reply_returns_none(tmp_path: Path) -> None:
    search = SearchDatabase(tmp_path / "search.db")
    search.connect()
    try:
        _insert_corpus_message(search, telegram_msg_id=42, text="ordinary announcement")
        message = search.recent(limit=1)[0]

        assert message.reply_to_message_id is None
        assert search.parent_text_for_reply(message) is None
    finally:
        search.close()


def test_reply_measurement_names_each_population(tmp_path: Path) -> None:
    search = SearchDatabase(tmp_path / "search.db")
    search.connect()
    try:
        _insert_corpus_message(search, telegram_msg_id=10, text="Где хороший стоматолог?")
        _insert_corpus_message(
            search,
            telegram_msg_id=11,
            text="citi dental",
            reply_to_message_id=10,
        )
        _insert_corpus_message(
            search,
            telegram_msg_id=12,
            text="maps.example/dentist",
            reply_to_message_id=999,
        )
        _insert_corpus_message(
            search,
            telegram_msg_id=20,
            text="Other chat reply",
            reply_to_message_id=19,
            chat_id=-200,
        )

        whole = search.reply_linkage_stats()
        scoped = search.reply_linkage_stats(chat_id=-100123)

        assert whole["rows_total"] == 4
        assert whole["reply_rows"] == {"count": 3, "population": "rows_total"}
        assert whole["replies_with_parent"] == {"count": 1, "population": "reply_rows"}
        question = whole["replies_whose_parent_is_question"]
        assert isinstance(question, dict)
        assert question["count"] == 1
        assert question["population"] == "replies_with_parent"
        assert scoped["scope"] == "chat"
        assert scoped["chat_id"] == -100123
        assert scoped["rows_total"] == 3
    finally:
        search.close()


async def test_scout_migration_adds_reply_column(tmp_path: Path) -> None:
    path = tmp_path / "legacy-scout.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE scout_messages (
                chat_id INTEGER NOT NULL,
                telegram_msg_id INTEGER NOT NULL,
                sender_id INTEGER,
                sender_name TEXT,
                text TEXT,
                date TIMESTAMP NOT NULL,
                entities_json TEXT,
                forward_chat_id INTEGER,
                forward_message_id INTEGER,
                content_hash TEXT,
                source TEXT NOT NULL DEFAULT 'live',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chat_id, telegram_msg_id)
            )
            """
        )

    scout = ScoutDatabase(path)
    await scout.connect()
    try:
        columns = {
            row[1]
            for row in await (
                await scout.conn.execute("PRAGMA table_info(scout_messages)")
            ).fetchall()
        }
        assert "reply_to_message_id" in columns
    finally:
        await scout.close()
