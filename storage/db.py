"""SQLite database wrapper with async access and auto-migration."""

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """Async SQLite wrapper for Eidolon message and alert storage."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the database connection and run migrations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._migrate()
        logger.info("Database connected: %s", self.db_path)

    async def _migrate(self) -> None:
        """Run schema.sql to create tables if they don't exist, then apply incremental migrations."""
        schema = SCHEMA_PATH.read_text()
        await self._conn.executescript(schema)
        await self._conn.commit()
        await self._apply_migrations()

    async def _apply_migrations(self) -> None:
        """Apply incremental schema changes for existing databases."""
        # Add chat_title column to messages if missing
        cursor = await self._conn.execute("PRAGMA table_info(messages)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "chat_title" not in columns:
            await self._conn.execute("ALTER TABLE messages ADD COLUMN chat_title TEXT")
            await self._conn.commit()
            logger.info("Migration: added chat_title column to messages")

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
                (telegram_msg_id, chat_id, chat_title, sender_id, sender_name, text, date, raw_json),
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
        """Store an alert and return its row ID."""
        cursor = await self.conn.execute(
            """
            INSERT INTO alerts (watcher_name, message_id, filter_level, score, llm_response)
            VALUES (?, ?, ?, ?, ?)
            """,
            (watcher_name, message_id, filter_level, score, llm_response),
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def mark_alert_sent(self, alert_id: int) -> None:
        """Mark an alert as sent (set sent_at timestamp)."""
        await self.conn.execute(
            "UPDATE alerts SET sent_at = CURRENT_TIMESTAMP WHERE id = ?",
            (alert_id,),
        )
        await self.conn.commit()

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

    async def update_filter_stats(
        self,
        *,
        watcher_name: str,
        level_passed: int | None = None,
        alert_sent: bool = False,
    ) -> None:
        """Increment filter stats for today."""
        await self.conn.execute(
            """
            INSERT INTO filter_stats (watcher_name, date, messages_total)
            VALUES (?, DATE('now'), 1)
            ON CONFLICT(watcher_name, date) DO UPDATE SET
                messages_total = filter_stats.messages_total + 1
            """,
            (watcher_name,),
        )
        if level_passed is not None:
            col = f"passed_level{level_passed}"
            await self.conn.execute(
                f"UPDATE filter_stats SET {col} = {col} + 1 "  # noqa: S608
                "WHERE watcher_name = ? AND date = DATE('now')",
                (watcher_name,),
            )
        if alert_sent:
            await self.conn.execute(
                "UPDATE filter_stats SET alerts_sent = alerts_sent + 1 "
                "WHERE watcher_name = ? AND date = DATE('now')",
                (watcher_name,),
            )
        await self.conn.commit()

    async def get_daily_messages(
        self,
        chat_ids: list[int],
        date: str,
    ) -> list[dict]:
        """Fetch all messages for given chats on a specific date (YYYY-MM-DD).

        Returns list of dicts with chat_title, sender_name, text, date.
        """
        if not chat_ids:
            return []
        placeholders = ",".join("?" * len(chat_ids))
        cursor = await self.conn.execute(
            f"SELECT chat_id, chat_title, sender_name, text, date FROM messages "  # noqa: S608
            f"WHERE chat_id IN ({placeholders}) AND DATE(date) = ? AND text IS NOT NULL "
            f"ORDER BY date ASC",
            (*chat_ids, date),
        )
        rows = await cursor.fetchall()
        return [
            {
                "chat_id": row[0],
                "chat_title": row[1] or "Unknown",
                "sender_name": row[2] or "Unknown",
                "text": row[3],
                "date": row[4],
            }
            for row in rows
        ]
