"""Integration tests for the full pipeline: ingest → filter → dispatch."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from config.watchers import WatcherRules, get_chat_watchers, load_watchers
from pipeline.filters import RuleFilter
from storage.db import Database


@pytest.fixture
def watchers_config(tmp_path: Path) -> Path:
    """Create a test watchers config."""
    config = {
        "watchers": [
            {
                "name": "test-housing",
                "chats": [-100111],
                "rules": {
                    "keywords": ["house", "villa", "rent"],
                    "keywords_negative": ["looking for"],
                    "min_length": 10,
                },
                "alert": "immediate",
                "llm_level": 1,
            }
        ]
    }
    path = tmp_path / "watchers.yml"
    path.write_text(yaml.dump(config))
    return path


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Test database."""
    database = Database(tmp_path / "test.db")
    await database.connect()
    yield database
    await database.close()


async def test_full_pipeline_match(
    db: Database, watchers_config: Path
) -> None:
    """Message matching watcher keywords should be stored and flagged."""
    watchers = load_watchers(watchers_config)
    chat_map = get_chat_watchers(watchers)
    filters = {w.name: RuleFilter(w) for w in watchers}

    # Simulate a message in a monitored chat
    chat_id = -100111
    text = "Beautiful villa for rent near the beach"
    watcher_list = chat_map.get(chat_id, [])

    assert len(watcher_list) == 1
    watcher = watcher_list[0]

    # Store message
    msg_id = await db.store_message(
        telegram_msg_id=1,
        chat_id=chat_id,
        sender_id=42,
        sender_name="Alice",
        text=text,
        date="2026-03-04 12:00:00",
    )
    assert msg_id is not None

    # Run filter
    result = filters[watcher.name].check(text)
    assert result.passed is True
    assert result.matched_keyword == "villa"

    # Store alert
    alert_id = await db.store_alert(
        watcher_name=watcher.name,
        message_id=msg_id,
        filter_level=1,
        score=1.0,
    )
    assert alert_id > 0

    # Update stats
    await db.update_filter_stats(watcher_name=watcher.name, level_passed=1, alert_sent=True)

    # Verify stats
    cursor = await db.conn.execute(
        "SELECT messages_total, passed_level1, alerts_sent FROM filter_stats WHERE watcher_name = ?",
        (watcher.name,),
    )
    row = await cursor.fetchone()
    assert row[0] == 1
    assert row[1] == 1
    assert row[2] == 1


async def test_full_pipeline_no_match(
    db: Database, watchers_config: Path
) -> None:
    """Message not matching keywords should be stored but no alert created."""
    watchers = load_watchers(watchers_config)
    filters = {w.name: RuleFilter(w) for w in watchers}

    text = "Anyone know a good restaurant near Thong Sala?"
    result = filters["test-housing"].check(text)
    assert result.passed is False

    # Message still stored
    msg_id = await db.store_message(
        telegram_msg_id=2,
        chat_id=-100111,
        sender_id=43,
        sender_name="Bob",
        text=text,
        date="2026-03-04 12:01:00",
    )
    assert msg_id is not None

    # No alert
    cursor = await db.conn.execute("SELECT COUNT(*) FROM alerts")
    count = (await cursor.fetchone())[0]
    assert count == 0


async def test_unmonitored_chat_ignored(watchers_config: Path) -> None:
    """Messages from chats not in any watcher should be skipped."""
    watchers = load_watchers(watchers_config)
    chat_map = get_chat_watchers(watchers)

    # Chat -100999 is not monitored
    assert -100999 not in chat_map


async def test_negative_keyword_blocks_pipeline(
    db: Database, watchers_config: Path
) -> None:
    """Negative keywords should prevent alert even with positive match."""
    watchers = load_watchers(watchers_config)
    filters = {w.name: RuleFilter(w) for w in watchers}

    text = "Looking for a house to rent on Phangan"
    result = filters["test-housing"].check(text)
    assert result.passed is False
    assert result.reason == "negative_keyword"
