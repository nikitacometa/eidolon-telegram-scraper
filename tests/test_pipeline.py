"""Offline integration tests for watcher config, rules, and persistence."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import yaml

from config.watchers import get_chat_watchers, load_watchers
from pipeline.filters import RuleFilter
from pipeline.models import PipelineOutcome
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
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Test database."""
    database = Database(tmp_path / "test.db")
    await database.connect()
    yield database
    await database.close()


async def test_rule_match_persists_alert_and_pipeline_outcome(
    db: Database, watchers_config: Path
) -> None:
    """A deterministic match should produce one alert and one aggregate outcome."""
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

    inserted = await db.record_pipeline_outcome(
        PipelineOutcome(
            message_id=msg_id,
            watcher_name=watcher.name,
            rule_passed=True,
            alert_created=True,
        )
    )
    assert inserted is True

    # Verify stats
    cursor = await db.conn.execute(
        """
        SELECT messages_total, passed_level1, passed_level2, passed_level3, alerts_sent
        FROM filter_stats
        WHERE watcher_name = ?
        """,
        (watcher.name,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == (1, 1, 0, 0, 0)


async def test_rule_rejection_persists_outcome_without_alert(
    db: Database, watchers_config: Path
) -> None:
    """A rejected message should still count once without creating an alert."""
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

    assert await db.record_pipeline_outcome(
        PipelineOutcome(
            message_id=msg_id,
            watcher_name="test-housing",
            rule_passed=False,
        )
    )

    # No alert
    cursor = await db.conn.execute("SELECT COUNT(*) FROM alerts")
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 0

    cursor = await db.conn.execute(
        """
        SELECT messages_total, passed_level1, passed_level2, passed_level3, alerts_sent
        FROM filter_stats
        WHERE watcher_name = ?
        """,
        ("test-housing",),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == (1, 0, 0, 0, 0)


async def test_unmonitored_chat_ignored(watchers_config: Path) -> None:
    """Messages from chats not in any watcher should be skipped."""
    watchers = load_watchers(watchers_config)
    chat_map = get_chat_watchers(watchers)

    # Chat -100999 is not monitored
    assert -100999 not in chat_map


async def test_negative_keyword_blocks_pipeline(db: Database, watchers_config: Path) -> None:
    """Negative keywords should prevent alert even with positive match."""
    watchers = load_watchers(watchers_config)
    filters = {w.name: RuleFilter(w) for w in watchers}

    text = "Looking for a house to rent on Phangan"
    result = filters["test-housing"].check(text)
    assert result.passed is False
    assert result.reason == "negative_keyword"
