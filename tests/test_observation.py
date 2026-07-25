"""Tests for the chat observation registry and ingest routing."""

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from config.watchers import Watcher, WatcherRules
from main import Eidolon
from pipeline.filters import RuleFilter
from pipeline.models import (
    ObservationMode,
    ObservationSource,
    ObservedChat,
)
from pipeline.processor import MessageProcessor
from storage.db import Database
from storage.scout import ScoutDatabase


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Open a live monitoring database in a temporary directory."""
    database = Database(tmp_path / "eidolon.db")
    await database.connect()
    yield database
    await database.close()


def _watcher(name: str, chats: list[int]) -> Watcher:
    return Watcher(name=name, chats=chats, rules=WatcherRules(keywords=["villa"]))


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


async def test_config_chats_are_reconciled_into_the_registry(db: Database) -> None:
    """A policy file chat becomes an observed, monitored chat."""
    await db.sync_config_bindings([_watcher("housing", [-100123, -100456])])

    snapshot = await db.observation_snapshot()

    assert set(snapshot) == {-100123, -100456}
    assert snapshot[-100123].mode is ObservationMode.MONITOR
    assert snapshot[-100123].source is ObservationSource.CONFIG
    assert snapshot[-100123].watcher_names == ("housing",)


async def test_one_chat_can_carry_several_policies(db: Database) -> None:
    """Two policies watching one chat both appear in its bindings."""
    await db.sync_config_bindings([_watcher("housing", [-100123]), _watcher("scooters", [-100123])])

    snapshot = await db.observation_snapshot()

    assert snapshot[-100123].watcher_names == ("housing", "scooters")


async def test_removing_a_chat_from_config_stops_observation(db: Database) -> None:
    """The policy file stays authoritative for what it declares."""
    await db.sync_config_bindings([_watcher("housing", [-100123, -100456])])

    await db.sync_config_bindings([_watcher("housing", [-100123])])

    snapshot = await db.observation_snapshot()
    assert set(snapshot) == {-100123}


async def test_reconciliation_does_not_unpromote_a_discovered_chat(db: Database) -> None:
    """A chat promoted by reconnaissance must survive the next deploy.

    Config reconciliation owns only what the file declares. Without this rule
    every restart would quietly discard everything the crawl had promoted.
    """
    await db.sync_config_bindings([_watcher("housing", [-100123])])
    await db.observe_chat(
        chat_id=-100999,
        mode=ObservationMode.MONITOR,
        source=ObservationSource.RECON,
        job_id="job-1",
    )
    assert await db.bind_policy(chat_id=-100999, watcher_name="housing", job_id="job-1")

    await db.sync_config_bindings([_watcher("housing", [-100123])])

    snapshot = await db.observation_snapshot()
    assert set(snapshot) == {-100123, -100999}
    assert snapshot[-100999].watcher_names == ("housing",)
    assert snapshot[-100999].job_id == "job-1"


async def test_binding_requires_an_observed_chat(db: Database) -> None:
    """A binding without a registry entry would be invisible to ingestion."""
    assert not await db.bind_policy(chat_id=-100777, watcher_name="housing")


async def test_promotion_is_a_mode_change(db: Database) -> None:
    """Moving a crawled chat into monitoring must not need a config rewrite."""
    await db.observe_chat(
        chat_id=-100999,
        mode=ObservationMode.RECON,
        source=ObservationSource.RECON,
        title="Da Nang Housing",
    )

    await db.observe_chat(
        chat_id=-100999,
        mode=ObservationMode.MONITOR,
        source=ObservationSource.RECON,
    )
    await db.bind_policy(chat_id=-100999, watcher_name="housing")

    snapshot = await db.observation_snapshot()
    assert snapshot[-100999].mode is ObservationMode.MONITOR
    assert snapshot[-100999].title == "Da Nang Housing"


async def test_chat_without_policies_is_still_observed(db: Database) -> None:
    """A chat under reconnaissance has no policy yet and must still be tracked."""
    await db.observe_chat(
        chat_id=-100999,
        mode=ObservationMode.RECON,
        source=ObservationSource.RECON,
    )

    snapshot = await db.observation_snapshot()

    assert snapshot[-100999].watcher_names == ()
    assert snapshot[-100999].mode is ObservationMode.RECON


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------


def _app_with_watchers(*watchers: Watcher) -> Eidolon:
    """Build an Eidolon shell wired like __init__ leaves it."""
    app = Eidolon.__new__(Eidolon)
    app.watchers = list(watchers)
    app.watchers_by_name = {watcher.name: watcher for watcher in watchers}
    app.chat_watchers = {}
    app.observed_chats = {}
    app.filters = {watcher.name: RuleFilter(watcher) for watcher in watchers}
    return app


async def test_reload_routes_only_monitored_chats_with_policies() -> None:
    """Reconnaissance and paused chats must not reach the alerting pipeline."""
    housing = _watcher("housing", [])
    app = _app_with_watchers(housing)
    app.db = AsyncMock()
    app.db.observation_snapshot.return_value = {
        -1: ObservedChat(-1, ObservationMode.MONITOR, ObservationSource.CONFIG, ("housing",)),
        -2: ObservedChat(-2, ObservationMode.RECON, ObservationSource.RECON),
        -3: ObservedChat(-3, ObservationMode.PAUSED, ObservationSource.MANUAL, ("housing",)),
        -4: ObservedChat(-4, ObservationMode.MONITOR, ObservationSource.RECON),
    }

    await app.reload_observation()

    assert set(app.chat_watchers) == {-1}
    assert set(app.observed_chats) == {-1, -2, -3, -4}


async def test_reload_mutates_the_mapping_the_processor_holds() -> None:
    """The processor was handed the filter mapping by reference.

    Rebinding the attribute instead of mutating it would leave the processor
    using the previous mapping, so a newly promoted chat would raise KeyError
    on its first message.
    """
    housing = _watcher("housing", [])
    app = _app_with_watchers(housing)
    processor = MessageProcessor(
        store=AsyncMock(),
        rule_filters=app.filters,
        embedding_filter=AsyncMock(),
        llm_classifier=AsyncMock(),
    )
    app.db = AsyncMock()
    app.db.observation_snapshot.return_value = {}
    scooters = _watcher("scooters", [])
    app.watchers.append(scooters)
    app.watchers_by_name[scooters.name] = scooters

    await app.reload_observation()

    assert "scooters" in processor._rule_filters
    assert processor._rule_filters is app.filters


async def test_unregistered_chat_is_ignored() -> None:
    """The account sits in many chats that are none of our business."""
    app = _app_with_watchers(_watcher("housing", []))
    app.observed_chats = {}

    assert await app._ingest_update(SimpleNamespace(chat_id=-100123, text="hi")) is None


async def test_paused_chat_is_ignored() -> None:
    """A paused chat stays in the registry but produces no work."""
    app = _app_with_watchers(_watcher("housing", []))
    app.observed_chats = {
        -100123: ObservedChat(
            -100123, ObservationMode.PAUSED, ObservationSource.MANUAL, ("housing",)
        )
    }

    assert await app._ingest_update(SimpleNamespace(chat_id=-100123, text="hi")) is None


async def test_recon_chat_is_captured_without_entering_the_pipeline(
    tmp_path: Path,
) -> None:
    """Messages arriving before promotion must be kept, not dropped.

    This is the window backfill cannot recover on its own: whatever is posted
    between joining a chat and promoting it exists only as a live update.
    """
    app = _app_with_watchers(_watcher("housing", []))
    app.observed_chats = {
        -100999: ObservedChat(-100999, ObservationMode.RECON, ObservationSource.RECON)
    }
    scout = ScoutDatabase(tmp_path / "scout.db")
    await scout.connect()
    app.scout = scout
    event = SimpleNamespace(
        chat_id=-100999,
        text="Villa for rent in Da Nang",
        message=SimpleNamespace(
            id=555,
            date="2026-07-25 10:00:00",
            sender_id=42,
            fwd_from=None,
        ),
    )

    work_item = await app._ingest_update(event)

    assert work_item is None
    assert await scout.message_count(-100999) == 1
    await scout.close()


async def test_live_capture_and_backfill_do_not_duplicate(tmp_path: Path) -> None:
    """The seam between live capture and backfill is closed by the key."""
    from pipeline.recon_models import ScoutMessage

    scout = ScoutDatabase(tmp_path / "scout.db")
    await scout.connect()

    assert await scout.store_message(
        ScoutMessage(chat_id=-1, telegram_msg_id=7, date="2026-07-25", text="hello", source="live")
    )
    assert not await scout.store_message(
        ScoutMessage(
            chat_id=-1, telegram_msg_id=7, date="2026-07-25", text="hello", source="backfill"
        )
    )

    assert await scout.message_count(-1) == 1
    await scout.close()


async def test_backfill_page_is_stored_in_one_transaction(tmp_path: Path) -> None:
    """A page is the unit a crawl can resume from."""
    from pipeline.recon_models import ScoutMessage

    scout = ScoutDatabase(tmp_path / "scout.db")
    await scout.connect()
    page = [
        ScoutMessage(
            chat_id=-1,
            telegram_msg_id=index,
            date="2026-07-25",
            text=f"message {index}",
            source="backfill",
        )
        for index in range(1, 101)
    ]

    stored = await scout.store_messages(page)
    replayed = await scout.store_messages(page)

    assert stored == 100
    assert replayed == 0
    assert await scout.message_count(-1) == 100
    await scout.close()
