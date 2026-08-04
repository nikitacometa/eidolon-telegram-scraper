"""Storage for watchers the assistant authors, and how the daemon notices them."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from storage.db import Database

DEFINITION = json.dumps(
    {
        "name": "agent-danang-events",
        "chats": [],
        "rules": {"keywords": [], "min_length": 60},
        "llm_level": 3,
        "examples": {"positive": ["Концерт в баре в пятницу"], "negative": ["Сдаю квартиру"]},
    }
)


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "eidolon.db")
    await database.connect()
    yield database
    await database.close()


class TestGeneration:
    async def test_a_fresh_database_starts_at_zero(self, db: Database) -> None:
        assert await db.agent_watcher_generation() == 0

    async def test_every_write_moves_the_counter(self, db: Database) -> None:
        # This counter is the daemon's only signal that anything changed; a
        # write that does not move it is a watcher that never goes live.
        before = await db.agent_watcher_generation()
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        after = await db.agent_watcher_generation()
        assert after > before

    async def test_a_revoke_moves_it_too(self, db: Database) -> None:
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        mid = await db.agent_watcher_generation()
        assert await db.revoke_agent_watcher("agent-a") is True
        assert await db.agent_watcher_generation() > mid

    async def test_revoking_something_absent_does_not_move_it(self, db: Database) -> None:
        # Otherwise every no-op revoke wakes the daemon into a pointless reload.
        before = await db.agent_watcher_generation()
        assert await db.revoke_agent_watcher("agent-missing") is False
        assert await db.agent_watcher_generation() == before

    async def test_the_row_is_visible_to_anyone_who_saw_the_new_counter(self, db: Database) -> None:
        # The bump shares a transaction with the write precisely so a reader
        # cannot observe the counter without the row it announces.
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        generation = await db.agent_watcher_generation()
        active = await db.active_agent_watchers()
        assert generation > 0
        assert [name for name, _, _ in active] == ["agent-a"]


class TestLifecycle:
    async def test_a_saved_watcher_comes_back_intact(self, db: Database) -> None:
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        name, definition, created_by = (await db.active_agent_watchers())[0]
        assert name == "agent-a"
        assert json.loads(definition)["llm_level"] == 3
        assert created_by == "openclaw:nikita"

    async def test_saving_the_same_name_replaces_rather_than_duplicates(self, db: Database) -> None:
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        updated = json.dumps({**json.loads(DEFINITION), "llm_level": 2})
        await db.save_agent_watcher(
            name="agent-a", definition_json=updated, created_by="openclaw:nikita"
        )
        active = await db.active_agent_watchers()
        assert len(active) == 1
        assert json.loads(active[0][1])["llm_level"] == 2

    async def test_a_revoked_watcher_leaves_the_active_set(self, db: Database) -> None:
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        await db.revoke_agent_watcher("agent-a")
        assert await db.active_agent_watchers() == []

    async def test_re_saving_a_revoked_name_brings_it_back(self, db: Database) -> None:
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        await db.revoke_agent_watcher("agent-a")
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        assert [n for n, _, _ in await db.active_agent_watchers()] == ["agent-a"]

    async def test_who_asked_is_recorded(self, db: Database) -> None:
        # Julia's bridge is read-only and must never reach this method. The
        # column is what makes it visible if that ever stops being true.
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:julia"
        )
        assert (await db.active_agent_watchers())[0][2] == "openclaw:julia"


class TestNameValidation:
    @pytest.mark.parametrize("name", ["agent-events", "agent-danang-live-music", "agent-a1"])
    def test_agent_prefixed_names_are_accepted(self, name: str) -> None:
        from pipeline.agent_watchers import validate_agent_name

        assert validate_agent_name(name) == name

    @pytest.mark.parametrize(
        "name",
        ["danang-signal", "agent", "agent-", "Agent-Events", "agent_events", "../etc/passwd", ""],
    )
    def test_anything_else_is_refused(self, name: str) -> None:
        # The prefix is what keeps the two policy sources in disjoint name
        # spaces, so a merge is a union rather than a field-by-field overlay.
        from pipeline.agent_watchers import AgentWatcherError, validate_agent_name

        with pytest.raises(AgentWatcherError):
            validate_agent_name(name)


class TestDefinitionParsing:
    def _payload(self, **overrides: object) -> str:
        base: dict[str, object] = {
            "rules": {"keywords": [], "min_length": 60},
            "llm_level": 3,
            "examples": {"positive": ["Концерт в баре"], "negative": ["Сдаю квартиру"]},
        }
        base.update(overrides)
        return json.dumps(base)

    def test_a_valid_definition_parses(self) -> None:
        from pipeline.agent_watchers import parse_agent_watcher

        watcher = parse_agent_watcher("agent-events", self._payload())
        assert watcher.name == "agent-events"
        assert watcher.llm_level == 3

    def test_chats_are_forced_empty(self) -> None:
        # Which chats a policy watches is runtime state. A definition that could
        # bind itself to chats would let a malformed row choose its own scope.
        from pipeline.agent_watchers import parse_agent_watcher

        watcher = parse_agent_watcher("agent-events", self._payload(chats=[-100, -200]))
        assert watcher.chats == []

    def test_the_name_in_the_payload_cannot_override_the_row(self) -> None:
        from pipeline.agent_watchers import parse_agent_watcher

        watcher = parse_agent_watcher("agent-events", self._payload(name="danang-signal"))
        assert watcher.name == "agent-events"

    def test_a_keyword_only_watcher_is_refused(self) -> None:
        # llm_level 1 is a keyword list, which measured 56% misses on real
        # announcements in this corpus — the failure semantic watchers replace.
        from pipeline.agent_watchers import AgentWatcherError, parse_agent_watcher

        with pytest.raises(AgentWatcherError, match="llm_level"):
            parse_agent_watcher("agent-events", self._payload(llm_level=1))

    def test_malformed_json_is_refused_not_crashed(self) -> None:
        from pipeline.agent_watchers import AgentWatcherError, parse_agent_watcher

        with pytest.raises(AgentWatcherError, match="not JSON"):
            parse_agent_watcher("agent-events", "{not json")

    def test_an_unknown_field_is_refused(self) -> None:
        from pipeline.agent_watchers import AgentWatcherError, parse_agent_watcher

        with pytest.raises(AgentWatcherError):
            parse_agent_watcher("agent-events", self._payload(sudo=True))


class TestMerge:
    def _stored(self, name: str, **overrides: object) -> tuple[str, str, str]:
        payload: dict[str, object] = {
            "rules": {"keywords": [], "min_length": 60},
            "llm_level": 3,
            "examples": {"positive": ["Концерт"], "negative": ["Сдаю"]},
        }
        payload.update(overrides)
        return (name, json.dumps(payload), "openclaw:nikita")

    def test_a_new_definition_is_reported_as_added(self) -> None:
        from pipeline.agent_watchers import merge_agent_watchers

        merged, result = merge_agent_watchers(
            config_watchers=[], stored=[self._stored("agent-a")], current={}
        )
        assert list(merged) == ["agent-a"]
        assert result.added == ["agent-a"]
        assert result.changed

    def test_an_unchanged_definition_is_not_reported(self) -> None:
        # A reload that reports change every tick would reseed the semantic
        # index every tick, which is a paid API call per watcher.
        from pipeline.agent_watchers import merge_agent_watchers

        first, _ = merge_agent_watchers(
            config_watchers=[], stored=[self._stored("agent-a")], current={}
        )
        _, result = merge_agent_watchers(
            config_watchers=[], stored=[self._stored("agent-a")], current=first
        )
        assert not result.changed

    def test_an_edited_definition_is_reported_as_updated(self) -> None:
        from pipeline.agent_watchers import merge_agent_watchers

        first, _ = merge_agent_watchers(
            config_watchers=[], stored=[self._stored("agent-a")], current={}
        )
        _, result = merge_agent_watchers(
            config_watchers=[],
            stored=[self._stored("agent-a", llm_level=2)],
            current=first,
        )
        assert result.updated == ["agent-a"]

    def test_a_vanished_definition_is_reported_as_removed(self) -> None:
        from pipeline.agent_watchers import merge_agent_watchers

        first, _ = merge_agent_watchers(
            config_watchers=[], stored=[self._stored("agent-a")], current={}
        )
        merged, result = merge_agent_watchers(config_watchers=[], stored=[], current=first)
        assert merged == {}
        assert result.removed == ["agent-a"]

    def test_config_wins_a_name_collision(self, tmp_path: Path) -> None:
        from config.watchers import Watcher
        from pipeline.agent_watchers import merge_agent_watchers

        config = Watcher(name="agent-a", rules={"keywords": ["x"]}, llm_level=1)
        merged, result = merge_agent_watchers(
            config_watchers=[config], stored=[self._stored("agent-a")], current={}
        )
        assert merged == {}
        assert result.rejected == ["agent-a"]

    def test_one_bad_row_does_not_block_the_others(self) -> None:
        from pipeline.agent_watchers import merge_agent_watchers

        merged, result = merge_agent_watchers(
            config_watchers=[],
            stored=[("agent-bad", "{not json", "x"), self._stored("agent-good")],
            current={},
        )
        assert list(merged) == ["agent-good"]
        assert result.rejected == ["agent-bad"]


class TestSyncLoop:
    async def test_a_tick_with_no_change_does_not_reconcile(self) -> None:
        # Reconciling every tick would reseed each watcher's semantic index on a
        # schedule, which is a paid embedding call per watcher per tick.
        from pipeline.agent_watchers import AgentWatcherSync, ReconcileResult

        calls = {"n": 0}

        async def generation() -> int:
            return 7

        async def reconcile() -> ReconcileResult:
            calls["n"] += 1
            return ReconcileResult()

        sync = AgentWatcherSync(read_generation=generation, reconcile=reconcile)
        await sync.run_once()
        await sync.run_once()
        await sync.run_once()
        assert calls["n"] == 1

    async def test_a_moved_counter_reconciles_again(self) -> None:
        from pipeline.agent_watchers import AgentWatcherSync, ReconcileResult

        counter = {"value": 1}
        calls = {"n": 0}

        async def generation() -> int:
            return counter["value"]

        async def reconcile() -> ReconcileResult:
            calls["n"] += 1
            return ReconcileResult(added=["agent-a"])

        sync = AgentWatcherSync(read_generation=generation, reconcile=reconcile)
        await sync.run_once()
        counter["value"] = 2
        await sync.run_once()
        assert calls["n"] == 2

    async def test_a_failed_reconciliation_is_retried_not_marked_seen(self) -> None:
        # Marking the generation seen before the apply succeeded would strand the
        # change until the next unrelated edit.
        from pipeline.agent_watchers import AgentWatcherSync, ReconcileResult

        attempts = {"n": 0}

        async def generation() -> int:
            return 5

        async def reconcile() -> ReconcileResult:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("chroma unavailable")
            return ReconcileResult(added=["agent-a"])

        sync = AgentWatcherSync(read_generation=generation, reconcile=reconcile)
        with pytest.raises(RuntimeError):
            await sync.run_once()
        await sync.run_once()
        assert attempts["n"] == 2

    async def test_the_loop_survives_a_failing_tick(self) -> None:
        # The daemon owns the Telegram session; this loop must never be the
        # reason it stops.
        import asyncio

        from pipeline.agent_watchers import AgentWatcherSync, ReconcileResult

        ticks = {"n": 0}
        shutdown = asyncio.Event()

        async def generation() -> int:
            ticks["n"] += 1
            if ticks["n"] >= 3:
                shutdown.set()
            raise RuntimeError("database is locked")

        async def reconcile() -> ReconcileResult:
            return ReconcileResult()

        sync = AgentWatcherSync(read_generation=generation, reconcile=reconcile, poll_seconds=0.01)
        await asyncio.wait_for(sync.run_forever(shutdown), timeout=5)
        assert ticks["n"] >= 3


class TestDaemonReconciliation:
    """The behaviour the whole feature exists for: live, without a restart."""

    @staticmethod
    def _app(db: Database) -> object:
        from unittest.mock import AsyncMock

        from config.watchers import Watcher, WatcherRules
        from main import Eidolon
        from pipeline.filters import RuleFilter

        config = Watcher(
            name="danang-signal",
            rules=WatcherRules(keywords=[], min_length=60),
            llm_level=3,
            prompt="Announcements of events a person could attend in Da Nang.",
        )
        app = Eidolon.__new__(Eidolon)
        app.db = db
        app._config_watchers = [config]
        app._agent_watchers = {}
        app.watchers = [config]
        app.watchers_by_name = {config.name: config}
        app.watcher_fingerprints = {config.name: "fp"}
        app.chat_watchers = {}
        app.observed_chats = {}
        app.filters = {config.name: RuleFilter(config)}
        app.embedding_filter = AsyncMock()
        return app

    async def test_a_new_watcher_becomes_live_without_a_restart(self, db: Database) -> None:
        app = self._app(db)
        await db.save_agent_watcher(
            name="agent-events", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        result = await app.reconcile_agent_watchers()  # type: ignore[attr-defined]
        assert result.added == ["agent-events"]
        assert "agent-events" in app.watchers_by_name  # type: ignore[attr-defined]

    async def test_its_semantic_index_is_seeded_before_it_can_judge_anything(
        self, db: Database
    ) -> None:
        # Without this the watcher is live and blind: every message degrades at
        # Level 2 and the default policy renders that as silence, so it would
        # look enabled and never alert.
        app = self._app(db)
        await db.save_agent_watcher(
            name="agent-events", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        await app.reconcile_agent_watchers()  # type: ignore[attr-defined]
        seeded = app.embedding_filter.start.await_args.args[0]  # type: ignore[attr-defined]
        assert "agent-events" in {w.name for w in seeded}

    async def test_a_revoked_watcher_stops_being_routed(self, db: Database) -> None:
        app = self._app(db)
        await db.save_agent_watcher(
            name="agent-events", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        await app.reconcile_agent_watchers()  # type: ignore[attr-defined]
        await db.revoke_agent_watcher("agent-events")
        result = await app.reconcile_agent_watchers()  # type: ignore[attr-defined]
        assert result.removed == ["agent-events"]
        assert "agent-events" not in app.watchers_by_name  # type: ignore[attr-defined]

    async def test_an_unchanged_set_does_not_reseed(self, db: Database) -> None:
        # Reseeding is a paid embedding call per watcher; doing it on every tick
        # would turn a 30-second poll into a standing bill.
        app = self._app(db)
        await db.save_agent_watcher(
            name="agent-events", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        await app.reconcile_agent_watchers()  # type: ignore[attr-defined]
        calls = app.embedding_filter.start.await_count  # type: ignore[attr-defined]
        await app.reconcile_agent_watchers()  # type: ignore[attr-defined]
        assert app.embedding_filter.start.await_count == calls  # type: ignore[attr-defined]

    async def test_the_config_watcher_survives_every_reconciliation(self, db: Database) -> None:
        app = self._app(db)
        await db.save_agent_watcher(
            name="agent-events", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        await app.reconcile_agent_watchers()  # type: ignore[attr-defined]
        await db.revoke_agent_watcher("agent-events")
        await app.reconcile_agent_watchers()  # type: ignore[attr-defined]
        assert "danang-signal" in app.watchers_by_name  # type: ignore[attr-defined]

    async def test_the_startup_path_merges_without_reseeding(self, db: Database) -> None:
        # At startup the caller seeds and rebuilds routing itself; doing it here
        # too would embed every reference set twice on every boot.
        app = self._app(db)
        await db.save_agent_watcher(
            name="agent-events", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        result = await app.reconcile_agent_watchers(apply=False)  # type: ignore[attr-defined]
        assert result.added == ["agent-events"]
        assert "agent-events" in app.watchers_by_name  # type: ignore[attr-defined]
        app.embedding_filter.start.assert_not_awaited()  # type: ignore[attr-defined]


class TestBinding:
    """A watcher that is loaded but bound to nothing never sees a message."""

    @staticmethod
    async def _observe(db: Database, chat_id: int, mode: str = "monitor") -> None:
        await db.conn.execute(
            "INSERT INTO observed_chats (chat_id, mode, title, source) VALUES (?, ?, ?, 'manual')",
            (chat_id, mode, f"chat {chat_id}"),
        )
        await db.conn.commit()

    async def test_saving_binds_to_every_monitored_chat_by_default(self, db: Database) -> None:
        for chat_id in (-100, -200, -300):
            await self._observe(db, chat_id)
        bound = await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        assert bound == 3

    async def test_chats_not_under_monitoring_are_not_bound(self, db: Database) -> None:
        # A binding to a paused or recon chat is routing that ingestion ignores.
        await self._observe(db, -100, "monitor")
        await self._observe(db, -200, "paused")
        await self._observe(db, -300, "recon")
        assert (
            await db.save_agent_watcher(
                name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
            )
            == 1
        )

    async def test_an_explicit_subset_is_honoured(self, db: Database) -> None:
        for chat_id in (-100, -200, -300):
            await self._observe(db, chat_id)
        assert (
            await db.save_agent_watcher(
                name="agent-a",
                definition_json=DEFINITION,
                created_by="openclaw:nikita",
                chat_ids=[-100, -300],
            )
            == 2
        )

    async def test_an_unobserved_chat_id_is_silently_not_bound(self, db: Database) -> None:
        await self._observe(db, -100)
        assert (
            await db.save_agent_watcher(
                name="agent-a",
                definition_json=DEFINITION,
                created_by="openclaw:nikita",
                chat_ids=[-100, -999],
            )
            == 1
        )

    async def test_bindings_are_manual_so_a_config_reload_cannot_drop_them(
        self, db: Database
    ) -> None:
        # sync_config_bindings deletes every `config`-origin binding it no longer
        # sees declared in the YAML; a config-origin binding here would be
        # unbound on the next reload without anyone asking.
        await self._observe(db, -100)
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        async with db.conn.execute(
            "SELECT source FROM chat_policy_bindings WHERE watcher_name = 'agent-a'"
        ) as cursor:
            assert [row[0] for row in await cursor.fetchall()] == ["manual"]

    async def test_revoking_also_removes_the_routing(self, db: Database) -> None:
        await self._observe(db, -100)
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        await db.revoke_agent_watcher("agent-a")
        async with db.conn.execute(
            "SELECT count(*) FROM chat_policy_bindings WHERE watcher_name = 'agent-a'"
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0

    async def test_saving_twice_does_not_duplicate_bindings(self, db: Database) -> None:
        await self._observe(db, -100)
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        await db.save_agent_watcher(
            name="agent-a", definition_json=DEFINITION, created_by="openclaw:nikita"
        )
        async with db.conn.execute(
            "SELECT count(*) FROM chat_policy_bindings WHERE watcher_name = 'agent-a'"
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1
