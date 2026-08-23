"""The bridge's write tools: requests to the daemon, never actions on Telegram."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from eidolon_mcp import READ_TOOLS, WRITE_TOOLS, EidolonTools
from storage.search import SearchDatabase

LIVE_SCHEMA = """
CREATE TABLE chats (chat_id INTEGER PRIMARY KEY, title TEXT);
CREATE TABLE observed_chats (
    chat_id INTEGER PRIMARY KEY, mode TEXT NOT NULL DEFAULT 'monitor', title TEXT,
    source TEXT NOT NULL DEFAULT 'config', job_id TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE chat_policy_bindings (
    chat_id INTEGER NOT NULL, watcher_name TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'config', job_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, watcher_name));
CREATE TABLE agent_watchers_meta (id INTEGER PRIMARY KEY CHECK (id = 1), generation INTEGER NOT NULL DEFAULT 0);
INSERT INTO agent_watchers_meta (id, generation) VALUES (1, 0);
CREATE TABLE agent_watchers (name TEXT PRIMARY KEY, status TEXT NOT NULL);
INSERT INTO agent_watchers VALUES ('agent-live-music', 'active');
INSERT INTO agent_watchers VALUES ('agent-retired', 'revoked');
"""

SCOUT_SCHEMA = """
CREATE TABLE backfill_targets (
    chat_id INTEGER PRIMARY KEY, label TEXT, target_days INTEGER NOT NULL DEFAULT 730,
    oldest_message_id INTEGER, oldest_message_date TIMESTAMP,
    messages_stored INTEGER NOT NULL DEFAULT 0, pages_done INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'pending', last_error TEXT, not_before TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE join_queue (chat_ref TEXT PRIMARY KEY, label TEXT, watcher_name TEXT,
    target_days INTEGER, state TEXT NOT NULL DEFAULT 'pending', joined_chat_id INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
"""


@pytest.fixture(autouse=True)
def workers_running(monkeypatch: Any) -> None:
    """Most tests are about tool logic, not about whether a worker exists."""
    import config.settings as cs

    monkeypatch.setattr(cs.settings, "backfill_enabled", True)
    monkeypatch.setattr(cs.settings, "join_queue_enabled", True)


@pytest.fixture
def tools(tmp_path: Path) -> EidolonTools:
    live, scout, search = tmp_path / "live.db", tmp_path / "scout.db", tmp_path / "search.db"
    with sqlite3.connect(live) as conn:
        conn.executescript(LIVE_SCHEMA)
        conn.execute("INSERT INTO chats VALUES (-100, 'Близкие на Пангане')")
        conn.execute(
            "INSERT INTO observed_chats VALUES (-200, 'monitor', 'Дананг', 'config', NULL, NULL, NULL)"
        )
        conn.execute(
            "INSERT INTO chat_policy_bindings VALUES (-200, 'danang-signal', 'config', NULL, NULL)"
        )
    with sqlite3.connect(scout) as conn:
        conn.executescript(SCOUT_SCHEMA)
    SearchDatabase(search).connect()
    return EidolonTools(search_db=search, scout_db=scout, live_db=live, writable=True)


def _generation(tools: EidolonTools) -> int:
    with sqlite3.connect(tools._live_db) as conn:
        return int(conn.execute("SELECT generation FROM agent_watchers_meta").fetchone()[0])


class TestResumeChat:
    async def test_an_unobserved_chat_starts_being_monitored(self, tools: EidolonTools) -> None:
        # The gap this closes: a chat the account is a member of, whose messages
        # Telegram delivers and the daemon discards, with nothing in the log.
        result = await tools.resume_chat(-100)
        assert result["mode"] == "monitor"
        with sqlite3.connect(tools._live_db) as conn:
            row = conn.execute(
                "SELECT mode, title FROM observed_chats WHERE chat_id=-100"
            ).fetchone()
        assert row == ("monitor", "Близкие на Пангане")

    async def test_it_is_bound_to_the_same_watchers_as_its_neighbours(
        self, tools: EidolonTools
    ) -> None:
        # A resumed chat bound to nothing is observed and still silent.
        result = await tools.resume_chat(-100)
        assert result["watchers"] == ["danang-signal"]
        with sqlite3.connect(tools._live_db) as conn:
            bound = conn.execute(
                "SELECT watcher_name FROM chat_policy_bindings WHERE chat_id=-100"
            ).fetchall()
        assert bound == [("danang-signal",)]

    async def test_the_daemon_is_told_to_reload(self, tools: EidolonTools) -> None:
        # Routing lives in the daemon's memory; a row nobody reloads is a change
        # that did not happen.
        before = _generation(tools)
        await tools.resume_chat(-100)
        assert _generation(tools) > before

    async def test_pausing_does_not_bind_anything(self, tools: EidolonTools) -> None:
        result = await tools.resume_chat(-100, mode="paused")
        assert result["watchers"] == []
        with sqlite3.connect(tools._live_db) as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM chat_policy_bindings WHERE chat_id=-100"
                ).fetchone()[0]
                == 0
            )

    async def test_an_unknown_mode_is_refused(self, tools: EidolonTools) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            await tools.resume_chat(-100, mode="watch-really-hard")

    async def test_a_readonly_bridge_refuses(self, tmp_path: Path, tools: EidolonTools) -> None:
        readonly = EidolonTools(
            search_db=tools._search_db,
            scout_db=tools._scout_db,
            live_db=tools._live_db,
            writable=False,
        )
        with pytest.raises(PermissionError):
            await readonly.resume_chat(-100)


class TestBackfillChat:
    async def test_a_target_is_created(self, tools: EidolonTools) -> None:
        result = await tools.backfill_chat(-100, days=365)
        assert result["target_days"] == 365
        assert result["previous"] is None
        with sqlite3.connect(tools._scout_db) as conn:
            row = conn.execute(
                "SELECT target_days, state FROM backfill_targets WHERE chat_id=-100"
            ).fetchone()
        assert row == (365, "pending")

    async def test_asking_again_for_less_does_nothing(self, tools: EidolonTools) -> None:
        # Otherwise a second ask restarts a finished walk and re-spends the
        # account's action budget on history already stored.
        await tools.backfill_chat(-100, days=730)
        with sqlite3.connect(tools._scout_db) as conn:
            conn.execute("UPDATE backfill_targets SET state='complete' WHERE chat_id=-100")
        await tools.backfill_chat(-100, days=90)
        with sqlite3.connect(tools._scout_db) as conn:
            row = conn.execute(
                "SELECT target_days, state FROM backfill_targets WHERE chat_id=-100"
            ).fetchone()
        assert row == (730, "complete")

    async def test_asking_for_more_reopens_a_finished_walk(self, tools: EidolonTools) -> None:
        await tools.backfill_chat(-100, days=90)
        with sqlite3.connect(tools._scout_db) as conn:
            conn.execute("UPDATE backfill_targets SET state='complete' WHERE chat_id=-100")
        await tools.backfill_chat(-100, days=730)
        with sqlite3.connect(tools._scout_db) as conn:
            row = conn.execute(
                "SELECT target_days, state FROM backfill_targets WHERE chat_id=-100"
            ).fetchone()
        assert row == (730, "pending")

    @pytest.mark.parametrize("days", [0, -1, 4000])
    async def test_an_absurd_window_is_refused(self, tools: EidolonTools, days: int) -> None:
        with pytest.raises(ValueError, match="days must be"):
            await tools.backfill_chat(-100, days=days)

    async def test_a_readonly_bridge_refuses(self, tools: EidolonTools) -> None:
        readonly = EidolonTools(
            search_db=tools._search_db,
            scout_db=tools._scout_db,
            live_db=tools._live_db,
            writable=False,
        )
        with pytest.raises(PermissionError):
            await readonly.backfill_chat(-100)


class TestToolExposure:
    def test_the_write_tools_are_not_in_the_read_only_catalogue(self) -> None:
        # Julia's bridge registers READ_TOOLS only; a write tool leaking there
        # would spend Nikita's Telegram action budget from her instance.
        read_names = {tool.name for tool in READ_TOOLS}
        assert "resume_chat" not in read_names
        assert "backfill_chat" not in read_names

    def test_every_write_tool_is_accounted_for(self) -> None:
        assert {tool.name for tool in WRITE_TOOLS} == {
            "queue_chat_join",
            "resume_chat",
            "backfill_chat",
        }

    @pytest.mark.parametrize("tool", WRITE_TOOLS + READ_TOOLS, ids=lambda t: t.name)
    def test_optional_arguments_accept_null(self, tool: Any) -> None:
        # A gpt-5.6 agent routinely passes null for arguments it is not using.
        # An optional field typed as a bare scalar rejects that and wastes a turn
        # on a validation error the model cannot read its way out of.
        schema = tool.inputSchema
        required = set(schema.get("required", []))
        for name, spec in schema.get("properties", {}).items():
            if name in required:
                continue
            allowed = _declared_types(spec)
            if not allowed:
                continue
            assert "null" in allowed, f"{tool.name}.{name} is optional but rejects null"

    @pytest.mark.parametrize("tool", WRITE_TOOLS + READ_TOOLS, ids=lambda t: t.name)
    def test_a_nullable_array_is_a_union_of_branches(self, tool: Any) -> None:
        # OpenClaw validates arguments with TypeBox 1.3.3, whose code for
        # {"type": ["array", "null"], "items": ...} is
        #   (Array.isArray(x) || x === null) && x.every(...)
        # so a null argument throws instead of validating. Measured 2026-08-17:
        # three of four search_messages calls from the bot died that way. An
        # anyOf compiles to separately guarded branches and survives null.
        for name, spec in tool.inputSchema.get("properties", {}).items():
            if "items" in spec:
                declared = spec.get("type")
                assert not isinstance(declared, list), (
                    f"{tool.name}.{name}: typed union with items breaks under TypeBox; use anyOf"
                )


def _declared_types(spec: dict[str, Any]) -> list[str]:
    declared = spec.get("type")
    if declared is not None:
        return declared if isinstance(declared, list) else [declared]
    return [t for branch in spec.get("anyOf", []) for t in _declared_types(branch)]


class TestNullArgumentHandling:
    """The schemas accept null; the dispatch must make it mean 'not supplied'."""

    def test_nulls_are_dropped(self) -> None:
        from eidolon_mcp import supplied_arguments

        assert supplied_arguments({"chat_id": -100, "days": None, "label": None}) == {
            "chat_id": -100
        }

    def test_falsy_values_that_are_not_null_survive(self) -> None:
        # 0 and "" are answers, not absences; dropping them would silently
        # override a caller who meant them.
        from eidolon_mcp import supplied_arguments

        assert supplied_arguments({"limit": 0, "name": "", "semantic": False}) == {
            "limit": 0,
            "name": "",
            "semantic": False,
        }

    def test_no_arguments_at_all_is_an_empty_mapping(self) -> None:
        from eidolon_mcp import supplied_arguments

        assert supplied_arguments(None) == {}

    async def test_a_null_optional_reaches_the_handler_as_a_default(
        self, tools: EidolonTools
    ) -> None:
        from eidolon_mcp import supplied_arguments

        result = await tools.backfill_chat(
            **supplied_arguments({"chat_id": -100, "days": None, "label": None})
        )
        assert result["target_days"] == 730

    async def test_a_null_limit_does_not_break_a_search(self, tools: EidolonTools) -> None:
        from eidolon_mcp import supplied_arguments

        result = await tools.search_messages(
            **supplied_arguments({"query": "концерт", "limit": None, "days": None})
        )
        assert result["result_count"] == 0  # empty index, but it answered


class TestDisabledWorkers:
    """A request nothing will execute must fail loudly, not report success."""

    async def test_backfill_is_refused_when_no_worker_runs(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        # main.py only starts BackfillWorker when the setting is on; otherwise
        # the row sits pending forever while the caller is told history is
        # downloading.
        import config.settings as cs

        monkeypatch.setattr(cs.settings, "backfill_enabled", False)
        with pytest.raises(RuntimeError, match="BACKFILL_ENABLED"):
            await tools.backfill_chat(-100)
        with sqlite3.connect(tools._scout_db) as conn:
            assert conn.execute("SELECT count(*) FROM backfill_targets").fetchone()[0] == 0

    async def test_joining_is_refused_when_no_worker_runs(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        import config.settings as cs

        monkeypatch.setattr(cs.settings, "join_queue_enabled", False)
        with pytest.raises(RuntimeError, match="JOIN_QUEUE_ENABLED"):
            await tools.queue_chat_join("somechat")
        with sqlite3.connect(tools._scout_db) as conn:
            assert conn.execute("SELECT count(*) FROM join_queue").fetchone()[0] == 0

    async def test_they_work_when_the_workers_are_running(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        import config.settings as cs

        monkeypatch.setattr(cs.settings, "backfill_enabled", True)
        monkeypatch.setattr(cs.settings, "join_queue_enabled", True)
        assert (await tools.backfill_chat(-100))["target_days"] == 730
        assert (await tools.queue_chat_join("somechat"))["queued"] is True


class TestQueueChatJoin:
    async def test_a_policy_makes_the_join_monitored(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(tools, "known_watcher_names", lambda: ["danang-signal"])

        result = await tools.queue_chat_join("@DalatChat", watcher_name="danang-signal")

        assert result["queued"] is True
        assert "danang-signal" in result["monitoring"]
        with sqlite3.connect(tools._scout_db) as conn:
            row = conn.execute("SELECT chat_ref, watcher_name FROM join_queue").fetchone()
        assert row == ("dalatchat", "danang-signal")

    async def test_no_policy_is_said_to_be_silent(self, tools: EidolonTools) -> None:
        # The old reply claimed the chat "is monitored"; four chats were archived
        # for weeks with no alerts on the strength of that sentence.
        result = await tools.queue_chat_join("@DalatChat")

        assert result["watcher_name"] is None
        assert "no alerts" in result["monitoring"]
        with sqlite3.connect(tools._scout_db) as conn:
            assert conn.execute("SELECT watcher_name FROM join_queue").fetchone() == (None,)

    async def test_an_unknown_policy_is_refused_before_anything_is_queued(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(tools, "known_watcher_names", lambda: ["danang-signal"])

        with pytest.raises(ValueError, match="unknown watcher"):
            await tools.queue_chat_join("@DalatChat", watcher_name="dalat-signa")
        with sqlite3.connect(tools._scout_db) as conn:
            assert conn.execute("SELECT count(*) FROM join_queue").fetchone()[0] == 0

    async def test_an_invite_link_keeps_its_hash_case(self, tools: EidolonTools) -> None:
        result = await tools.queue_chat_join("https://t.me/+mqaI5aYDQuI5ZWFi")

        assert result["chat_ref"] == "+mqaI5aYDQuI5ZWFi"

    def test_known_policies_come_from_config_and_stored_watchers(
        self, tools: EidolonTools, monkeypatch: Any, tmp_path: Path
    ) -> None:
        import config.settings as cs

        config = tmp_path / "watchers.yml"
        config.write_text(
            "watchers:\n  - name: dalat-signal\n    rules:\n      keywords: [концерт]\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(cs.settings, "watchers_path", config)

        assert tools.known_watcher_names() == ["dalat-signal", "agent-live-music"]

    def test_a_missing_config_still_knows_stored_watchers(
        self, tools: EidolonTools, monkeypatch: Any, tmp_path: Path
    ) -> None:
        import config.settings as cs

        monkeypatch.setattr(cs.settings, "watchers_path", tmp_path / "absent.yml")

        assert tools.known_watcher_names() == ["agent-live-music"]


class TestUpgradingAQueuedJoin:
    async def test_a_pending_row_takes_the_policy(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(tools, "known_watcher_names", lambda: ["danang-signal"])
        await tools.queue_chat_join("@DalatChat")

        result = await tools.queue_chat_join("@DalatChat", watcher_name="danang-signal")

        assert result["queued"] is False
        assert result["watcher_name"] == "danang-signal"
        with sqlite3.connect(tools._scout_db) as conn:
            assert conn.execute("SELECT watcher_name FROM join_queue").fetchone() == (
                "danang-signal",
            )

    async def test_a_joined_chat_is_bound_to_that_policy_alone(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        # The live DB already carries a danang-signal binding; an explicit
        # policy must not drag it along the way the neighbour default does.
        monkeypatch.setattr(tools, "known_watcher_names", lambda: ["agent-live-music"])
        with sqlite3.connect(tools._scout_db) as conn:
            conn.execute(
                "INSERT INTO join_queue (chat_ref, state, joined_chat_id) "
                "VALUES ('dalatchat', 'joined', -300)"
            )
            conn.commit()

        result = await tools.queue_chat_join("dalatchat", watcher_name="agent-live-music")

        assert result["already"] == "joined"
        assert result["watcher_name"] == "agent-live-music"
        with sqlite3.connect(tools._live_db) as conn:
            bindings = conn.execute(
                "SELECT watcher_name FROM chat_policy_bindings WHERE chat_id = -300"
            ).fetchall()
            mode = conn.execute("SELECT mode FROM observed_chats WHERE chat_id = -300").fetchone()
        assert bindings == [("agent-live-music",)]
        assert mode == ("monitor",)
        with sqlite3.connect(tools._scout_db) as conn:
            assert conn.execute(
                "SELECT watcher_name FROM join_queue WHERE chat_ref = 'dalatchat'"
            ).fetchone() == ("agent-live-music",)

    async def test_an_explicit_policy_keeps_the_bindings_a_chat_already_has(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        # -200 is bound to danang-signal in the fixture. Adding a second policy
        # by name must not be the call that drops the first.
        monkeypatch.setattr(tools, "known_watcher_names", lambda: ["agent-live-music"])

        await tools.resume_chat(-200, watcher_name="agent-live-music")

        with sqlite3.connect(tools._live_db) as conn:
            bindings = conn.execute(
                "SELECT watcher_name FROM chat_policy_bindings WHERE chat_id = -200 ORDER BY 1"
            ).fetchall()
        assert bindings == [("agent-live-music",), ("danang-signal",)]

    async def test_a_repeat_without_a_policy_changes_nothing(self, tools: EidolonTools) -> None:
        await tools.queue_chat_join("@DalatChat")

        result = await tools.queue_chat_join("@DalatChat")

        assert result == {
            "chat_ref": "dalatchat",
            "queued": False,
            "already": "pending",
            "joined_chat_id": None,
            "watcher_name": None,
        }


class TestResumeWithAPolicy:
    async def test_naming_a_policy_binds_only_it(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(tools, "known_watcher_names", lambda: ["agent-live-music"])

        result = await tools.resume_chat(-100, watcher_name="agent-live-music")

        assert result["watchers"] == ["agent-live-music"]
        with sqlite3.connect(tools._live_db) as conn:
            bindings = conn.execute(
                "SELECT watcher_name FROM chat_policy_bindings WHERE chat_id = -100"
            ).fetchall()
        assert bindings == [("agent-live-music",)]

    async def test_a_policy_with_a_non_monitor_mode_is_refused(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(tools, "known_watcher_names", lambda: ["agent-live-music"])
        with pytest.raises(ValueError, match="monitor"):
            await tools.resume_chat(-100, mode="recon", watcher_name="agent-live-music")
