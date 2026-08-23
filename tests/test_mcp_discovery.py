"""The bridge's discovery tools: a job for the daemon, and reading what it found."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from eidolon_mcp import READ_TOOLS, WRITE_TOOLS, EidolonTools
from storage.search import SearchDatabase

LIVE_SCHEMA = """
CREATE TABLE chats (chat_id INTEGER PRIMARY KEY, title TEXT);
CREATE TABLE observed_chats (chat_id INTEGER PRIMARY KEY, mode TEXT, title TEXT, source TEXT);
INSERT INTO observed_chats VALUES (-1001791474778, 'monitor', 'ДАЛАТ ЧАТ', 'recon');
"""


@pytest.fixture(autouse=True)
def discovery_on(monkeypatch: Any) -> None:
    import config.settings as cs

    monkeypatch.setattr(cs.settings, "discovery_enabled", True)


@pytest.fixture
def tools(tmp_path: Path) -> EidolonTools:
    live, scout, search = tmp_path / "live.db", tmp_path / "scout.db", tmp_path / "search.db"
    with sqlite3.connect(live) as conn:
        conn.executescript(LIVE_SCHEMA)
    with sqlite3.connect(scout) as conn:
        conn.executescript(Path("storage/scout_schema.sql").read_text(encoding="utf-8"))
    SearchDatabase(search).connect()
    return EidolonTools(search_db=search, scout_db=scout, live_db=live, writable=True)


def _job_rows(tools: EidolonTools) -> list[tuple[Any, ...]]:
    with sqlite3.connect(tools._scout_db) as conn:
        return conn.execute(
            "SELECT topic, location, seeds, status, max_join_attempts, max_waves FROM recon_jobs"
        ).fetchall()


class TestDiscoverChats:
    async def test_a_job_is_queued_without_any_joins(self, tools: EidolonTools) -> None:
        result = await tools.discover_chats(
            topic="expats community", location="Da Lat", seeds=["@Dalat_Vietnam"]
        )

        assert result["queued"] is True
        assert _job_rows(tools) == [
            ("expats community", "Da Lat", '["dalat_vietnam"]', "queued", 0, 1)
        ]

    async def test_the_same_ask_on_the_same_day_is_one_job(self, tools: EidolonTools) -> None:
        first = await tools.discover_chats(topic="Expats", location="da lat")
        second = await tools.discover_chats(topic="expats", location="Da Lat")

        assert second["queued"] is False
        assert second["job_id"] == first["job_id"]
        assert len(_job_rows(tools)) == 1

    async def test_it_is_refused_when_no_worker_runs(
        self, tools: EidolonTools, monkeypatch: Any
    ) -> None:
        import config.settings as cs

        monkeypatch.setattr(cs.settings, "discovery_enabled", False)
        with pytest.raises(RuntimeError, match="DISCOVERY_ENABLED"):
            await tools.discover_chats(topic="expats", location="Da Lat")
        assert _job_rows(tools) == []

    async def test_a_readonly_bridge_refuses(self, tools: EidolonTools) -> None:
        tools._writable = False
        with pytest.raises(PermissionError):
            await tools.discover_chats(topic="expats")

    async def test_an_empty_topic_is_refused(self, tools: EidolonTools) -> None:
        with pytest.raises(ValueError):
            await tools.discover_chats(topic="   ")


class TestDiscoveryResults:
    def _seed(self, tools: EidolonTools) -> str:
        with sqlite3.connect(tools._scout_db) as conn:
            conn.execute(
                "INSERT INTO recon_jobs (id, idempotency_key, topic, location, status, "
                "stop_reason, max_join_attempts) VALUES ('j1', 'k', 'expats', 'Da Lat', "
                "'completed', 'frontier empty', 0)"
            )
            conn.executemany(
                "INSERT INTO scout_chats (id, telegram_chat_id, username, title, chat_type, "
                "participants, visibility) VALUES (?, ?, ?, ?, 'supergroup', ?, 'public')",
                [
                    ("c1", -1001791474778, "Dalat_Vietnam", "ДАЛАТ ЧАТ", 4406),
                    ("c2", None, "dalat_ru", "Далат объявления", 457),
                    ("c3", None, "scamcoin", "DALAT CRYPTO PUMP", 90000),
                ],
            )
            conn.executemany(
                "INSERT INTO job_candidates (job_id, chat_uuid, state, policy_score, "
                "independent_sources, risk_flags) VALUES ('j1', ?, ?, ?, ?, ?)",
                [
                    ("c1", "approved", 80.0, 2, "[]"),
                    ("c2", "approved", 70.0, 1, "[]"),
                    ("c3", "rejected", 5.0, 1, '["scam"]'),
                ],
            )
            conn.execute("INSERT INTO join_queue (chat_ref, state) VALUES ('dalat_ru', 'pending')")
            conn.commit()
        return "j1"

    async def test_candidates_come_back_scored_with_what_we_already_have(
        self, tools: EidolonTools
    ) -> None:
        job_id = self._seed(tools)

        result = await tools.discovery_results(job_id=job_id)

        assert result["job"]["status"] == "completed"
        by_name = {c["username"]: c for c in result["candidates"]}
        assert [c["username"] for c in result["candidates"]] == [
            "Dalat_Vietnam",
            "dalat_ru",
            "scamcoin",
        ]
        assert by_name["Dalat_Vietnam"]["already"] == "in corpus"
        assert by_name["dalat_ru"]["already"] == "join queue: pending"
        assert by_name["scamcoin"]["already"] is None
        assert by_name["scamcoin"]["risk_flags"] == ["scam"]
        assert by_name["scamcoin"]["state"] == "rejected"

    async def test_no_job_id_means_the_latest_job(self, tools: EidolonTools) -> None:
        self._seed(tools)

        result = await tools.discovery_results()

        assert result["job"]["id"] == "j1"
        assert result["result_count"] == 3

    async def test_an_empty_store_answers_plainly(self, tools: EidolonTools) -> None:
        assert (await tools.discovery_results())["job"] is None


class TestExposure:
    def test_results_are_readable_by_julia_but_jobs_are_not(self) -> None:
        assert "discovery_results" in {t.name for t in READ_TOOLS}
        assert "discover_chats" in {t.name for t in WRITE_TOOLS}
        assert "discover_chats" not in {t.name for t in READ_TOOLS}
