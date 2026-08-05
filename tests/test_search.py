"""Tests for storage/search.py — the derived corpus index."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from storage.search import (
    SearchDatabase,
    build_fts_query,
    content_digest,
    escape_fts_term,
    extract_contacts,
    fold_ascii,
    message_link,
)

LIVE_SCHEMA = """
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_msg_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    chat_title TEXT,
    sender_id INTEGER,
    sender_name TEXT,
    text TEXT,
    date TIMESTAMP NOT NULL
);
CREATE TABLE chats (chat_id INTEGER PRIMARY KEY, title TEXT);
CREATE TABLE observed_chats (chat_id INTEGER PRIMARY KEY, title TEXT, mode TEXT);
"""

SCOUT_SCHEMA = """
CREATE TABLE scout_messages (
    chat_id INTEGER NOT NULL,
    telegram_msg_id INTEGER NOT NULL,
    sender_id INTEGER,
    sender_name TEXT,
    text TEXT,
    date TIMESTAMP NOT NULL,
    content_hash TEXT,
    source TEXT NOT NULL DEFAULT 'backfill',
    PRIMARY KEY (chat_id, telegram_msg_id)
);
CREATE TABLE backfill_targets (chat_id INTEGER PRIMARY KEY, label TEXT);
CREATE TABLE join_queue (chat_ref TEXT PRIMARY KEY, label TEXT, target_days INTEGER,
    state TEXT NOT NULL DEFAULT 'pending', joined_chat_id INTEGER);
CREATE TABLE scout_chats (chat_id INTEGER PRIMARY KEY, username TEXT);
"""


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Path, Path]:
    """Two source databases shaped like the real live and scout stores."""
    live, scout = tmp_path / "live.db", tmp_path / "scout.db"
    with sqlite3.connect(live) as conn:
        conn.executescript(LIVE_SCHEMA)
        conn.executemany(
            "INSERT INTO messages (telegram_msg_id, chat_id, chat_title, sender_name, text, date)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    1,
                    -100,
                    "Live Chat",
                    "ann",
                    "Открытый микрофон в Sound Cafe в эту субботу в 19:00, приходите петь и играть, вход свободный",
                    "2026-08-01T10:00:00+00:00",
                ),
                (2, -100, "Live Chat", "bob", "", "2026-08-01T11:00:00+00:00"),
                (
                    3,
                    -100,
                    "Live Chat",
                    "cat",
                    "Сдаю квартиру на длительный срок, депозит один месяц, есть бассейн и еженедельная уборка",
                    "2026-08-02T10:00:00+00:00",
                ),
            ],
        )
        conn.execute("INSERT INTO observed_chats VALUES (-100, 'Live Chat', 'monitor')")
    with sqlite3.connect(scout) as conn:
        conn.executescript(SCOUT_SCHEMA)
        conn.executemany(
            "INSERT INTO scout_messages (chat_id, telegram_msg_id, text, date, content_hash)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                (
                    -200,
                    10,
                    "Концерт в баре Corner Music Bar в пятницу вечером, акустический лайв и каверы",
                    "2026-07-01T10:00:00+00:00",
                    "h1",
                ),
                (
                    -200,
                    11,
                    "Концерт в баре Corner Music Bar в пятницу вечером, акустический лайв и каверы",
                    "2026-07-01T10:05:00+00:00",
                    "h1",
                ),
                (
                    -300,
                    12,
                    "WELCOME TO SYNCHØUSE COMMUNITY, качественная музыка",
                    "2026-06-01T10:00:00+00:00",
                    "h2",
                ),
                (
                    -300,
                    13,
                    "Заходите в наш чат https://t.me/danangevents там анонсы",
                    "2026-06-02T10:00:00+00:00",
                    "h3",
                ),
            ],
        )
        conn.execute("INSERT INTO backfill_targets VALUES (-200, 'Music Chat')")
    return live, scout


@pytest.fixture
def search(tmp_path: Path) -> Iterator[SearchDatabase]:
    db = SearchDatabase(tmp_path / "search.db")
    db.connect()
    yield db
    db.close()


def _sync(search: SearchDatabase, sources: tuple[Path, Path]) -> dict[str, int]:
    return search.sync(live_db=sources[0], scout_db=sources[1])


class TestFolding:
    def test_stylized_letters_fold_to_ascii(self) -> None:
        # The reason venue names need folding at all: FTS5 leaves U+00D8 alone,
        # so a search for the name a person types finds nothing without this.
        assert fold_ascii("SYNCHØUSE") == "synchouse"
        assert fold_ascii("ĐEN Studio") == "den studio"
        assert fold_ascii("Café Zürich") == "cafe zurich"

    def test_folding_collapses_whitespace(self) -> None:
        assert fold_ascii("  Corner   Music  Bar ") == "corner music bar"

    def test_cyrillic_is_left_alone(self) -> None:
        assert fold_ascii("Кафе Пространство") == "кафе пространство"


class TestQueryBuilding:
    def test_terms_become_or_of_prefixes(self) -> None:
        assert build_fts_query(["концерт", "бар"]) == '"концерт"* OR "бар"*'

    def test_phrases_are_quoted_without_a_prefix_star(self) -> None:
        # FTS5 rejects `"a b"*`; a phrase must stay a bare quoted phrase.
        assert build_fts_query(["живая музыка"]) == '"живая музыка"'

    def test_quotes_in_input_are_escaped_not_executed(self) -> None:
        assert escape_fts_term('say "hi"') == '"say ""hi"""'

    @pytest.mark.parametrize("hostile", ['" OR "', "AND NOT", "a-b", "*", '"'])
    def test_hostile_input_stays_a_valid_query(
        self, search: SearchDatabase, sources: tuple[Path, Path], hostile: str
    ) -> None:
        # Model-supplied terms reach MATCH directly. Unescaped, these are syntax
        # errors that fail the whole call rather than returning nothing.
        _sync(search, sources)
        assert search.lexical_search(build_fts_query([hostile]), limit=5) == []

    def test_empty_terms_are_dropped(self) -> None:
        assert build_fts_query(["", "  ", "бар"]) == '"бар"*'


class TestSync:
    def test_both_stores_land_in_one_corpus(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        report = _sync(search, sources)
        assert report["live"] == 2
        assert report["scout"] == 4
        assert search.status()["messages_indexed"] == 6

    def test_empty_messages_are_not_indexed(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        rows = search.conn.execute("SELECT text FROM corpus_messages").fetchall()
        assert all(r["text"].strip() for r in rows)

    def test_second_sync_is_a_no_op(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        second = _sync(search, sources)
        assert (second["live"], second["scout"], second["contacts"]) == (0, 0, 0)

    def test_new_rows_are_picked_up_incrementally(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        with sqlite3.connect(sources[0]) as conn:
            conn.execute(
                "INSERT INTO messages (telegram_msg_id, chat_id, text, date)"
                " VALUES (99, -100, 'Квиз в баре вечером', '2026-08-03T10:00:00+00:00')"
            )
        assert _sync(search, sources)["live"] == 1

    def test_a_purged_source_row_does_not_trigger_a_rescan(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # messages.id is AUTOINCREMENT, so the daily retention purge cannot
        # cause id reuse. Treating a purge as drift would rescan the table on
        # every tick forever after the first purge.
        _sync(search, sources)
        with sqlite3.connect(sources[0]) as conn:
            conn.execute("DELETE FROM messages WHERE telegram_msg_id = 1")
        assert _sync(search, sources)["live"] == 0
        cursor = search.conn.execute(
            "SELECT last_id FROM sync_state WHERE source = 'live'"
        ).fetchone()["last_id"]
        assert cursor > 0

    def test_purged_messages_survive_in_the_corpus(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # Retention drops live messages after 30 days; the corpus is the only
        # place that history continues to exist.
        _sync(search, sources)
        with sqlite3.connect(sources[0]) as conn:
            conn.execute("DELETE FROM messages")
        _sync(search, sources)
        assert search.lexical_search(build_fts_query(["микрофон"]), limit=5)

    def test_a_scheduled_rescan_recovers_a_row_hidden_by_rowid_reuse(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # scout_messages has only a composite key, so SQLite reuses its rowid
        # after a delete. Delete one row and insert another and the row COUNT is
        # unchanged while the new row sits below the cursor -- so no count-based
        # check can see it. Only re-reading the stream can.
        _sync(search, sources)
        with sqlite3.connect(sources[1]) as conn:
            conn.execute("DELETE FROM scout_messages WHERE telegram_msg_id = 13")
            conn.execute(
                "INSERT INTO scout_messages (chat_id, telegram_msg_id, text, date, content_hash)"
                " VALUES (-300, 77, 'Джем-сессия в студии по четвергам, приносите инструменты и хорошее настроение',"
                " '2026-06-03T10:00:00+00:00', 'h9')"
            )
        assert _sync(search, sources)["scout"] == 0  # invisible to the cursor
        assert search.sync(live_db=sources[0], scout_db=sources[1], rescan_every=1)["scout"] == 1
        assert search.lexical_search(build_fts_query(["джем"]), limit=5)

    def test_the_rescan_counter_resets_after_a_full_pass(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        search.sync(live_db=sources[0], scout_db=sources[1], rescan_every=1)
        ticks = search.conn.execute(
            "SELECT ticks FROM sync_state WHERE source = 'scout'"
        ).fetchone()["ticks"]
        assert ticks == 1  # reset to 0 by the rescan, then incremented once

    def test_chat_titles_come_from_the_observation_registry(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        titles = {c["chat_id"]: c["title"] for c in search.chats()}
        assert titles[-100] == "Live Chat"
        assert titles[-200] == "Music Chat"  # fell back to the backfill label

    def test_chats_without_an_observation_row_are_marked_history_only(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        modes = {c["chat_id"]: c["mode"] for c in search.chats()}
        assert modes[-100] == "monitor"
        assert modes[-200] == "history-only"


class TestLexicalSearch:
    def test_russian_suffixes_match_a_stem(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # "Концерт в баре" must be found by the stem "бар". An exact-token index
        # misses this, which measured at 14.6% recall on the real corpus.
        _sync(search, sources)
        assert len(search.lexical_search(build_fts_query(["бар"]), limit=10)) == 2

    def test_date_filter_bounds_results(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        rows = search.lexical_search(
            build_fts_query(["концерт"]), limit=10, since="2026-08-01T00:00:00+00:00"
        )
        assert rows == []

    def test_chat_filter_bounds_results(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        rows = search.lexical_search(build_fts_query(["концерт"]), limit=10, chat_ids=[-999])
        assert rows == []

    def test_stylized_venue_name_is_unreachable_lexically(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # Documents the limit that justifies the ASCII-folded places index:
        # the corpus spells it SYNCHØUSE and no tokenizer folds U+00D8.
        _sync(search, sources)
        assert search.lexical_search(build_fts_query(["synchouse"]), limit=5) == []
        assert search.lexical_search(build_fts_query(["synchøuse"]), limit=5)


class TestDeduplication:
    def test_crossposts_collapse_into_one_result(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # A quarter of the real corpus is the same announcement posted to
        # several chats; without rollup one event fills the whole answer.
        _sync(search, sources)
        rows = search.hybrid_search(
            match_query=build_fts_query(["концерт"]), query_vector=None, limit=10
        )
        assert len(rows) == 1

    def test_the_other_chats_are_reported_not_discarded(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        rows = search.hybrid_search(
            match_query=build_fts_query(["концерт"]), query_vector=None, limit=10
        )
        assert rows[0].duplicate_chat_ids == []  # same chat, so not "also posted in"

    def test_content_digest_matches_the_scout_convention(self) -> None:
        # Rows synced from the live store get their digest computed here; if it
        # disagreed with scout's, the same message would never dedupe.
        assert content_digest("  Hello   World  ") == content_digest("hello world")
        assert content_digest("") is None


class TestSemanticSearch:
    def test_no_vectors_means_no_results_not_an_error(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        assert search.semantic_search([0.1] * 4, limit=5) == []

    def test_ranks_by_cosine_similarity(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        ids = [
            r["corpus_id"]
            for r in search.conn.execute("SELECT corpus_id FROM corpus_messages ORDER BY corpus_id")
        ]
        search.store_embeddings(
            [(ids[0], [1.0, 0.0]), (ids[1], [0.0, 1.0]), (ids[2], [0.7, 0.7])],
            model="test",
        )
        rows = search.semantic_search([1.0, 0.0], limit=3)
        assert [r.corpus_id for r in rows] == [ids[0], ids[2], ids[1]]

    def test_a_zero_query_vector_returns_nothing(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        ids = [r["corpus_id"] for r in search.conn.execute("SELECT corpus_id FROM corpus_messages")]
        search.store_embeddings([(ids[0], [1.0, 0.0])], model="test")
        assert search.semantic_search([0.0, 0.0], limit=3) == []

    def test_filters_survive_the_vector_prefetch(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # The vector lane ranks before SQL filters run, so a naive
        # top-k-then-filter empties the result whenever the best matches sit
        # outside the requested chat.
        _sync(search, sources)
        rows = search.conn.execute(
            "SELECT corpus_id, chat_id FROM corpus_messages ORDER BY corpus_id"
        ).fetchall()
        search.store_embeddings([(r["corpus_id"], [1.0, 0.0]) for r in rows], model="test")
        target = next(r["chat_id"] for r in rows)
        got = search.semantic_search([1.0, 0.0], limit=1, chat_ids=[target])
        assert got and all(r.chat_id == target for r in got)


class TestHybridFusion:
    def test_a_hit_in_either_lane_survives(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # The whole point of RRF here: semantic is the recall net for posts that
        # use none of the expected vocabulary.
        _sync(search, sources)
        rows = search.conn.execute(
            "SELECT corpus_id, text FROM corpus_messages ORDER BY corpus_id"
        ).fetchall()
        rental = next(r["corpus_id"] for r in rows if "Сдаю" in r["text"])
        search.store_embeddings([(rental, [1.0, 0.0])], model="test")
        fused = search.hybrid_search(
            match_query=build_fts_query(["концерт"]), query_vector=[1.0, 0.0], limit=10
        )
        found = {r.corpus_id for r in fused}
        assert rental in found  # semantic-only hit
        assert len(found) > 1  # lexical-only hits too

    def test_each_result_reports_the_lane_that_found_it(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # Callers use `matched_via` to judge how much to trust a hit; a
        # semantic-only result labelled with nothing reads as a lexical match
        # that simply scored badly.
        _sync(search, sources)
        rows = search.conn.execute(
            "SELECT corpus_id, text FROM corpus_messages ORDER BY corpus_id"
        ).fetchall()
        rental = next(r["corpus_id"] for r in rows if "Сдаю" in r["text"])
        search.store_embeddings([(rental, [1.0, 0.0])], model="test")
        fused = {
            r.corpus_id: r
            for r in search.hybrid_search(
                match_query=build_fts_query(["концерт"]), query_vector=[1.0, 0.0], limit=10
            )
        }
        assert fused[rental].lanes == ["semantic"]
        lexical_only = next(r for cid, r in fused.items() if cid != rental)
        assert lexical_only.lanes == ["lexical"]

    def test_a_hit_in_both_lanes_reports_both(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        row = search.conn.execute(
            "SELECT corpus_id FROM corpus_messages WHERE text LIKE '%Концерт%' ORDER BY corpus_id"
        ).fetchone()
        search.store_embeddings([(row["corpus_id"], [1.0, 0.0])], model="test")
        fused = search.hybrid_search(
            match_query=build_fts_query(["концерт"]), query_vector=[1.0, 0.0], limit=10
        )
        assert sorted(fused[0].lanes) == ["lexical", "semantic"]

    def test_no_query_at_all_returns_nothing(self, search: SearchDatabase) -> None:
        assert search.hybrid_search(match_query=None, query_vector=None, limit=5) == []


class TestPlaces:
    def _corpus_id(self, search: SearchDatabase) -> int:
        return int(search.conn.execute("SELECT min(corpus_id) FROM corpus_messages").fetchone()[0])

    def test_extraction_stores_a_venue_with_its_evidence(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        stored = search.record_extraction(
            self._corpus_id(search),
            [
                {
                    "name": "Sound Cafe",
                    "place_type": "cafe",
                    "city_area": "Da Nang",
                    "event_types": ["open_mic"],
                    "evidence": "Открытый микрофон в Sound Cafe",
                    "confidence": 0.9,
                }
            ],
            model="test-model",
        )
        assert stored == 1
        found = search.search_places(name_query="sound cafe")
        assert found[0]["name"] == "Sound Cafe"
        assert found[0]["evidence"][0]["quote"] == "Открытый микрофон в Sound Cafe"

    def test_a_stylized_name_is_found_by_its_plain_spelling(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # This is the gap the message index provably cannot close.
        _sync(search, sources)
        search.record_extraction(
            self._corpus_id(search),
            [
                {
                    "name": "SYNCHØUSE",
                    "place_type": "club",
                    "city_area": "Da Nang",
                    "event_types": ["dj_set"],
                    "evidence": "WELCOME TO SYNCHØUSE",
                    "confidence": 0.8,
                }
            ],
            model="test-model",
        )
        assert search.search_places(name_query="synchouse")[0]["name"] == "SYNCHØUSE"

    def test_the_same_venue_from_two_messages_is_one_place(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        ids = [
            r["corpus_id"]
            for r in search.conn.execute(
                "SELECT corpus_id FROM corpus_messages ORDER BY corpus_id LIMIT 2"
            )
        ]
        for corpus_id in ids:
            search.record_extraction(
                corpus_id,
                [
                    {
                        "name": "Corner Music Bar",
                        "place_type": "bar",
                        "city_area": "Da Nang",
                        "event_types": ["concert"],
                        "evidence": "Corner Music Bar",
                        "confidence": 0.9,
                    }
                ],
                model="test-model",
            )
        found = search.search_places(name_query="corner music bar")
        assert len(found) == 1
        assert found[0]["mentions"] == 2

    def test_event_type_filter_selects_the_right_venues(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        ids = [
            r["corpus_id"]
            for r in search.conn.execute(
                "SELECT corpus_id FROM corpus_messages ORDER BY corpus_id LIMIT 2"
            )
        ]
        search.record_extraction(
            ids[0],
            [
                {
                    "name": "Music Bar",
                    "place_type": "bar",
                    "city_area": "Da Nang",
                    "event_types": ["concert", "live_music"],
                    "evidence": "q",
                    "confidence": 0.9,
                }
            ],
            model="m",
        )
        search.record_extraction(
            ids[1],
            [
                {
                    "name": "Yoga Room",
                    "place_type": "studio",
                    "city_area": "Da Nang",
                    "event_types": ["yoga"],
                    "evidence": "q",
                    "confidence": 0.9,
                }
            ],
            model="m",
        )
        names = {p["name"] for p in search.search_places(event_types=["concert"])}
        assert names == {"Music Bar"}

    def test_a_message_with_no_venue_is_closed_out_not_retried(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        corpus_id = self._corpus_id(search)
        search.record_extraction(corpus_id, [], model="test-model")
        status = search.conn.execute(
            "SELECT status FROM extraction_state WHERE corpus_id = ?", (corpus_id,)
        ).fetchone()["status"]
        assert status == "no_venue"

    def test_an_extraction_error_is_recorded_not_swallowed(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        corpus_id = self._corpus_id(search)
        search.record_extraction(corpus_id, [], model="m", error="RateLimitError: slow down")
        row = search.conn.execute(
            "SELECT status, error FROM extraction_state WHERE corpus_id = ?", (corpus_id,)
        ).fetchone()
        assert row["status"] == "error"
        assert "RateLimit" in row["error"]

    def test_short_messages_are_skipped_not_queued(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        skipped = search.conn.execute(
            "SELECT count(*) FROM extraction_state WHERE status = 'skipped'"
        ).fetchone()[0]
        assert skipped > 0
        assert all(len(r["text"]) >= 80 for r in search.pending_extractions(50))


class TestChatReferences:
    def test_a_tme_link_becomes_a_candidate(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        search.refresh_chat_references()
        refs = {r["ref"] for r in search.chat_references(min_mentions=1)}
        assert "danangevents" in refs

    def test_already_known_chats_are_filtered_out(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # Telling the agent to chase a chat the account already joined wastes a
        # turn and, if it acts, a slice of the Telegram action budget.
        _sync(search, sources)
        search.refresh_chat_references(known={"danangevents": "joined"})
        refs = {r["ref"] for r in search.chat_references(min_mentions=1, only_unknown=True)}
        assert "danangevents" not in refs
        all_refs = {r["ref"] for r in search.chat_references(min_mentions=1, only_unknown=False)}
        assert "danangevents" in all_refs


class TestLinks:
    def test_supergroup_ids_lose_their_prefix(self) -> None:
        assert message_link(-1001914345108, 42) == "https://t.me/c/1914345108/42"

    def test_plain_group_ids_lose_only_the_sign(self) -> None:
        assert message_link(-4438983220, 7) == "https://t.me/c/4438983220/7"


class TestStatus:
    def test_status_reports_the_backlogs_that_gate_answers(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        status = search.status()
        assert status["messages_indexed"] == 6
        assert status["embedding_backlog"] == 6
        assert status["extraction_backlog"] > 0
        assert {"live", "scout"} <= {s["source"] for s in status["sync"]}


class TestJoinRequestNormalization:
    """The MCP bridge must key join requests exactly as the daemon does."""

    @pytest.fixture(autouse=True)
    def _worker_running(self, monkeypatch: Any) -> None:
        # These tests are about how a reference is normalised, not about whether
        # a join worker exists to act on it.
        import config.settings as cs

        monkeypatch.setattr(cs.settings, "join_queue_enabled", True)

    @pytest.mark.parametrize(
        "given",
        [
            "danangevents",
            "@danangevents",
            "https://t.me/danangevents",
            "https://t.me/danangevents/",
            "t.me/danangevents?start=1",
            "DanangEvents",
        ],
    )
    async def test_every_spelling_of_one_chat_queues_the_same_row(
        self, tmp_path: Path, sources: tuple[Path, Path], given: str
    ) -> None:
        # Each distinct key would be a separate join_queue row and a separate
        # join attempt against an action budget measured in a few per day.
        from eidolon_mcp import EidolonTools

        search_db = tmp_path / "search.db"
        SearchDatabase(search_db).connect()
        tools = EidolonTools(search_db=search_db, scout_db=sources[1], writable=True)
        result = await tools.queue_chat_join(given)
        assert result["chat_ref"] == "danangevents"
        rows = sqlite3.connect(sources[1]).execute("SELECT chat_ref FROM join_queue").fetchall()
        assert rows == [("danangevents",)]

    async def test_requeueing_an_existing_chat_reports_it_instead_of_duplicating(
        self, tmp_path: Path, sources: tuple[Path, Path]
    ) -> None:
        from eidolon_mcp import EidolonTools

        search_db = tmp_path / "search.db"
        SearchDatabase(search_db).connect()
        tools = EidolonTools(search_db=search_db, scout_db=sources[1], writable=True)
        await tools.queue_chat_join("danangevents")
        again = await tools.queue_chat_join("https://t.me/danangevents/")
        assert again["queued"] is False
        assert again["already"] == "pending"

    async def test_a_readonly_bridge_refuses_to_queue(
        self, tmp_path: Path, sources: tuple[Path, Path]
    ) -> None:
        # Julia's instance runs read-only; a write reaching the daemon from
        # there would spend Nikita's Telegram budget.
        from eidolon_mcp import EidolonTools

        search_db = tmp_path / "search.db"
        SearchDatabase(search_db).connect()
        tools = EidolonTools(search_db=search_db, scout_db=sources[1], writable=False)
        with pytest.raises(PermissionError):
            await tools.queue_chat_join("danangevents")
        assert (
            sqlite3.connect(sources[1]).execute("SELECT count(*) FROM join_queue").fetchone()[0]
            == 0
        )

    async def test_readonly_profile_does_not_advertise_the_write_tool(
        self, tmp_path: Path, sources: tuple[Path, Path]
    ) -> None:
        from eidolon_mcp import READ_TOOLS, WRITE_TOOLS

        read_names = {t.name for t in READ_TOOLS}
        assert not {"queue_chat_join", "resume_chat", "backfill_chat"} & read_names
        assert {t.name for t in WRITE_TOOLS} == {
            "queue_chat_join",
            "resume_chat",
            "backfill_chat",
        }


class HardStop(BaseException):
    """Escapes gather(return_exceptions=True), as a signal or an OOM would."""


class TestChunkedExtraction:
    """A full-corpus pass is tens of minutes of paid calls; it must checkpoint."""

    @staticmethod
    def _queue(search: SearchDatabase, count: int) -> None:
        for i in range(count):
            search.conn.execute(
                "INSERT INTO corpus_messages (source, chat_id, telegram_msg_id, text, date,"
                " content_hash) VALUES ('scout', -900, ?, ?, '2026-07-01T00:00:00+00:00', ?)",
                (
                    i,
                    f"Концерт номер {i} в баре, живая музыка и хорошее настроение всем гостям",
                    f"c{i}",
                ),
            )
        search.conn.execute(
            "INSERT INTO extraction_state (corpus_id, status)"
            " SELECT corpus_id, 'pending' FROM corpus_messages WHERE chat_id = -900"
        )
        search.conn.commit()

    async def test_finished_chunks_are_committed_before_the_pass_ends(
        self, search: SearchDatabase
    ) -> None:
        # The distinguishing property of chunking is WHEN the write happens. A
        # single gather over the whole backlog commits nothing until the end, so
        # an interruption anywhere throws away the entire pass -- tens of
        # minutes of paid calls. Here the pass dies between chunks; whatever the
        # first chunk finished must already be on disk.
        from pipeline.indexer import PlaceExtractor

        self._queue(search, 12)
        extractor = PlaceExtractor(search, client=object(), chunk_size=4, concurrency=2)

        async def fake(row: object) -> tuple[int, list[dict[str, object]], str | None]:
            return (
                row["corpus_id"],  # type: ignore[index]
                [
                    {
                        "name": "Corner Music Bar",
                        "place_type": "bar",
                        "city_area": "Da Nang",
                        "event_types": ["concert"],
                        "evidence": "в баре",
                        "confidence": 0.9,
                    }
                ],
                None,
            )

        extractor._extract = fake  # type: ignore[method-assign]

        real_pending = search.pending_extractions
        fetches = {"n": 0}

        def dying_pending(limit: int) -> list[sqlite3.Row]:
            fetches["n"] += 1
            if fetches["n"] > 1:
                raise HardStop("process died between chunks")
            return real_pending(limit)

        search.pending_extractions = dying_pending  # type: ignore[method-assign]
        with pytest.raises(HardStop):
            await extractor.run(limit=12)
        search.pending_extractions = real_pending  # type: ignore[method-assign]

        found = search.search_places(name_query="corner music bar")
        assert found, "an interrupted pass lost every result it had already paid for"
        assert found[0]["mentions"] == 4, "only the completed chunk should be committed"
        assert (
            search.conn.execute(
                "SELECT count(*) FROM extraction_state WHERE status = 'pending'"
            ).fetchone()[0]
            == 8
        ), "unfinished messages must stay pending so the next run resumes them"

    async def test_a_second_run_does_not_redo_finished_work(self, search: SearchDatabase) -> None:
        from pipeline.indexer import PlaceExtractor

        self._queue(search, 6)
        seen: list[int] = []

        async def fake(row: object) -> tuple[int, list[dict[str, object]], str | None]:
            seen.append(row["corpus_id"])  # type: ignore[index]
            return (row["corpus_id"], [], None)  # type: ignore[index]

        first = PlaceExtractor(search, client=object(), chunk_size=3)
        first._extract = fake  # type: ignore[method-assign]
        await first.run(limit=6)
        assert len(seen) == 6

        second = PlaceExtractor(search, client=object(), chunk_size=3)
        second._extract = fake  # type: ignore[method-assign]
        assert (await second.run(limit=6))["processed"] == 0
        assert len(seen) == 6  # no message paid for twice


class TestVenueNameValidation:
    """A bad name is permanent: nothing later in the pipeline removes it."""

    @pytest.mark.parametrize(
        "rejected",
        [
            "https://maps.app.goo.gl/z3HZSSKzKqAbC",
            "http://example.com/place",
            "www.somebar.vn",
            "t.me/danangevents",
            "@some_channel",
            "maps.app.goo.gl/xyz",
            "+84 905 123 456",
            "18:30",
            "",
            "a",
        ],
    )
    def test_non_names_never_enter_the_index(self, rejected: str) -> None:
        from storage.search import is_venue_name

        assert not is_venue_name(rejected)

    @pytest.mark.parametrize(
        "accepted",
        ["Corner Music Bar", "SYNCHØUSE", "Sound Cafe", "PlantLab", "Кафе Пространство", "IMIX"],
    )
    def test_real_venue_names_pass(self, accepted: str) -> None:
        from storage.search import is_venue_name

        assert is_venue_name(accepted)

    def test_a_url_returned_as_a_name_is_dropped_not_stored(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # Observed for real: the model was asked for the venue name and returned
        # the Google Maps link that sat next to it in the message.
        _sync(search, sources)
        corpus_id = int(
            search.conn.execute("SELECT min(corpus_id) FROM corpus_messages").fetchone()[0]
        )
        stored = search.record_extraction(
            corpus_id,
            [
                {
                    "name": "https://maps.app.goo.gl/z3HZSSKzKq",
                    "place_type": "bar",
                    "city_area": "Da Nang",
                    "event_types": ["live_music"],
                    "evidence": "📍 https://maps.app.goo.gl/z3HZSSKzKq",
                    "confidence": 0.7,
                },
                {
                    "name": "Corner Music Bar",
                    "place_type": "bar",
                    "city_area": "Da Nang",
                    "event_types": ["live_music"],
                    "evidence": "в Corner Music Bar",
                    "confidence": 0.9,
                },
            ],
            model="test-model",
        )
        assert stored == 1
        names = {p["name"] for p in search.search_places()}
        assert names == {"Corner Music Bar"}


class TestExtractionRetry:
    """A provider timeout must not become a permanent hole in the venue index."""

    def _one(self, search: SearchDatabase) -> int:
        return int(search.conn.execute("SELECT min(corpus_id) FROM corpus_messages").fetchone()[0])

    def test_a_fresh_failure_is_not_retried_immediately(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # Retrying inside the same pass would just hit the same outage again.
        _sync(search, sources)
        corpus_id = self._one(search)
        search.record_extraction(corpus_id, [], model="m", error="APITimeoutError: timed out")
        assert corpus_id not in {r["corpus_id"] for r in search.pending_extractions(50)}

    def test_a_stale_failure_comes_back_as_work(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # The timestamp is aged by rewriting it in the SAME format the code
        # writes, ISO-8601 with a "T". Setting it with SQLite's own datetime()
        # instead would compare space-separated text against space-separated
        # text and pass no matter how the query is written.
        _sync(search, sources)
        corpus_id = self._one(search)
        search.record_extraction(corpus_id, [], model="m", error="APITimeoutError: timed out")
        stale = (datetime.now(UTC) - timedelta(days=2)).isoformat(timespec="seconds")
        search.conn.execute(
            "UPDATE extraction_state SET attempted_at = ? WHERE corpus_id = ?", (stale, corpus_id)
        )
        search.conn.commit()
        assert corpus_id in {r["corpus_id"] for r in search.pending_extractions(50)}

    def test_a_same_day_failure_past_the_cooldown_is_retried(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # The case a raw text comparison gets wrong: the date matches, so the
        # separator decides. "…T04:00" sorts above "… 05:00", so a failure
        # hours past the cooldown reads as newer than the cutoff and the
        # six-hour window silently becomes "not until tomorrow".
        _sync(search, sources)
        corpus_id = self._one(search)
        search.record_extraction(corpus_id, [], model="m", error="APITimeoutError: timed out")
        seven_hours_ago = (datetime.now(UTC) - timedelta(hours=7)).isoformat(timespec="seconds")
        search.conn.execute(
            "UPDATE extraction_state SET attempted_at = ? WHERE corpus_id = ?",
            (seven_hours_ago, corpus_id),
        )
        search.conn.commit()
        assert corpus_id in {
            r["corpus_id"] for r in search.pending_extractions(50, retry_after_hours=6)
        }

    def test_a_message_that_keeps_failing_is_eventually_left_alone(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # Something about this specific message breaks extraction. Retrying it
        # for the life of the corpus only spends money.
        _sync(search, sources)
        corpus_id = self._one(search)
        stale = (datetime.now(UTC) - timedelta(days=2)).isoformat(timespec="seconds")
        for _ in range(3):
            search.record_extraction(corpus_id, [], model="m", error="connection reset")
            search.conn.execute(
                "UPDATE extraction_state SET attempted_at = ? WHERE corpus_id = ?",
                (stale, corpus_id),
            )
            search.conn.commit()
        assert corpus_id not in {r["corpus_id"] for r in search.pending_extractions(50)}

    def test_a_successful_extraction_is_never_retried(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        corpus_id = self._one(search)
        search.record_extraction(corpus_id, [], model="m")
        stale = (datetime.now(UTC) - timedelta(days=9)).isoformat(timespec="seconds")
        search.conn.execute(
            "UPDATE extraction_state SET attempted_at = ? WHERE corpus_id = ?", (stale, corpus_id)
        )
        search.conn.commit()
        assert corpus_id not in {r["corpus_id"] for r in search.pending_extractions(50)}

    def test_migration_adds_attempts_to_an_index_built_before_it(self, tmp_path: Path) -> None:
        # The index is rebuildable in principle, but a rebuild re-pays for the
        # whole extraction pass, so the column has to land in place.
        legacy = tmp_path / "legacy.db"
        with sqlite3.connect(legacy) as conn:
            conn.execute(
                "CREATE TABLE extraction_state (corpus_id INTEGER PRIMARY KEY,"
                " status TEXT NOT NULL DEFAULT 'pending', attempted_at TIMESTAMP, error TEXT)"
            )
            conn.execute("INSERT INTO extraction_state (corpus_id, status) VALUES (1, 'error')")
        db = SearchDatabase(legacy)
        db.connect()
        columns = {r["name"] for r in db.conn.execute("PRAGMA table_info(extraction_state)")}
        assert "attempts" in columns
        assert (
            db.conn.execute("SELECT attempts FROM extraction_state WHERE corpus_id = 1").fetchone()[
                "attempts"
            ]
            == 0
        )
        db.close()


class TestTransientErrorClassification:
    """Only a provider fault is worth paying for the same call twice."""

    @pytest.mark.parametrize(
        "transient",
        [
            "APITimeoutError: Request timed out.",
            "RateLimitError: rate limit exceeded",
            "APIConnectionError: Connection error",
            "InternalServerError: 503 Service Unavailable",
            "server overloaded",
        ],
    )
    def test_provider_faults_are_retryable(self, transient: str) -> None:
        from storage.search import is_transient_error

        assert is_transient_error(transient)

    @pytest.mark.parametrize(
        "terminal",
        [
            "unparsed response: refusal",
            "unparsed response: no content",
            "json decode: Expecting value",
            "missing places array",
        ],
    )
    def test_message_specific_failures_are_not(self, terminal: str) -> None:
        from storage.search import is_transient_error

        assert not is_transient_error(terminal)

    def test_a_refusal_is_never_sent_back_to_the_model(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # The same input would produce the same refusal, at the same price.
        _sync(search, sources)
        corpus_id = int(
            search.conn.execute("SELECT min(corpus_id) FROM corpus_messages").fetchone()[0]
        )
        search.record_extraction(corpus_id, [], model="m", error="unparsed response: refusal")
        stale = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
        search.conn.execute(
            "UPDATE extraction_state SET attempted_at = ? WHERE corpus_id = ?", (stale, corpus_id)
        )
        search.conn.commit()
        assert corpus_id not in {r["corpus_id"] for r in search.pending_extractions(50)}

    def test_a_timeout_on_the_same_message_still_is(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        corpus_id = int(
            search.conn.execute("SELECT min(corpus_id) FROM corpus_messages").fetchone()[0]
        )
        search.record_extraction(corpus_id, [], model="m", error="APITimeoutError: timed out")
        stale = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="seconds")
        search.conn.execute(
            "UPDATE extraction_state SET attempted_at = ? WHERE corpus_id = ?", (stale, corpus_id)
        )
        search.conn.commit()
        assert corpus_id in {r["corpus_id"] for r in search.pending_extractions(50)}


class TestEmbeddingSpaceMatchesProduction:
    """A backtest run in a different vector space than production is fiction."""

    def test_the_indexer_defaults_to_the_models_native_width(self) -> None:
        # pipeline/embeddings.py (the live L2 path) passes no `dimensions`, so it
        # gets the model's native width. The corpus index must ask for the same
        # thing, or a proposed watcher is scored against vectors production will
        # never compute. Measured while they differed (corpus 512, live 1536):
        # 17.5% of near-threshold decisions flipped, and the disagreements ran
        # one way — the narrower space predicted passes production would not make.
        from openai import NOT_GIVEN

        from pipeline.indexer import EmbeddingIndexer

        indexer = EmbeddingIndexer(search=None, client=object())  # type: ignore[arg-type]
        assert indexer._dimensions is None
        assert indexer._width is NOT_GIVEN

    def test_the_live_filter_does_not_pin_a_width_either(self) -> None:
        # The invariant is symmetric: if either side starts pinning a width, the
        # other has to follow in the same commit.
        from pathlib import Path

        source = Path("pipeline/embeddings.py").read_text(encoding="utf-8")
        pinned = [
            line
            for line in source.splitlines()
            if "dimensions" in line and not line.lstrip().startswith("#")
        ]
        assert not pinned, (
            "pipeline/embeddings.py now pins an embedding width; "
            f"pipeline/indexer.py must match it in the same change: {pinned}"
        )

    def test_an_explicit_width_is_still_honoured(self) -> None:
        from pipeline.indexer import EmbeddingIndexer

        indexer = EmbeddingIndexer(search=None, client=object(), dimensions=512)  # type: ignore[arg-type]
        assert indexer._width == 512


class TestContentTermExtraction:
    """A question is not a keyword list."""

    def test_russian_stopwords_are_dropped(self) -> None:
        from storage.search import content_terms

        assert content_terms("площадка в Дананге где проводят живую музыку и концерты") == [
            "площадка",
            "Дананге",
            "проводят",
            "живую",
            "музыку",
            "концерты",
        ]

    def test_english_stopwords_are_dropped(self) -> None:
        from storage.search import content_terms

        assert "the" not in content_terms("a venue in the city that hosts live music")
        assert "live" in content_terms("a venue in the city that hosts live music")

    def test_punctuation_and_digits_do_not_become_terms(self) -> None:
        from storage.search import content_terms

        assert content_terms("концерт 6 августа, 20:00!") == ["концерт", "августа"]

    def test_a_query_of_only_stopwords_yields_nothing_to_search(self) -> None:
        # The caller must then fall back to semantic search rather than run a
        # lexical query that matches everything.
        from storage.search import content_terms

        assert content_terms("а что там и как") == []

    def test_the_resulting_query_excludes_the_corpus_wide_terms(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        # The failure this prevents: a one-letter preposition, prefix-matched,
        # hits nearly every message, and BM25 then ranks on it.
        from storage.search import build_fts_query, content_terms

        _sync(search, sources)
        naive = search.lexical_search(build_fts_query(["концерт", "в", "баре"]), limit=20)
        focused = search.lexical_search(build_fts_query(content_terms("концерт в баре")), limit=20)
        assert len(focused) <= len(naive)
        assert focused, "dropping stopwords must not drop the real terms"


class TestContactExtraction:
    """Contacts are lexical, so the risk is precision, not recall."""

    def test_telegram_handles_collapse_across_spellings(self) -> None:
        found = extract_contacts("пишите @AUM_danang или https://t.me/aum_danang")
        telegram = [c for c in found if c[0] == "telegram"]
        assert len(telegram) == 1, "the same handle in two spellings is one contact"
        assert telegram[0][1] == "aum_danang"

    def test_vietnamese_local_and_international_are_one_number(self) -> None:
        local = extract_contacts("звоните 0905 123 456")
        intl = extract_contacts("call +84 905 123 456")
        assert [c[1] for c in local] == ["+84905123456"]
        assert [c[1] for c in intl] == ["+84905123456"]

    @pytest.mark.parametrize(
        "text",
        [
            "аренда 500.000 VND в сутки",
            "квартира 1 500 000 vnd в месяц",
            "площадь 120 m2 рядом с пляжем",
            "цена 12 000 000đ",
            # Long enough to clear every length bound, so only the leading-zero
            # mobile prefix separates it from a phone number.
            "продаю дом за 1 200 000 000 донгов",
            "оборот 4 500 000 000 vnd за год",
        ],
    )
    def test_prices_and_measurements_are_not_phone_numbers(self, text: str) -> None:
        # This corpus is mostly rental ads. A price read as a phone number is
        # worse than no number: the reader only finds out by dialling it.
        assert [c for c in extract_contacts(text) if c[0] == "phone"] == []

    def test_social_links_are_taken_only_from_a_full_url(self) -> None:
        found = {
            kind: value
            for kind, value, _ in extract_contacts(
                "наш инстаграм instagram.com/aum.danang, whatsapp wa.me/84905123456, "
                "карта https://maps.app.goo.gl/abc123XY"
            )
        }
        assert found["instagram"] == "aum.danang"
        assert found["whatsapp"] == "84905123456"
        assert found["maps"].endswith("abc123xy")

    def test_a_bare_handle_is_not_claimed_for_instagram(self) -> None:
        # "@someone" in a sentence about Instagram is indistinguishable from a
        # Telegram handle; guessing hands the reader a contact that does not exist.
        kinds = {kind for kind, _, _ in extract_contacts("наш инстаграм @aum_danang")}
        assert kinds == {"telegram"}


class TestContactsInTheIndex:
    def test_sync_mines_contacts_out_of_message_text(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        stats = _sync(search, sources)
        assert stats["contacts"] >= 1
        rows = search.conn.execute(
            "SELECT kind, value FROM message_contacts ORDER BY value"
        ).fetchall()
        assert ("telegram", "danangevents") in [(r["kind"], r["value"]) for r in rows]

    def test_rescanning_does_not_duplicate_contacts(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        before = search.conn.execute("SELECT count(*) FROM message_contacts").fetchone()[0]
        _sync(search, sources)
        after = search.conn.execute("SELECT count(*) FROM message_contacts").fetchone()[0]
        assert after == before

    def test_places_carry_the_contacts_of_the_messages_that_name_them(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        corpus_id = search.conn.execute(
            "SELECT corpus_id FROM corpus_messages WHERE text LIKE '%danangevents%'"
        ).fetchone()["corpus_id"]
        search.conn.execute(
            "INSERT INTO places (canonical, name, mention_count) VALUES ('venue', 'Venue', 1)"
        )
        place_id = search.conn.execute("SELECT place_id FROM places").fetchone()["place_id"]
        search.conn.execute(
            "INSERT INTO place_mentions (place_id, corpus_id, evidence_quote, extracted_by)"
            " VALUES (?, ?, 'quote', 'test')",
            (place_id, corpus_id),
        )
        search.conn.commit()

        results = search.search_places(name_query=None)

        assert [c["value"] for c in results[0]["contacts"]] == ["danangevents"]

    def test_contacts_can_be_left_out(
        self, search: SearchDatabase, sources: tuple[Path, Path]
    ) -> None:
        _sync(search, sources)
        search.conn.execute(
            "INSERT INTO places (canonical, name, mention_count) VALUES ('venue', 'Venue', 1)"
        )
        search.conn.commit()
        assert search.search_places(include_contacts=False)[0]["contacts"] == []
