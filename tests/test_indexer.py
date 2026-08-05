"""Tests for the venue extractor — specifically, packing several messages per call.

The saving here is real but the failure mode is quiet: a pack that comes back
short, or with a venue attached to the wrong message, corrupts the venue index
without any error surfacing. Everything below is about that.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pipeline.indexer import PlaceExtractor
from storage.search import SearchDatabase, content_digest


@pytest.fixture
def search(tmp_path: Path) -> Iterator[SearchDatabase]:
    db = SearchDatabase(tmp_path / "search.db")
    db.connect()
    yield db
    db.close()


def _queue(search: SearchDatabase, texts: list[str]) -> list[int]:
    ids = []
    for index, text in enumerate(texts, start=1):
        cur = search.conn.execute(
            """
            INSERT INTO corpus_messages (source, chat_id, telegram_msg_id, chat_title, text,
                                         date, content_hash)
            VALUES ('scout', -100, ?, 'Chat', ?, '2026-07-01T10:00:00+00:00', ?)
            """,
            (index, text, content_digest(text) or f"h{index}"),
        )
        ids.append(int(cur.lastrowid or 0))
    search.conn.commit()
    search._seed_extraction_state()
    return ids


class FakePacked:
    """An OpenAI client that answers packed extraction requests."""

    def __init__(self, *, drop: set[int] | None = None, invent: int | None = None) -> None:
        self.drop = drop or set()
        self.invent = invent
        self.requests: list[dict[str, Any]] = []
        self.chat = self  # type: ignore[assignment]
        self.completions = self  # type: ignore[assignment]

    async def parse(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        from pipeline.indexer import (
            BatchExtractionResult,
            ExtractedPlace,
            ExtractionResult,
            MessageExtraction,
        )

        content = kwargs["messages"][1]["content"]
        ids = [
            int(line.split()[2])
            for line in content.splitlines()
            if line.startswith("--- message ") and line.split()[2].isdigit()
        ]

        def place(tag: int) -> ExtractedPlace:
            return ExtractedPlace(
                name=f"Venue {tag}",
                place_type="bar",
                city_area="Da Nang",
                event_types=["live_music"],
                evidence=f"evidence {tag}",
                confidence=0.9,
            )

        if kwargs["response_format"] is ExtractionResult:
            # Single-message fallback path: the id is not in the text.
            tag = int(content.split("telegram_msg_id=")[-1]) if "telegram_msg_id=" in content else 0
            parsed: Any = ExtractionResult(places=[place(tag)])
        else:
            entries = [
                MessageExtraction(message_id=i, places=[place(i)])
                for i in ids
                if i not in self.drop
            ]
            if self.invent is not None:
                entries.append(MessageExtraction(message_id=self.invent, places=[place(0)]))
            parsed = BatchExtractionResult(results=entries)

        message = type("M", (), {"parsed": parsed, "refusal": None})()
        choice = type("C", (), {"message": message})()
        usage = type(
            "U",
            (),
            {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": None},
        )()
        return type("R", (), {"choices": [choice], "usage": usage})()


TEXTS = [
    f"Событие номер {n} в заведении, живая музыка и открытый микрофон, приходите послушать вечером"
    for n in range(1, 7)
]


async def test_a_pack_costs_one_call_not_one_per_message(search: SearchDatabase) -> None:
    ids = _queue(search, TEXTS)
    client = FakePacked()

    report = await PlaceExtractor(search, client=client, pack_size=6).run(limit=10)

    assert report["processed"] == len(ids)
    assert len(client.requests) == 1, "six messages should ride in one request"
    assert report["calls"] == 1


async def test_every_venue_lands_on_the_message_that_named_it(search: SearchDatabase) -> None:
    ids = _queue(search, TEXTS)

    await PlaceExtractor(search, client=FakePacked(), pack_size=6).run(limit=10)

    rows = search.conn.execute(
        """
        SELECT pm.corpus_id, p.name FROM place_mentions pm JOIN places p USING(place_id)
         ORDER BY pm.corpus_id
        """
    ).fetchall()
    assert [(r["corpus_id"], r["name"]) for r in rows] == [(i, f"Venue {i}") for i in ids]


async def test_ids_missing_from_the_answer_are_re_asked_not_dropped(
    search: SearchDatabase,
) -> None:
    ids = _queue(search, TEXTS)
    dropped = {ids[1], ids[4]}
    client = FakePacked(drop=dropped)

    await PlaceExtractor(search, client=client, pack_size=6).run(limit=10)

    settled = {
        r["corpus_id"]: r["status"]
        for r in search.conn.execute("SELECT corpus_id, status FROM extraction_state")
    }
    assert all(settled[i] != "pending" for i in ids), "a short pack must not strand messages"
    assert len(client.requests) == 1 + len(dropped), "the missing ids get their own calls"


async def test_an_invented_id_is_not_recorded(search: SearchDatabase) -> None:
    ids = _queue(search, TEXTS)
    ghost = max(ids) + 999
    client = FakePacked(invent=ghost)

    await PlaceExtractor(search, client=client, pack_size=6).run(limit=10)

    seen = {r["corpus_id"] for r in search.conn.execute("SELECT corpus_id FROM place_mentions")}
    assert ghost not in seen
    assert seen == set(ids)


async def test_a_failed_pack_settles_every_message_in_it(search: SearchDatabase) -> None:
    ids = _queue(search, TEXTS)

    class Boom(FakePacked):
        async def parse(self, **kwargs: Any) -> Any:
            raise RuntimeError("provider exploded")

    await PlaceExtractor(search, client=Boom(), pack_size=6).run(limit=10)

    states = {
        r["corpus_id"]: r["status"]
        for r in search.conn.execute("SELECT corpus_id, status FROM extraction_state")
    }
    assert all(states[i] == "error" for i in ids)


async def test_pack_size_one_keeps_the_original_prompt(search: SearchDatabase) -> None:
    # The single-message prompt must stay byte-identical to the one every
    # already-extracted row was produced with, or a re-run silently changes
    # answers rather than costs.
    from pipeline.indexer import BATCH_PROMPT

    _queue(search, TEXTS[:2])
    client = FakePacked()

    await PlaceExtractor(search, client=client, pack_size=1).run(limit=10)

    assert len(client.requests) == 2
    for request in client.requests:
        assert BATCH_PROMPT not in request["messages"][0]["content"]
