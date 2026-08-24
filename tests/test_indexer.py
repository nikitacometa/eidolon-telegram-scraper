"""Tests for the open-taxonomy entity extractor and packed attribution.

The saving here is real but the failure mode is quiet: a pack that comes back
short, or with an entity attached to the wrong message, corrupts the index
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


def _queue(
    search: SearchDatabase,
    texts: list[str],
    *,
    senders: list[tuple[int, str] | None] | None = None,
) -> list[int]:
    if senders is not None:
        assert len(senders) == len(texts)
    ids = []
    for index, text in enumerate(texts, start=1):
        sender = senders[index - 1] if senders is not None else None
        cur = search.conn.execute(
            """
            INSERT INTO corpus_messages (
                source, chat_id, telegram_msg_id, chat_title, sender_id, sender_name,
                text, date, content_hash
            ) VALUES ('scout', -100, ?, 'Chat', ?, ?, ?, '2026-07-01T10:00:00+00:00', ?)
            """,
            (
                index,
                sender[0] if sender else None,
                sender[1] if sender else None,
                text,
                content_digest(text) or f"h{index}",
            ),
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
            ExtractedEntity,
            ExtractionResult,
            MessageExtraction,
        )

        content = kwargs["messages"][1]["content"]
        ids = [
            int(line.split()[2])
            for line in content.splitlines()
            if line.startswith("--- message ") and line.split()[2].isdigit()
        ]

        def entity(tag: int) -> ExtractedEntity:
            return ExtractedEntity(
                name=f"Venue {tag}",
                aliases=[],
                entity_kind="place",
                access_modes=["visit"],
                descriptor="music bar",
                descriptor_language="en",
                offerings=["live music"],
                city_area="Da Nang",
                evidence=f"evidence {tag}",
                confidence=0.9,
            )

        if kwargs["response_format"] is ExtractionResult:
            # Single-message fallback path: the id is not in the text.
            tag = int(content.split("telegram_msg_id=")[-1]) if "telegram_msg_id=" in content else 0
            parsed: Any = ExtractionResult(entities=[entity(tag)])
        else:
            entries = [
                MessageExtraction(message_id=i, entities=[entity(i)])
                for i in ids
                if i not in self.drop
            ]
            if self.invent is not None:
                entries.append(MessageExtraction(message_id=self.invent, entities=[entity(0)]))
            parsed = BatchExtractionResult(results=entries)

        message = type("M", (), {"parsed": parsed, "refusal": None})()
        choice = type("C", (), {"message": message})()
        usage = type(
            "U",
            (),
            {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": None},
        )()
        return type("R", (), {"choices": [choice], "usage": usage})()


class FakeEntity:
    """A single-message response carrying one open descriptor."""

    def __init__(self, *, name: str, place_type: str, evidence: str) -> None:
        self.name = name
        self.place_type = place_type
        self.evidence = evidence
        self.requests: list[dict[str, Any]] = []
        self.chat = self  # type: ignore[assignment]
        self.completions = self  # type: ignore[assignment]

    async def parse(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        from pipeline.indexer import ExtractedEntity, ExtractionResult

        parsed = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name=self.name,
                    aliases=[],
                    entity_kind="place",
                    access_modes=["visit"],
                    descriptor=self.place_type,
                    descriptor_language="en",
                    offerings=["local service"],
                    city_area="Da Nang",
                    evidence=self.evidence,
                    confidence=0.95,
                )
            ]
        )
        message = type("M", (), {"parsed": parsed, "refusal": None})()
        choice = type("C", (), {"message": message})()
        usage = type(
            "U",
            (),
            {"prompt_tokens": 650, "completion_tokens": 40, "prompt_tokens_details": None},
        )()
        return type("R", (), {"choices": [choice], "usage": usage})()


class FakeNoEntity(FakeEntity):
    """A successful response that keeps a targeted row at ``no_venue``."""

    def __init__(self) -> None:
        super().__init__(name="unused", place_type="other", evidence="unused")

    async def parse(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        from pipeline.indexer import ExtractionResult

        message = type("M", (), {"parsed": ExtractionResult(entities=[]), "refusal": None})()
        choice = type("C", (), {"message": message})()
        usage = type(
            "U",
            (),
            {"prompt_tokens": 650, "completion_tokens": 20, "prompt_tokens_details": None},
        )()
        return type("R", (), {"choices": [choice], "usage": usage})()


class FakeAuthorPolicy(FakePacked):
    """Deterministic stand-in for the measured author/body extraction policy."""

    async def parse(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        from pipeline.indexer import BatchExtractionResult, ExtractedEntity, MessageExtraction

        system = kwargs["messages"][0]["content"]
        content = kwargs["messages"][1]["content"]
        assert "`From:` identifies the author" in system
        assert "never quote `From:` itself" in system

        entries = []
        for block in content.split("--- message ")[1:]:
            id_line, message = block.split("\n", 1)
            message_id = int(id_line.removesuffix(" ---"))
            if "Я барбер, стригу мужчин в Дананге" in message:
                assert "From: @barber_danang (Иван)" in message
                entities = [
                    ExtractedEntity(
                        name="@barber_danang",
                        aliases=["Иван"],
                        entity_kind="person",
                        access_modes=["unknown"],
                        descriptor="барбер",
                        descriptor_language="ru",
                        offerings=["стригу мужчин"],
                        city_area="Da Nang",
                        evidence="Я барбер, стригу мужчин в Дананге",
                        confidence=0.95,
                    )
                ]
            else:
                assert "Сегодня отличная погода" in message
                assert "From: @weather_person (Пётр)" in message
                entities = []
            entries.append(MessageExtraction(message_id=message_id, entities=entities))

        parsed = BatchExtractionResult(results=entries)
        response_message = type("M", (), {"parsed": parsed, "refusal": None})()
        choice = type("C", (), {"message": response_message})()
        usage = type(
            "U",
            (),
            {"prompt_tokens": 800, "completion_tokens": 40, "prompt_tokens_details": None},
        )()
        return type("R", (), {"choices": [choice], "usage": usage})()


TEXTS = [
    f"Событие номер {n} в заведении, живая музыка и открытый микрофон, приходите послушать вечером"
    for n in range(1, 7)
]


def test_entity_schema_is_open_taxonomy_and_strict() -> None:
    from pipeline.indexer import ExtractedEntity, ExtractionResult

    result_schema = ExtractionResult.model_json_schema()
    entity_schema = ExtractedEntity.model_json_schema()

    assert result_schema["required"] == ["entities"]
    assert result_schema["additionalProperties"] is False
    assert entity_schema["additionalProperties"] is False
    assert {"descriptor", "offerings", "entity_kind", "access_modes"} <= set(
        entity_schema["required"]
    )
    assert "place_type" not in entity_schema["properties"]
    assert "event_types" not in entity_schema["properties"]


@pytest.mark.parametrize(
    ("text", "name", "place_type"),
    [
        (
            "Для консультации по анализам рекомендую Lotus Medical Clinic на улице Нгуен Ван Линь, запись по телефону.",
            "Lotus Medical Clinic",
            "clinic",
        ),
        (
            "MacLab Da Nang repair shop чинит MacBook после залития, диагностика и приём техники по будням.",
            "MacLab Da Nang",
            "repair",
        ),
        (
            "В Rose Beauty Salon делают стрижки, маникюр и окрашивание; салон принимает клиентов ежедневно.",
            "Rose Beauty Salon",
            "salon",
        ),
        (
            "Сегодня вечером концерт и открытый микрофон в Corner Music Bar, приходите слушать живую музыку.",
            "Corner Music Bar",
            "bar",
        ),
    ],
)
async def test_medical_service_and_existing_types_reach_the_place_index(
    search: SearchDatabase, text: str, name: str, place_type: str
) -> None:
    from pipeline.indexer import EXTRACTION_PROMPT_VERSION, SYSTEM_PROMPT

    _queue(search, [text])
    client = FakeEntity(name=name, place_type=place_type, evidence=name)

    report = await PlaceExtractor(search, client=client, pack_size=1).run(limit=1)

    assert report["processed"] == 1
    result = search.search_places(name_query=name)
    assert result[0]["place_type"] == place_type
    assert EXTRACTION_PROMPT_VERSION == "entities-v5"
    assert "descriptor" in SYSTEM_PROMPT


async def test_targeted_no_venue_pass_is_a_bounded_snapshot(search: SearchDatabase) -> None:
    ids = _queue(search, TEXTS[:4])
    for corpus_id in ids[:3]:
        search.record_extraction(corpus_id, [], model="places-v2", prompt_version="places-v2")
    client = FakeNoEntity()

    report = await PlaceExtractor(search, client=client, pack_size=1).run(
        limit=5, statuses=["no_venue"]
    )

    assert report["processed"] == 3
    assert len(client.requests) == 3
    pending = search.conn.execute(
        "SELECT status FROM extraction_state WHERE corpus_id = ?", (ids[3],)
    ).fetchone()
    assert pending["status"] == "pending"


async def test_targeted_no_venue_pass_honours_limit(search: SearchDatabase) -> None:
    ids = _queue(search, TEXTS[:4])
    for corpus_id in ids:
        search.record_extraction(corpus_id, [], model="places-v2", prompt_version="places-v2")
    client = FakeNoEntity()

    report = await PlaceExtractor(search, client=client, pack_size=1).run(
        limit=2, statuses=["no_venue"]
    )

    attempts = [
        row["attempts"]
        for row in search.conn.execute(
            "SELECT attempts FROM extraction_jobs WHERE prompt_version='entities-v5' "
            "ORDER BY corpus_id"
        ).fetchall()
    ]
    assert report["processed"] == 2
    assert sorted(attempts) == [0, 0, 1, 1]


async def test_a_pack_costs_one_call_not_one_per_message(search: SearchDatabase) -> None:
    ids = _queue(search, TEXTS)
    client = FakePacked()

    report = await PlaceExtractor(search, client=client, pack_size=6).run(limit=10)

    assert report["processed"] == len(ids)
    assert len(client.requests) == 1, "six messages should ride in one request"
    assert report["calls"] == 1


async def test_author_only_supplies_identity_when_body_establishes_service(
    search: SearchDatabase,
) -> None:
    ids = _queue(
        search,
        ["Я барбер, стригу мужчин в Дананге", "Сегодня отличная погода"],
        senders=[
            (700, "Иван @barber_danang"),
            (701, "Пётр @weather_person"),
        ],
    )
    client = FakeAuthorPolicy()

    report = await PlaceExtractor(search, client=client, pack_size=2).run(limit=2)

    assert report["entities"] == 1
    entities = search.conn.execute(
        "SELECT name, entity_kind FROM places ORDER BY place_id"
    ).fetchall()
    assert [(row["name"], row["entity_kind"]) for row in entities] == [("@barber_danang", "person")]
    states = {
        row["corpus_id"]: row["status"]
        for row in search.conn.execute("SELECT corpus_id, status FROM extraction_state")
    }
    assert states == {ids[0]: "extracted", ids[1]: "no_venue"}
    assert "From: @barber_danang (Иван)" in client.requests[0]["messages"][1]["content"]


async def test_single_message_path_also_carries_author_metadata(search: SearchDatabase) -> None:
    _queue(search, ["Сегодня отличная погода"], senders=[(701, "@weather_person (Пётр)")])
    client = FakeNoEntity()

    await PlaceExtractor(search, client=client, pack_size=1).run(limit=1)

    content = client.requests[0]["messages"][1]["content"]
    assert "\nFrom: @weather_person (Пётр)\n--- message ---\n" in content


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
        for r in search.conn.execute(
            "SELECT corpus_id, status FROM extraction_jobs WHERE prompt_version='entities-v5'"
        )
    }
    assert all(states[i] == "error" for i in ids)
    active_states = {
        r["corpus_id"]: r["status"]
        for r in search.conn.execute("SELECT corpus_id, status FROM extraction_state")
    }
    assert all(active_states[i] == "pending" for i in ids)


async def test_pack_size_one_keeps_batch_instructions_out_of_the_prompt(
    search: SearchDatabase,
) -> None:
    # Packing instructions solve attribution between message ids and do not
    # belong in the single-message path, where they can only confuse the model.
    from pipeline.indexer import BATCH_PROMPT

    _queue(search, TEXTS[:2])
    client = FakePacked()

    await PlaceExtractor(search, client=client, pack_size=1).run(limit=10)

    assert len(client.requests) == 2
    for request in client.requests:
        assert BATCH_PROMPT not in request["messages"][0]["content"]
