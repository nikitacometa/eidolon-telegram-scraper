"""Tests for the historical listing extractor and the price trend."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from pipeline.housing.history import (
    MAX_PLAUSIBLE_THB,
    MIN_LISTINGS_MONTH,
    ListingExtractor,
    price_trend,
)
from storage.search import SEARCH_SCHEMA_PATH


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(tmp_path / "search.db")
    connection.row_factory = sqlite3.Row
    connection.executescript(SEARCH_SCHEMA_PATH.read_text())
    return connection


def _message(conn: sqlite3.Connection, corpus_id: int, text: str, date: str, **kwargs: Any) -> None:
    conn.execute(
        """
        INSERT INTO corpus_messages (
            corpus_id, source, chat_id, telegram_msg_id, text, date, content_hash
        )
        VALUES (?, 'scout', ?, ?, ?, ?, ?)
        """,
        (
            corpus_id,
            kwargs.get("chat_id", -100777),
            corpus_id,
            text,
            date,
            kwargs.get("content_hash", f"hash-{corpus_id}"),
        ),
    )
    conn.commit()


def _listing(
    conn: sqlite3.Connection,
    listing_id: int,
    *,
    date: str,
    price: int | None,
    bedrooms: int | None = 2,
    chat_id: int = -100777,
    content_hash: str | None = None,
) -> None:
    _message(
        conn,
        listing_id,
        "Сдаю дом",
        date,
        chat_id=chat_id,
        content_hash=content_hash or f"h{listing_id}",
    )
    conn.execute(
        """
        INSERT INTO housing_listings (
            corpus_id, chat_id, content_hash, posted_at, is_rental_offer,
            is_vehicle_ad, bedrooms, monthly_price_thb, extractor_version
        )
        VALUES (?, ?, ?, ?, 1, 0, ?, ?, 'housing-text-v1')
        """,
        (listing_id, chat_id, content_hash or f"h{listing_id}", date, bedrooms, price),
    )
    conn.commit()


class FakeOpenAI:
    """Answers a packed extraction request with what the test dictates."""

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = answers
        self.requests: list[str] = []
        self.chat = self

    @property
    def completions(self) -> "FakeOpenAI":
        return self

    async def create(self, **kwargs: Any) -> Any:
        import json
        from types import SimpleNamespace

        self.requests.append(kwargs["messages"][1]["content"])
        payload = json.dumps({"listings": self.answers})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


async def test_seeding_marks_a_chats_archive_as_pending(conn: sqlite3.Connection) -> None:
    _message(conn, 1, "Сдаю дом 2 спальни 25000 бат в месяц", "2026-01-05T10:00:00+00:00")
    _message(conn, 2, "Всем привет, как дела на острове сегодня?", "2026-01-05T11:00:00+00:00")

    seeded = ListingExtractor(conn).seed([-100777])

    assert seeded == 2


async def test_chatter_is_gated_before_any_model_call(conn: sqlite3.Connection) -> None:
    """The archive is where a wasted call is multiplied by thousands."""
    _message(
        conn,
        1,
        "Кто подскажет хорошего стоматолога недалеко от пирса? Спасибо",
        "2026-01-05T10:00:00+00:00",
    )
    extractor = ListingExtractor(conn, client=FakeOpenAI([]))
    extractor.seed([-100777])

    run = await extractor.run(limit=10)

    assert run.gated == 1
    assert run.extracted == 0
    status = conn.execute("SELECT status FROM housing_listing_state").fetchone()
    assert status["status"] == "gated"


async def test_a_listing_becomes_a_row_with_its_price(conn: sqlite3.Connection) -> None:
    _message(
        conn,
        1,
        "Сдаю дом на длительный срок, 2 спальни, 2 ванные, 28000 бат в месяц, Шритану",
        "2026-01-05T10:00:00+00:00",
    )
    client = FakeOpenAI(
        [
            {
                "id": 1,
                "is_rental_offer": True,
                "is_vehicle_ad": False,
                "bedrooms": 2,
                "bathrooms": 2,
                "monthly_price_thb": 28000,
                "price_note": None,
                "tv_present": None,
                "tv_size_class": None,
                "area_raw": "Шритану",
                "evidence_quote": "Сдаю дом на длительный срок",
            }
        ]
    )
    extractor = ListingExtractor(conn, client=client)
    extractor.seed([-100777])

    run = await extractor.run(limit=10)

    assert run.listings == 1
    row = conn.execute("SELECT * FROM housing_listings").fetchone()
    assert row["monthly_price_thb"] == 28000
    assert row["bedrooms"] == 2
    assert row["area_raw"] == "Шритану"


async def test_a_scooter_advertisement_is_not_stored_as_a_listing(
    conn: sqlite3.Connection,
) -> None:
    _message(
        conn,
        1,
        "Сдаю в аренду байк Yamaha, 3500 бат в месяц, отличное состояние",
        "2026-01-05T10:00:00+00:00",
    )
    client = FakeOpenAI(
        [
            {
                "id": 1,
                "is_rental_offer": True,
                "is_vehicle_ad": True,
                "bedrooms": None,
                "bathrooms": None,
                "monthly_price_thb": 3500,
                "price_note": None,
                "tv_present": None,
                "tv_size_class": None,
                "area_raw": None,
                "evidence_quote": "Сдаю в аренду байк",
            }
        ]
    )
    extractor = ListingExtractor(conn, client=client)
    extractor.seed([-100777])

    run = await extractor.run(limit=10)

    assert run.listings == 0
    assert conn.execute("SELECT COUNT(*) FROM housing_listings").fetchone()[0] == 0
    assert conn.execute("SELECT status FROM housing_listing_state").fetchone()[0] == "not_housing"


async def test_a_price_outside_the_plausible_range_is_kept_out_of_the_series(
    conn: sqlite3.Connection,
) -> None:
    """A nightly rate or a phone number must not become a rent observation."""
    _message(
        conn, 1, "Сдаю виллу посуточно, 1200 бат за ночь, 2 спальни", "2026-01-05T10:00:00+00:00"
    )
    client = FakeOpenAI(
        [
            {
                "id": 1,
                "is_rental_offer": True,
                "is_vehicle_ad": False,
                "bedrooms": 2,
                "bathrooms": None,
                "monthly_price_thb": 1200,
                "price_note": None,
                "tv_present": None,
                "tv_size_class": None,
                "area_raw": None,
                "evidence_quote": "1200 бат за ночь",
            }
        ]
    )
    extractor = ListingExtractor(conn, client=client)
    extractor.seed([-100777])

    await extractor.run(limit=10)

    row = conn.execute("SELECT monthly_price_thb, price_note FROM housing_listings").fetchone()
    assert row["monthly_price_thb"] is None
    assert "out of plausible" in str(row["price_note"])


async def test_an_answer_missing_from_the_pack_is_recorded_not_lost(
    conn: sqlite3.Connection,
) -> None:
    """A pack that comes back short must leave a trace, never a silent gap."""
    _message(conn, 1, "Сдаю дом 2 спальни 25000 бат", "2026-01-05T10:00:00+00:00")
    _message(conn, 2, "Сдаю виллу 3 спальни 40000 бат", "2026-01-05T11:00:00+00:00")
    client = FakeOpenAI(
        [
            {
                "id": 1,
                "is_rental_offer": True,
                "is_vehicle_ad": False,
                "bedrooms": 2,
                "bathrooms": None,
                "monthly_price_thb": 25000,
                "price_note": None,
                "tv_present": None,
                "tv_size_class": None,
                "area_raw": None,
                "evidence_quote": "Сдаю дом",
            }
        ]
    )
    extractor = ListingExtractor(conn, client=client)
    extractor.seed([-100777])

    run = await extractor.run(limit=10)

    assert run.errors == 1
    missing = conn.execute(
        "SELECT status, error FROM housing_listing_state WHERE corpus_id = 2"
    ).fetchone()
    assert missing["status"] == "error"
    assert missing["error"] == "missing_from_pack"


# ---------------------------------------------------------------------------
# The trend
# ---------------------------------------------------------------------------


def test_a_thin_month_reports_insufficient_sample_rather_than_a_number(
    conn: sqlite3.Connection,
) -> None:
    """Four listings do not make a median, however much one would like them to."""
    for index in range(4):
        _listing(conn, index + 1, date=f"2026-01-0{index + 1}T10:00:00+00:00", price=20000 + index)

    series = price_trend(conn)

    assert len(series) == 1
    assert series[0]["median_thb"] is None
    assert "insufficient sample" in str(series[0]["note"])
    assert series[0]["n_priced"] == 4


def test_a_full_month_reports_a_median_and_quartiles(conn: sqlite3.Connection) -> None:
    for index in range(MIN_LISTINGS_MONTH):
        _listing(
            conn,
            index + 1,
            date=f"2026-02-{(index % 28) + 1:02d}T10:00:00+00:00",
            price=20000 + index * 1000,
        )

    series = price_trend(conn)

    assert series[0]["period"] == "2026-02"
    assert series[0]["n_priced"] == MIN_LISTINGS_MONTH
    assert series[0]["median_thb"] == 29500
    assert series[0]["p25_thb"] < series[0]["median_thb"] < series[0]["p75_thb"]


def test_a_crossposted_listing_counts_once(conn: sqlite3.Connection) -> None:
    """Whoever posts most widely must not weight the median.

    The same advertisement in three chats is one price, and counting it three
    times would tilt the whole series toward the agencies that spam.
    """
    for index in range(MIN_LISTINGS_MONTH):
        _listing(
            conn,
            index + 1,
            date=f"2026-03-{(index % 28) + 1:02d}T10:00:00+00:00",
            price=30000,
            content_hash="one-and-the-same",
        )

    series = price_trend(conn)

    assert series[0]["n_listings"] == 1
    assert series[0]["median_thb"] is None


def test_the_series_records_how_many_chats_fed_each_bucket(conn: sqlite3.Connection) -> None:
    """A month that changes between runs changed because coverage did."""
    _listing(conn, 1, date="2026-04-01T10:00:00+00:00", price=25000, chat_id=-100777)
    _listing(conn, 2, date="2026-04-02T10:00:00+00:00", price=27000, chat_id=-100888)

    series = price_trend(conn)

    assert series[0]["n_chats_included"] == 2


def test_segmenting_by_bedrooms_only_counts_that_segment(conn: sqlite3.Connection) -> None:
    _listing(conn, 1, date="2026-05-01T10:00:00+00:00", price=25000, bedrooms=2)
    _listing(conn, 2, date="2026-05-02T10:00:00+00:00", price=60000, bedrooms=3)

    two_bed = price_trend(conn, bedrooms="2")

    assert two_bed[0]["n_listings"] == 1
    assert two_bed[0]["n_priced"] == 1


def test_quarterly_buckets_gather_three_months(conn: sqlite3.Connection) -> None:
    _listing(conn, 1, date="2026-01-15T10:00:00+00:00", price=25000)
    _listing(conn, 2, date="2026-02-15T10:00:00+00:00", price=27000)
    _listing(conn, 3, date="2026-03-15T10:00:00+00:00", price=29000)

    series = price_trend(conn, period_kind="quarter")

    assert [entry["period"] for entry in series] == ["2026-Q1"]
    assert series[0]["n_listings"] == 3


def test_persisting_the_series_appends_rather_than_overwrites(conn: sqlite3.Connection) -> None:
    """Yesterday's answer stays readable after today's recompute."""
    _listing(conn, 1, date="2026-06-01T10:00:00+00:00", price=25000)

    price_trend(conn, persist=True)
    price_trend(conn, persist=True)

    rows = conn.execute("SELECT COUNT(*) FROM housing_price_trend").fetchone()[0]
    assert rows == 2


def test_a_listing_priced_above_the_plausible_ceiling_never_reaches_the_series(
    conn: sqlite3.Connection,
) -> None:
    _listing(conn, 1, date="2026-07-01T10:00:00+00:00", price=MAX_PLAUSIBLE_THB + 1)

    series = price_trend(conn)

    # It is still a listing, it simply carries no usable price.
    assert series[0]["n_listings"] == 1
    assert series[0]["n_priced"] == 1
