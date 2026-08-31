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


def _answer(corpus_id: int, **overrides: Any) -> dict[str, Any]:
    """A complete v2 answer for one packed advertisement."""
    answer: dict[str, Any] = {
        "id": corpus_id,
        "is_rental_offer": True,
        "is_vehicle_ad": False,
        "bedrooms": 2,
        "bathrooms": None,
        "monthly_price_thb": 30000,
        "price_note": None,
        "tv_present": None,
        "tv_size_class": None,
        "property_type": "house",
        "terrace": True,
        "private_setting": None,
        "nature_setting": True,
        "amenities": {
            "pool": True,
            "aircon": None,
            "kitchen": None,
            "wifi": None,
            "sea_view": None,
            "parking": None,
            "hot_water": None,
            "washing_machine": None,
        },
        "area_raw": None,
        "evidence_quote": None,
    }
    answer.update(overrides)
    return answer


async def test_v2_fields_land_in_the_listing_row(conn: sqlite3.Connection) -> None:
    _message(conn, 1, "Сдаю дом с террасой и бассейном, 2 спальни, 30000 бат", "2026-01-05")
    extractor = ListingExtractor(conn, client=FakeOpenAI([_answer(1)]))
    extractor.seed([-100777])

    await extractor.run(limit=10)

    row = conn.execute("SELECT * FROM housing_listings WHERE corpus_id = 1").fetchone()
    assert row["property_type"] == "house"
    assert row["terrace"] == 1
    assert row["nature_setting"] == 1
    assert row["private_setting"] is None
    assert '"pool": true' in row["amenities_json"]


async def test_a_fabricated_apartment_claim_is_guarded_in_the_archive_too(
    conn: sqlite3.Connection,
) -> None:
    """The archive path must carry the same guard as the live one: a class
    guard placed in one of its habitats leaves the others leaking."""
    _message(conn, 1, "Сдаю жильё 2 спальни 30000 бат", "2026-01-05")
    extractor = ListingExtractor(conn, client=FakeOpenAI([_answer(1, property_type="apartment")]))
    extractor.seed([-100777])

    await extractor.run(limit=10)

    row = conn.execute("SELECT property_type FROM housing_listings WHERE corpus_id = 1").fetchone()
    assert row["property_type"] is None


async def test_reseed_returns_only_stale_extractions_to_the_queue(
    conn: sqlite3.Connection,
) -> None:
    """A version bump re-works extracted and errored rows; the gated and
    not_housing verdicts stand — that classification did not change."""
    _listing(conn, 1, date="2026-01-05", price=30000)  # v1 listing
    conn.execute(
        "INSERT INTO housing_listing_state (corpus_id, status, extractor_version)"
        " VALUES (1, 'extracted', 'housing-text-v1')"
    )
    _message(conn, 2, "не листинг", "2026-01-05")
    conn.execute(
        "INSERT INTO housing_listing_state (corpus_id, status, extractor_version)"
        " VALUES (2, 'not_housing', 'housing-text-v1')"
    )
    _message(conn, 3, "ошибка", "2026-01-05")
    conn.execute(
        "INSERT INTO housing_listing_state (corpus_id, status, extractor_version, error)"
        " VALUES (3, 'error', 'housing-text-v1', 'pack_failed')"
    )
    # A listing already extracted under the CURRENT version must stand: a
    # reseed that re-queues it would re-pay the whole corpus on every run.
    from pipeline.housing.extractor import EXTRACTOR_VERSION

    _message(conn, 4, "Сдаю дом 2 спальни 30000", "2026-01-05")
    conn.execute(
        "INSERT INTO housing_listings (corpus_id, chat_id, content_hash, posted_at,"
        " is_rental_offer, is_vehicle_ad, extractor_version)"
        " VALUES (4, -100777, 'h4', '2026-01-05', 1, 0, ?)",
        (EXTRACTOR_VERSION,),
    )
    conn.execute(
        "INSERT INTO housing_listing_state (corpus_id, status, extractor_version)"
        " VALUES (4, 'extracted', ?)",
        (EXTRACTOR_VERSION,),
    )
    conn.commit()

    reseeded = ListingExtractor(conn).reseed_stale()

    assert reseeded == 2
    states = dict(conn.execute("SELECT corpus_id, status FROM housing_listing_state").fetchall())
    assert states == {1: "pending", 2: "not_housing", 3: "pending", 4: "extracted"}


async def test_a_crosspost_is_copied_from_its_donor_without_a_model_call(
    conn: sqlite3.Connection,
) -> None:
    """22.2% of the corpus shares its text with another row; one model call
    per distinct text, and the duplicates inherit the answer."""
    text = "Сдаю дом с террасой, 2 спальни, 30000 бат"
    _message(conn, 1, text, "2026-01-05", chat_id=-100777, content_hash="same")
    _message(conn, 2, text, "2026-01-06", chat_id=-100888, content_hash="same")
    client = FakeOpenAI([_answer(1)])
    extractor = ListingExtractor(conn, client=client)
    extractor.seed([-100777, -100888])

    run = await extractor.run(limit=10)

    assert run.listings == 2
    # Exactly one request went to the model: the duplicate copied the donor.
    assert len(client.requests) == 1
    copy = conn.execute("SELECT * FROM housing_listings WHERE corpus_id = 2").fetchone()
    assert copy["terrace"] == 1
    assert copy["chat_id"] == -100888
    assert copy["posted_at"] == "2026-01-06"
    states = {
        row["corpus_id"]: row["status"]
        for row in conn.execute("SELECT corpus_id, status FROM housing_listing_state")
    }
    assert states == {1: "extracted", 2: "extracted"}


async def test_a_not_housing_crosspost_settles_its_duplicates_too(
    conn: sqlite3.Connection,
) -> None:
    text = "Продаю виллу на Пангане, 15 млн бат, звоните"
    _message(conn, 1, text, "2026-01-05", content_hash="same")
    _message(conn, 2, text, "2026-01-06", chat_id=-100888, content_hash="same")
    extractor = ListingExtractor(conn, client=FakeOpenAI([_answer(1, is_rental_offer=False)]))
    extractor.seed([-100777, -100888])

    await extractor.run(limit=10)

    states = {
        row["corpus_id"]: row["status"]
        for row in conn.execute("SELECT corpus_id, status FROM housing_listing_state")
    }
    assert states == {1: "not_housing", 2: "not_housing"}
