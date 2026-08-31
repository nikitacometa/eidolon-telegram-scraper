"""Tests for the corpus value/quality ranking."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from pipeline.housing.rating import rate_listings
from storage.search import SEARCH_SCHEMA_PATH


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(tmp_path / "search.db")
    connection.row_factory = sqlite3.Row
    connection.executescript(SEARCH_SCHEMA_PATH.read_text())
    return connection


def _listing(
    conn: sqlite3.Connection,
    corpus_id: int,
    *,
    price: int | None,
    date: str = "2026-02-05",
    bedrooms: int | None = 2,
    property_type: str | None = "house",
    terrace: int | None = None,
    content_hash: str | None = None,
    chat_id: int = -100777,
    **extra: Any,
) -> None:
    conn.execute(
        """
        INSERT INTO corpus_messages (
            corpus_id, source, chat_id, telegram_msg_id, text, date, content_hash
        ) VALUES (?, 'scout', ?, ?, 'x', ?, ?)
        """,
        (corpus_id, chat_id, corpus_id, date, content_hash or f"h{corpus_id}"),
    )
    conn.execute(
        """
        INSERT INTO housing_listings (
            corpus_id, chat_id, content_hash, posted_at, is_rental_offer,
            is_vehicle_ad, bedrooms, monthly_price_thb, property_type,
            terrace, amenities_json, extractor_version
        ) VALUES (?, ?, ?, ?, 1, 0, ?, ?, ?, ?, ?, 'housing-text-v2')
        """,
        (
            corpus_id,
            chat_id,
            content_hash or f"h{corpus_id}",
            date,
            bedrooms,
            price,
            property_type,
            terrace,
            extra.get("amenities_json"),
        ),
    )
    conn.commit()


def _seed_bucket(conn: sqlite3.Connection, *, start_id: int, prices: list[int]) -> None:
    """Enough same-bucket listings for a percentile to mean something."""
    for offset, price in enumerate(prices):
        _listing(conn, start_id + offset, price=price)


def test_a_below_median_price_ranks_as_a_deal(conn: sqlite3.Connection) -> None:
    _seed_bucket(conn, start_id=100, prices=[30000] * 10)
    _listing(conn, 1, price=21000, terrace=1)

    report = rate_listings(conn, limit=5)

    top = report["listings"][0]
    assert top["corpus_id"] == 1
    assert top["value_discount_pct"] == 30
    assert top["comparable_median"] == 30000
    assert top["comparable_bucket"] == "house/2BR/Q1"


def test_a_crossposted_deal_is_counted_once(conn: sqlite3.Connection) -> None:
    _seed_bucket(conn, start_id=100, prices=[30000] * 10)
    _listing(conn, 1, price=20000, content_hash="same")
    _listing(conn, 2, price=20000, content_hash="same", chat_id=-100888, date="2026-02-06")

    report = rate_listings(conn)

    ids = [item["corpus_id"] for item in report["listings"]]
    assert ids.count(1) + ids.count(2) == 1


def test_a_confirmed_apartment_is_excluded_from_a_house_search(
    conn: sqlite3.Connection,
) -> None:
    """A cheap studio is not a deal on a house search."""
    _seed_bucket(conn, start_id=100, prices=[30000] * 10)
    _listing(conn, 1, price=8000, property_type="apartment")

    report = rate_listings(conn)

    assert all(item["corpus_id"] != 1 for item in report["listings"])
    assert report["excluded_hard_miss"] == 1


def test_a_priceless_listing_is_shown_unranked_not_hidden(
    conn: sqlite3.Connection,
) -> None:
    """ "Цена в личке" may still be the best house on the list."""
    _seed_bucket(conn, start_id=100, prices=[30000] * 10)
    _listing(conn, 1, price=None, terrace=1)

    report = rate_listings(conn)

    match = [item for item in report["listings"] if item["corpus_id"] == 1]
    assert len(match) == 1
    assert match[0]["value_discount_pct"] is None
    assert report["unranked_shown"] >= 1


def test_a_thin_seasonal_bucket_falls_back_rather_than_fakes(
    conn: sqlite3.Connection,
) -> None:
    """A percentile over seven prices is a number that looks like knowledge
    and is not; the ladder widens the bucket instead."""
    # 3 same-season comparables (thin), 10 more in another season.
    _seed_bucket(conn, start_id=100, prices=[30000, 31000, 29000])
    for offset, price in enumerate([32000] * 10):
        _listing(conn, 200 + offset, price=price, date="2026-07-05")
    _listing(conn, 1, price=20000)

    report = rate_listings(conn)

    top = [item for item in report["listings"] if item["corpus_id"] == 1][0]
    assert top["comparable_bucket"] == "house/2BR/all-seasons"
    assert top["comparable_count"] == 14


def test_deals_sort_by_discount_then_quality(conn: sqlite3.Connection) -> None:
    _seed_bucket(conn, start_id=100, prices=[30000] * 10)
    _listing(conn, 1, price=27000)  # 10% off
    _listing(conn, 2, price=21000)  # 30% off

    report = rate_listings(conn, limit=3)

    ids = [item["corpus_id"] for item in report["listings"][:2]]
    assert ids == [2, 1]


def test_dedup_keeps_the_newest_copy_of_a_crosspost(conn: sqlite3.Connection) -> None:
    """The representative of a hash group must be its newest copy — the
    bare-MAX() SQLite guarantee this query leans on, pinned here."""
    _seed_bucket(conn, start_id=100, prices=[30000] * 10)
    _listing(conn, 1, price=20000, content_hash="same", date="2026-01-01", chat_id=-1)
    _listing(conn, 2, price=20000, content_hash="same", date="2026-02-20", chat_id=-2)

    report = rate_listings(conn)

    kept = [item for item in report["listings"] if item["monthly_price_thb"] == 20000]
    assert len(kept) == 1
    assert kept[0]["corpus_id"] == 2
    assert kept[0]["posted_at"] == "2026-02-20"


def test_custom_requirements_change_the_ranking(conn: sqlite3.Connection) -> None:
    """The ranking judges by the owner's ACTIVE revision, not the defaults."""
    _seed_bucket(conn, start_id=100, prices=[30000] * 10)
    _listing(conn, 1, price=8000, property_type="apartment")

    lenient = rate_listings(
        conn,
        requirements={"bedrooms": {"operator": "at_least", "value": 2}},
    )

    assert any(item["corpus_id"] == 1 for item in lenient["listings"])
