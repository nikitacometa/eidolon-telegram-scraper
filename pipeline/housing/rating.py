"""Ranking two years of listings: what is good, and what is a deal.

Two scores, deliberately kept apart. Quality is the same weighted preference
score the live matcher computes — one concept, one implementation, so "80%"
means the same thing in an alert and in a corpus ranking. Value compares the
asking price against the median of COMPARABLE listings: same kind of
property, same bedroom count, same season-of-year pooled across both years —
pooling is what stops every low-season listing from looking like a bargain.

Everything is computed at query time from housing_listings. Four thousand
rows rank in milliseconds, and a persisted score would go stale the moment
the owner edits a weight over the bridge; the one thing worth persisting is
the trend series, which has its own table and run ids.

A bucket below MIN_BUCKET is never used: a percentile over seven prices is a
number that looks like knowledge and is not. The ladder falls back to coarser
buckets and finally reports the value as unknown rather than fake it.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass
from typing import Any

from pipeline.housing.history import MAX_PLAUSIBLE_THB, MIN_PLAUSIBLE_THB
from pipeline.housing.requirements import (
    DEFAULT_REQUIREMENTS,
    match_requirements,
)
from storage.housing import Verdict

# Below this many comparable prices a percentile is noise, not knowledge.
MIN_BUCKET = 10


@dataclass(frozen=True, slots=True)
class RatedListing:
    """One deduplicated advertisement with its two scores."""

    corpus_id: int
    chat_id: int
    posted_at: str
    bedrooms: int | None
    property_type: str | None
    monthly_price_thb: int | None
    quality_score: int
    verdict: str
    value_percentile: int | None
    value_discount_pct: int | None
    comparable_bucket: str | None
    comparable_median: int | None
    comparable_count: int | None
    amenities: list[str]
    area_raw: str | None
    evidence_quote: str | None

    def as_dict(self) -> dict[str, Any]:
        """Stable CLI/MCP-friendly form."""
        return {
            "corpus_id": self.corpus_id,
            "chat_id": self.chat_id,
            "posted_at": self.posted_at,
            "bedrooms": self.bedrooms,
            "property_type": self.property_type,
            "monthly_price_thb": self.monthly_price_thb,
            "quality_score": self.quality_score,
            "verdict": self.verdict,
            "value_percentile": self.value_percentile,
            "value_discount_pct": self.value_discount_pct,
            "comparable_bucket": self.comparable_bucket,
            "comparable_median": self.comparable_median,
            "comparable_count": self.comparable_count,
            "amenities": self.amenities,
            "area_raw": self.area_raw,
            "evidence_quote": self.evidence_quote,
        }


def _season(posted_at: str) -> str:
    """Season-of-year bucket, pooled across years.

    Q1 is measured 40k against Q2's 21k on this corpus; comparing a January
    listing against the whole year calls every high-season price a rip-off
    and every May price a steal. Pooling by quarter keeps the comparison
    honest while letting both years feed the same bucket.
    """
    try:
        month = int(posted_at[5:7])
    except (ValueError, IndexError):
        return "Q?"
    return f"Q{(month - 1) // 3 + 1}"


def _type_bucket(property_type: str | None) -> str:
    """house / apartment / untyped: room and hotel price like neither."""
    if property_type in ("house", "apartment", "room", "hotel"):
        return str(property_type)
    return "untyped"


def _facts_from_listing(row: sqlite3.Row) -> dict[str, Any]:
    """The live matcher's facts shape, built from an archive row.

    Sources are 'text' wherever a value exists: the archive is text-only, and
    the matcher needs a source to distinguish a stated wrong type (rejects)
    from a photographed one (does not).
    """
    facts: dict[str, Any] = {
        "is_rental_offer": row["is_rental_offer"],
        "bedrooms": row["bedrooms"],
        "bedrooms_source": "text" if row["bedrooms"] is not None else "unknown",
        "bathrooms": row["bathrooms"],
        "bathrooms_source": "text" if row["bathrooms"] is not None else "unknown",
        "monthly_price_thb": row["monthly_price_thb"],
        "price_source": "text" if row["monthly_price_thb"] is not None else "unknown",
        "tv_present": row["tv_present"],
        "tv_size_class": row["tv_size_class"],
        "tv_source": "text" if row["tv_present"] is not None or row["tv_size_class"] else "unknown",
        "property_type": row["property_type"],
        "property_type_source": "text" if row["property_type"] is not None else "unknown",
        "terrace": row["terrace"],
        "terrace_source": "text" if row["terrace"] else "unknown",
        "private_setting": row["private_setting"],
        "nature_setting": row["nature_setting"],
    }
    return facts


def _amenity_names(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    return sorted(name for name, value in parsed.items() if value is True)


def _price_buckets(conn: sqlite3.Connection) -> dict[tuple[str, int, str], list[int]]:
    """Every plausible price in the corpus, grouped by its comparables key.

    Built over the WHOLE corpus regardless of the ranking window: a listing
    from last week is judged against two years of comparable prices, not
    against the handful posted alongside it.
    """
    buckets: dict[tuple[str, int, str], list[int]] = {}
    for row in conn.execute(
        """
        SELECT property_type, bedrooms, monthly_price_thb, posted_at,
               content_hash
        FROM housing_listings
        WHERE is_rental_offer = 1 AND is_vehicle_ad = 0
          AND monthly_price_thb BETWEEN ? AND ?
        GROUP BY COALESCE(content_hash, corpus_id)
        """,
        (MIN_PLAUSIBLE_THB, MAX_PLAUSIBLE_THB),
    ):
        if row["bedrooms"] is None:
            continue
        key = (
            _type_bucket(row["property_type"]),
            int(row["bedrooms"]),
            _season(str(row["posted_at"])),
        )
        buckets.setdefault(key, []).append(int(row["monthly_price_thb"]))
    return buckets


def _comparables(
    buckets: dict[tuple[str, int, str], list[int]],
    *,
    type_bucket: str,
    bedrooms: int,
    season: str,
) -> tuple[str, list[int]] | None:
    """Walk the fallback ladder to the first bucket wide enough to trust."""
    seasonal = buckets.get((type_bucket, bedrooms, season), [])
    if len(seasonal) >= MIN_BUCKET:
        return f"{type_bucket}/{bedrooms}BR/{season}", seasonal
    all_time = [
        price
        for (t, b, _s), prices in buckets.items()
        for price in prices
        if t == type_bucket and b == bedrooms
    ]
    if len(all_time) >= MIN_BUCKET:
        return f"{type_bucket}/{bedrooms}BR/all-seasons", all_time
    by_bedrooms = [
        price for (_t, b, _s), prices in buckets.items() for price in prices if b == bedrooms
    ]
    if len(by_bedrooms) >= MIN_BUCKET:
        return f"any-type/{bedrooms}BR/all-seasons", by_bedrooms
    return None


def rate_listings(
    conn: sqlite3.Connection,
    *,
    requirements: dict[str, Any] | None = None,
    bedrooms: int | None = None,
    days: int | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Rank deduplicated listings by value, best deals first.

    Listings whose hard criteria are VIOLATED are excluded from the ranking —
    a cheap studio is not a deal on a house search — but a listing missing
    its price still appears (value unknown) rather than vanishing: it may be
    the best property on the list, and its price is one question away.
    """
    requirements = requirements or DEFAULT_REQUIREMENTS
    buckets = _price_buckets(conn)

    where = ["is_rental_offer = 1", "is_vehicle_ad = 0"]
    params: list[Any] = []
    if bedrooms is not None:
        where.append("bedrooms = ?")
        params.append(bedrooms)
    if days is not None:
        # A negative window would become a future cutoff and silently return
        # nothing at all.
        where.append("posted_at >= datetime('now', ?)")
        params.append(f"-{max(0, int(days))} days")
    rows = conn.execute(
        f"""
        -- The bare MAX() is load-bearing: with exactly one min/max
        -- aggregate, SQLite documents that the non-aggregated columns come
        -- from the row that produced the maximum, so each crosspost group
        -- yields its NEWEST copy (probed, not assumed).
        SELECT *, MAX(posted_at) AS latest_posted_at
        FROM housing_listings
        WHERE {" AND ".join(where)}
        GROUP BY COALESCE(content_hash, corpus_id)
        ORDER BY posted_at DESC
        """,  # noqa: S608 - clauses are fixed literals, values are bound
        params,
    ).fetchall()

    rated: list[RatedListing] = []
    skipped_hard_miss = 0
    for row in rows:
        facts = _facts_from_listing(row)
        result = match_requirements(facts, requirements)
        if result.verdict is Verdict.HARD_MISS:
            skipped_hard_miss += 1
            continue

        price = row["monthly_price_thb"]
        percentile = discount = median = count = None
        bucket_name = None
        if (
            isinstance(price, int)
            and MIN_PLAUSIBLE_THB <= price <= MAX_PLAUSIBLE_THB
            and row["bedrooms"] is not None
        ):
            found = _comparables(
                buckets,
                type_bucket=_type_bucket(row["property_type"]),
                bedrooms=int(row["bedrooms"]),
                season=_season(str(row["posted_at"])),
            )
            if found is not None:
                bucket_name, prices = found
                below = sum(1 for value in prices if value < price)
                percentile = round(100 * below / len(prices))
                median = int(statistics.median(prices))
                discount = round(100 * (median - price) / median)
                count = len(prices)

        rated.append(
            RatedListing(
                corpus_id=int(row["corpus_id"]),
                chat_id=int(row["chat_id"]),
                posted_at=str(row["posted_at"]),
                bedrooms=row["bedrooms"],
                property_type=row["property_type"],
                monthly_price_thb=price,
                quality_score=result.preference_score,
                verdict=result.verdict.value,
                value_percentile=percentile,
                value_discount_pct=discount,
                comparable_bucket=bucket_name,
                comparable_median=median,
                comparable_count=count,
                amenities=_amenity_names(row["amenities_json"]),
                area_raw=row["area_raw"],
                evidence_quote=row["evidence_quote"],
            )
        )

    # Best deals first: priced listings by discount, then quality; unpriced
    # ones follow, ranked by quality alone.
    rated.sort(
        key=lambda item: (
            item.value_discount_pct is None,
            -(item.value_discount_pct or 0),
            -item.quality_score,
        )
    )
    return {
        "listings": [item.as_dict() for item in rated[:limit]],
        "considered": len(rows),
        "excluded_hard_miss": skipped_hard_miss,
        "unranked_shown": sum(1 for item in rated[:limit] if item.value_discount_pct is None),
    }
