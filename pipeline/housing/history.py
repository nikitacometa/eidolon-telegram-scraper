"""Reading a year of history into listings, and listings into a price trend.

The live path answers "should he be told about this one". This answers "what
has the island been charging", which is a different job with different
economics: tens of thousands of archived messages, no deadline, and a bill
that only makes sense packed twenty to a request.

Everything here runs from index_cli.py against the derived search index. The
daemon never writes these tables — it owns the two live stores and this file
is a projection of them, rebuildable by re-running the extractor.

The trend is deliberately hard to publish. A median over four listings is a
number that looks like knowledge and is not, so a bucket below its threshold
returns "insufficient sample" instead. And because chats are still being
joined, each row records how many chats fed it: a month that changes between
two runs changed because coverage did, and that has to be visible rather than
silently overwritten.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import statistics
from dataclasses import dataclass
from typing import Any

from config.settings import settings
from pipeline.housing.extractor import EXTRACTOR_VERSION, SYSTEM_PROMPT
from pipeline.housing.gate import could_be_housing

logger = logging.getLogger(__name__)

# Packed twenty to a request, the same figure the venue extractor measured:
# below that the fixed system prompt dominates the bill, above it the model
# starts losing track of which answer belongs to which message.
DEFAULT_PACK_SIZE = 20

BATCH_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

You are given several advertisements at once, each with an id. Answer with one
object per id, in the `listings` array, and never merge two advertisements into
one answer or invent an id that was not given to you.
"""
)

BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "listings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "is_rental_offer": {"type": "boolean"},
                    "is_vehicle_ad": {"type": "boolean"},
                    "bedrooms": {"type": ["integer", "null"]},
                    "bathrooms": {"type": ["integer", "null"]},
                    "monthly_price_thb": {"type": ["integer", "null"]},
                    "price_note": {"type": ["string", "null"]},
                    "tv_present": {"type": ["boolean", "null"]},
                    "tv_size_class": {
                        "type": ["string", "null"],
                        "enum": ["none", "small", "medium", "large", "unclear", None],
                    },
                    "area_raw": {"type": ["string", "null"]},
                    "evidence_quote": {"type": ["string", "null"]},
                },
                "required": [
                    "id",
                    "is_rental_offer",
                    "is_vehicle_ad",
                    "bedrooms",
                    "bathrooms",
                    "monthly_price_thb",
                    "price_note",
                    "tv_present",
                    "tv_size_class",
                    "area_raw",
                    "evidence_quote",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["listings"],
    "additionalProperties": False,
}

# A monthly bucket needs this many distinct listings before a median means
# anything; a quarter, being three times the window, needs fewer per month.
MIN_LISTINGS_MONTH = 20
MIN_LISTINGS_QUARTER = 10
# Rent outside this range is not rent: a nightly rate quoted as monthly, a
# deposit, a phone number that parsed as a price, or a villa sale.
MIN_PLAUSIBLE_THB = 3_000
MAX_PLAUSIBLE_THB = 500_000


@dataclass(frozen=True, slots=True)
class ExtractionRun:
    """What one pass over the backlog did."""

    scanned: int
    gated: int
    extracted: int
    listings: int
    errors: int

    def as_dict(self) -> dict[str, int]:
        """Stable CLI-friendly form."""
        return {
            "scanned": self.scanned,
            "gated": self.gated,
            "extracted": self.extracted,
            "listings": self.listings,
            "errors": self.errors,
        }


class ListingExtractor:
    """Turns archived housing-chat messages into structured listings."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        client: Any | None = None,
        model: str | None = None,
        pack_size: int = DEFAULT_PACK_SIZE,
        concurrency: int = 4,
    ) -> None:
        self._conn = conn
        self._client = client
        self._model = model or settings.extraction_model
        self._pack_size = max(1, pack_size)
        self._concurrency = max(1, concurrency)

    def _require_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    def seed(self, chat_ids: list[int]) -> int:
        """Mark every archived message in these chats as awaiting extraction."""
        if not chat_ids:
            return 0
        placeholders = ",".join("?" * len(chat_ids))
        cursor = self._conn.execute(
            f"""
            INSERT INTO housing_listing_state (corpus_id, status, extractor_version)
            SELECT corpus_id, 'pending', ?
            FROM corpus_messages
            WHERE chat_id IN ({placeholders})
            ON CONFLICT(corpus_id) DO NOTHING
            """,  # noqa: S608 - placeholders are generated, ids are bound
            [EXTRACTOR_VERSION, *chat_ids],
        )
        self._conn.commit()
        return cursor.rowcount or 0

    async def run(self, *, limit: int = 500) -> ExtractionRun:
        """Extract up to ``limit`` pending messages, oldest first."""
        rows = self._conn.execute(
            """
            SELECT s.corpus_id, m.chat_id, m.text, m.date, m.content_hash
            FROM housing_listing_state s
            JOIN corpus_messages m ON m.corpus_id = s.corpus_id
            WHERE s.status = 'pending'
            ORDER BY s.corpus_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        if not rows:
            return ExtractionRun(0, 0, 0, 0, 0)

        # The same lexical gate the live path uses on general chats, applied
        # here to every chat: a year of archive is where a wasted call is
        # multiplied by thousands, and what it drops was measured.
        gated = [row for row in rows if not could_be_housing(row["text"])]
        candidates = [row for row in rows if could_be_housing(row["text"])]
        for row in gated:
            self._settle(int(row["corpus_id"]), "gated")
        self._conn.commit()

        packs = [
            candidates[index : index + self._pack_size]
            for index in range(0, len(candidates), self._pack_size)
        ]
        semaphore = asyncio.Semaphore(self._concurrency)

        async def one(pack: list[sqlite3.Row]) -> tuple[list[sqlite3.Row], dict[int, Any] | None]:
            async with semaphore:
                return pack, await self._extract_pack(pack)

        results = await asyncio.gather(*(one(pack) for pack in packs), return_exceptions=True)

        extracted = listings = errors = 0
        for outcome in results:
            if isinstance(outcome, BaseException):
                logger.warning("Listing pack failed: %s", outcome)
                errors += 1
                continue
            pack, answers = outcome
            if answers is None:
                for row in pack:
                    self._settle(int(row["corpus_id"]), "error", error="pack_failed")
                errors += len(pack)
                continue
            for row in pack:
                corpus_id = int(row["corpus_id"])
                answer = answers.get(corpus_id)
                if answer is None:
                    # The pack came back without this id. Left pending, it is
                    # retried on the next pass rather than silently lost.
                    self._settle(corpus_id, "error", error="missing_from_pack")
                    errors += 1
                    continue
                extracted += 1
                if answer.get("is_rental_offer") and not answer.get("is_vehicle_ad"):
                    self._store_listing(row, answer)
                    listings += 1
                    self._settle(corpus_id, "extracted")
                else:
                    self._settle(corpus_id, "not_housing")
        self._conn.commit()
        return ExtractionRun(len(rows), len(gated), extracted, listings, errors)

    async def _extract_pack(self, pack: list[sqlite3.Row]) -> dict[int, Any] | None:
        """One request carrying several advertisements."""
        payload = "\n\n".join(
            f"--- id: {int(row['corpus_id'])} ---\n{str(row['text'])[:3000]}" for row in pack
        )
        try:
            response = await self._require_client().chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "housing_listings",
                        "schema": BATCH_SCHEMA,
                        "strict": True,
                    },
                },
                timeout=settings.llm_timeout_seconds * 3,
            )
        except Exception as error:
            logger.warning("Listing extraction failed: %s", type(error).__name__)
            return None

        content = (response.choices[0].message.content or "").strip()
        if not content:
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        answers = {}
        for item in parsed.get("listings", []):
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                answers[int(item["id"])] = item
        return answers

    def _store_listing(self, row: sqlite3.Row, answer: dict[str, Any]) -> None:
        """Write one extracted advertisement."""
        price = answer.get("monthly_price_thb")
        if isinstance(price, bool) or not isinstance(price, int):
            price = None
        elif not MIN_PLAUSIBLE_THB <= price <= MAX_PLAUSIBLE_THB:
            # Out-of-range numbers are the parser's failures, not the market's:
            # a nightly rate, a deposit, a phone number. Kept as a note so the
            # listing is still findable, dropped from the price series.
            answer["price_note"] = f"{price} (out of plausible monthly range)"
            price = None
        self._conn.execute(
            """
            INSERT INTO housing_listings (
                corpus_id, chat_id, content_hash, posted_at,
                is_rental_offer, is_vehicle_ad, bedrooms, bathrooms,
                monthly_price_thb, price_note, tv_present, tv_size_class,
                area_raw, evidence_quote, extractor_version
            )
            VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(corpus_id) DO UPDATE SET
                bedrooms = excluded.bedrooms,
                bathrooms = excluded.bathrooms,
                monthly_price_thb = excluded.monthly_price_thb,
                price_note = excluded.price_note,
                tv_present = excluded.tv_present,
                tv_size_class = excluded.tv_size_class,
                area_raw = excluded.area_raw,
                evidence_quote = excluded.evidence_quote,
                extractor_version = excluded.extractor_version,
                extracted_at = CURRENT_TIMESTAMP
            """,
            (
                int(row["corpus_id"]),
                int(row["chat_id"]),
                row["content_hash"],
                str(row["date"]),
                1 if answer.get("is_vehicle_ad") else 0,
                _clean_int(answer.get("bedrooms")),
                _clean_int(answer.get("bathrooms")),
                price,
                answer.get("price_note"),
                _clean_bool(answer.get("tv_present")),
                answer.get("tv_size_class"),
                answer.get("area_raw"),
                answer.get("evidence_quote"),
                EXTRACTOR_VERSION,
            ),
        )

    def _settle(self, corpus_id: int, status: str, *, error: str | None = None) -> None:
        self._conn.execute(
            """
            UPDATE housing_listing_state
            SET status = ?, attempts = attempts + 1,
                attempted_at = CURRENT_TIMESTAMP, error = ?
            WHERE corpus_id = ?
            """,
            (status, error, corpus_id),
        )


def _clean_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _clean_bool(value: Any) -> int | None:
    return None if not isinstance(value, bool) else int(value)


def price_trend(
    conn: sqlite3.Connection,
    *,
    period_kind: str = "month",
    bedrooms: str = "all",
    persist: bool = False,
) -> list[dict[str, Any]]:
    """Aggregate listings into a defensible price series.

    Crossposts are counted once: the same advertisement in three chats is one
    price observation, and counting it three times would weight whoever posts
    most widely. A bucket without enough distinct listings is reported as
    insufficient rather than as a number.
    """
    if period_kind not in {"month", "quarter"}:
        raise ValueError("period_kind must be 'month' or 'quarter'")

    rows = conn.execute(
        """
        SELECT posted_at, monthly_price_thb, bedrooms, chat_id,
               COALESCE(content_hash, 'listing:' || listing_id) AS dedup_key
        FROM housing_listings
        WHERE is_rental_offer = 1 AND is_vehicle_ad = 0
        ORDER BY posted_at
        """
    ).fetchall()

    buckets: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        period = _period_of(str(row["posted_at"]), period_kind)
        if period is None:
            continue
        if bedrooms != "all" and _bedroom_bucket(row["bedrooms"]) != bedrooms:
            continue
        key = (period, str(row["dedup_key"]))
        if key in seen:
            continue
        seen.add(key)
        bucket = buckets.setdefault(period, {"prices": [], "n_listings": 0, "chats": set()})
        bucket["n_listings"] += 1
        bucket["chats"].add(int(row["chat_id"]))
        price = row["monthly_price_thb"]
        if isinstance(price, int):
            bucket["prices"].append(price)

    minimum = MIN_LISTINGS_MONTH if period_kind == "month" else MIN_LISTINGS_QUARTER
    series: list[dict[str, Any]] = []
    for period in sorted(buckets):
        bucket = buckets[period]
        prices = sorted(bucket["prices"])
        enough = len(prices) >= minimum
        entry: dict[str, Any] = {
            "period": period,
            "period_kind": period_kind,
            "bedrooms_bucket": bedrooms,
            "n_listings": bucket["n_listings"],
            "n_priced": len(prices),
            "n_chats_included": len(bucket["chats"]),
            "median_thb": statistics.median(prices) if enough else None,
            "p25_thb": _quantile(prices, 0.25) if enough else None,
            "p75_thb": _quantile(prices, 0.75) if enough else None,
            "note": None
            if enough
            else f"insufficient sample ({len(prices)} priced, need {minimum})",
        }
        series.append(entry)

    if persist:
        run_id = int(
            conn.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM housing_price_trend").fetchone()[
                0
            ]
        )
        conn.executemany(
            """
            INSERT INTO housing_price_trend (
                run_id, period_kind, period, bedrooms_bucket, n_listings, n_priced,
                median_thb, p25_thb, p75_thb, n_chats_included
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    entry["period_kind"],
                    entry["period"],
                    entry["bedrooms_bucket"],
                    entry["n_listings"],
                    entry["n_priced"],
                    entry["median_thb"],
                    entry["p25_thb"],
                    entry["p75_thb"],
                    entry["n_chats_included"],
                )
                for entry in series
            ],
        )
        conn.commit()
    return series


def _period_of(timestamp: str, period_kind: str) -> str | None:
    """Bucket a stored timestamp, tolerating both formats the corpus holds."""
    if len(timestamp) < 7:
        return None
    year, month = timestamp[:4], timestamp[5:7]
    if not year.isdigit() or not month.isdigit():
        return None
    if period_kind == "month":
        return f"{year}-{month}"
    return f"{year}-Q{(int(month) - 1) // 3 + 1}"


def _bedroom_bucket(value: Any) -> str:
    if not isinstance(value, int) or isinstance(value, bool):
        return "unknown"
    if value >= 3:
        return "3+"
    return str(value)


def _quantile(values: list[int], q: float) -> float:
    """Linear-interpolated quantile over a sorted list."""
    if not values:
        raise ValueError("no values")
    if len(values) == 1:
        return float(values[0])
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight
