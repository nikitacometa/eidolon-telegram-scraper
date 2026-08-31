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
from pipeline.housing.extractor import (
    EXTRACTOR_VERSION,
    SYSTEM_PROMPT,
    property_typed,
    televised,
)
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
                    "property_type": {
                        "type": ["string", "null"],
                        "enum": ["house", "apartment", "room", "hotel", None],
                    },
                    "terrace": {"type": ["boolean", "null"]},
                    "private_setting": {"type": ["boolean", "null"]},
                    "nature_setting": {"type": ["boolean", "null"]},
                    "amenities": {
                        "type": "object",
                        "properties": {
                            name: {"type": ["boolean", "null"]}
                            for name in (
                                "pool",
                                "aircon",
                                "kitchen",
                                "wifi",
                                "sea_view",
                                "parking",
                                "hot_water",
                                "washing_machine",
                            )
                        },
                        "required": [
                            "pool",
                            "aircon",
                            "kitchen",
                            "wifi",
                            "sea_view",
                            "parking",
                            "hot_water",
                            "washing_machine",
                        ],
                        "additionalProperties": False,
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
                    "property_type",
                    "terrace",
                    "private_setting",
                    "nature_setting",
                    "amenities",
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

    def reseed_stale(self) -> int:
        """Return listings extracted under an older prompt to the queue.

        seed() alone cannot do this: its ON CONFLICT DO NOTHING is what makes
        repeated seeding idempotent, and it deliberately never touches a row
        that was already worked. A version bump is the one case where worked
        rows must be worked again — the extracted ones, whose stored shape is
        now missing fields, and the errored ones, which cost nothing to
        retry. not_housing and gated verdicts stand: the classification of
        "is this a listing at all" did not change between versions.
        """
        cursor = self._conn.execute(
            """
            UPDATE housing_listing_state
            SET status = 'pending', error = NULL, extractor_version = ?
            WHERE status = 'error'
               OR (status = 'extracted' AND corpus_id IN (
                       SELECT corpus_id FROM housing_listings
                       WHERE extractor_version != ?))
            """,
            (EXTRACTOR_VERSION, EXTRACTOR_VERSION),
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

        # A crosspost — the same advertisement in several chats — shares its
        # content_hash, and 22.2% of this corpus is crossposts. One model
        # call per distinct text; the duplicates copy the representative's
        # answer afterwards.
        uniques: list[sqlite3.Row] = []
        duplicates: list[sqlite3.Row] = []
        seen_hashes: set[str] = set()
        for row in candidates:
            content_hash = row["content_hash"]
            if content_hash and content_hash in seen_hashes:
                duplicates.append(row)
                continue
            if content_hash:
                seen_hashes.add(str(content_hash))
            uniques.append(row)

        packs = [
            uniques[index : index + self._pack_size]
            for index in range(0, len(uniques), self._pack_size)
        ]
        semaphore = asyncio.Semaphore(self._concurrency)

        async def one(pack: list[sqlite3.Row]) -> tuple[list[sqlite3.Row], dict[int, Any] | None]:
            async with semaphore:
                return pack, await self._extract_pack(pack)

        extracted = listings = errors = 0
        judged_hashes: set[str] = set()
        # Settled and committed one pack at a time: a run interrupted at pack
        # nine keeps packs one through eight, instead of re-paying for all of
        # them the way a single commit at the end would.
        for finished in asyncio.as_completed([one(pack) for pack in packs]):
            try:
                pack, answers = await finished
            except Exception as error:
                logger.warning("Listing pack failed: %s", error)
                errors += 1
                continue
            if answers is None:
                for row in pack:
                    self._settle(int(row["corpus_id"]), "error", error="pack_failed")
                errors += len(pack)
                self._conn.commit()
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
                if row["content_hash"]:
                    judged_hashes.add(str(row["content_hash"]))
                if answer.get("is_rental_offer") and not answer.get("is_vehicle_ad"):
                    self._store_listing(row, answer)
                    listings += 1
                    self._settle(corpus_id, "extracted")
                else:
                    self._settle(corpus_id, "not_housing")
            self._conn.commit()

        for row in duplicates:
            copied = self._copy_from_donor(row)
            if copied is True:
                extracted += 1
                listings += 1
            elif copied is False and str(row["content_hash"]) in judged_hashes:
                # The representative was judged in this very run and left no
                # listing row: it was not a housing offer, so neither is its
                # word-for-word copy.
                self._settle(int(row["corpus_id"]), "not_housing")
                extracted += 1
            # A donor that errored leaves the duplicate pending for the next
            # pass — an unjudged text must not inherit a failure.
        self._conn.commit()
        return ExtractionRun(len(rows), len(gated), extracted, listings, errors)

    def _copy_from_donor(self, row: sqlite3.Row) -> bool | None:
        """Fill a crosspost from an already-extracted copy of the same text.

        Returns True when a listing row was copied, False when no donor
        listing exists, None when the row has no hash to look up by.
        """
        content_hash = row["content_hash"]
        if not content_hash:
            return None
        donor = self._conn.execute(
            """
            SELECT * FROM housing_listings
            WHERE content_hash = ? AND extractor_version = ?
            ORDER BY listing_id LIMIT 1
            """,
            (content_hash, EXTRACTOR_VERSION),
        ).fetchone()
        if donor is None:
            return False
        self._conn.execute(
            """
            INSERT INTO housing_listings (
                corpus_id, chat_id, content_hash, posted_at,
                is_rental_offer, is_vehicle_ad, bedrooms, bathrooms,
                monthly_price_thb, price_note, tv_present, tv_size_class,
                property_type, terrace, private_setting, nature_setting,
                amenities_json, area_raw, evidence_quote, extractor_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(corpus_id) DO UPDATE SET
                bedrooms = excluded.bedrooms,
                bathrooms = excluded.bathrooms,
                monthly_price_thb = excluded.monthly_price_thb,
                price_note = excluded.price_note,
                tv_present = excluded.tv_present,
                tv_size_class = excluded.tv_size_class,
                property_type = excluded.property_type,
                terrace = excluded.terrace,
                private_setting = excluded.private_setting,
                nature_setting = excluded.nature_setting,
                amenities_json = excluded.amenities_json,
                area_raw = excluded.area_raw,
                evidence_quote = excluded.evidence_quote,
                extractor_version = excluded.extractor_version,
                extracted_at = CURRENT_TIMESTAMP
            """,
            (
                int(row["corpus_id"]),
                int(row["chat_id"]),
                content_hash,
                str(row["date"]),
                1,
                donor["is_vehicle_ad"],
                donor["bedrooms"],
                donor["bathrooms"],
                donor["monthly_price_thb"],
                donor["price_note"],
                donor["tv_present"],
                donor["tv_size_class"],
                donor["property_type"],
                donor["terrace"],
                donor["private_setting"],
                donor["nature_setting"],
                donor["amenities_json"],
                donor["area_raw"],
                donor["evidence_quote"],
                EXTRACTOR_VERSION,
            ),
        )
        self._settle(int(row["corpus_id"]), "extracted")
        return True

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
        # The same corroboration the live path applies: a model that answers
        # "no television" for an advertisement that never mentions one is
        # reporting its own silence, not the property's.
        text = str(row["text"] or "")
        tv_present, tv_size_class = televised(
            text,
            present=_clean_raw_bool(answer.get("tv_present")),
            size_class=answer.get("tv_size_class"),
        )
        # The same fabrication guard the live path applies: a rejecting
        # property type must be corroborated by the text.
        claimed_type = answer.get("property_type")
        if claimed_type not in {None, "house", "apartment", "room", "hotel"}:
            claimed_type = None
        property_type = property_typed(text, claimed=claimed_type)
        raw_amenities = answer.get("amenities")
        amenities = (
            {name: True for name, value in raw_amenities.items() if value is True}
            if isinstance(raw_amenities, dict)
            else {}
        )
        self._conn.execute(
            """
            INSERT INTO housing_listings (
                corpus_id, chat_id, content_hash, posted_at,
                is_rental_offer, is_vehicle_ad, bedrooms, bathrooms,
                monthly_price_thb, price_note, tv_present, tv_size_class,
                property_type, terrace, private_setting, nature_setting,
                amenities_json, area_raw, evidence_quote, extractor_version
            )
            VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(corpus_id) DO UPDATE SET
                bedrooms = excluded.bedrooms,
                bathrooms = excluded.bathrooms,
                monthly_price_thb = excluded.monthly_price_thb,
                price_note = excluded.price_note,
                tv_present = excluded.tv_present,
                tv_size_class = excluded.tv_size_class,
                property_type = excluded.property_type,
                terrace = excluded.terrace,
                private_setting = excluded.private_setting,
                nature_setting = excluded.nature_setting,
                amenities_json = excluded.amenities_json,
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
                _clean_bool(tv_present),
                tv_size_class,
                property_type,
                1 if answer.get("terrace") is True else None,
                1 if answer.get("private_setting") is True else None,
                1 if answer.get("nature_setting") is True else None,
                json.dumps(amenities, ensure_ascii=False) if amenities else None,
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


def _clean_raw_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


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
