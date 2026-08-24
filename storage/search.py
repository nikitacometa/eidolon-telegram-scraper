"""Search index over the union of the live and reconnaissance message stores.

The two stores exist for a good reason -- bulk crawl writes must never queue
behind live alert delivery -- but that split makes the corpus unsearchable as a
whole: SQLite will not let an FTS5 table or a persisted view span two attached
files. This module maintains a third, purely derived file that holds the union
plus the indexes, and can be deleted and rebuilt at any time.

Everything here is synchronous ``sqlite3`` rather than ``aiosqlite``. The two
processes that touch this file -- the indexer and the MCP bridge -- are both
standalone and single-purpose, and neither shares an event loop with the
Telethon session. Async here would buy nothing and cost a connection model that
has to be reasoned about.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SEARCH_SCHEMA_PATH = Path(__file__).parent / "search_schema.sql"

# The current writer replaces active mentions in one transaction, so an explicit
# pilot may safely sample every settled legacy state, including ``extracted``.
REEXTRACTABLE_STATUSES = frozenset({"extracted", "no_venue", "skipped"})
ENTITY_EXTRACTION_VERSION = "entities-v4"
ENTITY_KINDS = frozenset({"place", "person", "organization"})
ACCESS_MODES = frozenset({"visit", "house_call", "delivery", "remote", "unknown"})

# Reciprocal-rank-fusion constant. 60 is the value the original RRF paper used
# and the one every implementation defaults to; it needs corpus-specific tuning
# only if the two lanes have wildly different list lengths, which they do not.
RRF_K = 60

# Telegram chat references as they appear in message text.
_TME_RE = re.compile(
    r"(?:https?://)?t\.me/(?P<body>\+[\w-]+|joinchat/[\w-]+|[A-Za-z][\w]{3,31})",
    re.IGNORECASE,
)
_AT_HANDLE_RE = re.compile(r"(?<![\w@/])@(?P<name>[A-Za-z][\w]{4,31})")

# A bare domain with a path: "maps.app.goo.gl/xyz", "t.me/chan". Catches the
# link-instead-of-name case even when the scheme was stripped upstream.
_URLISH_RE = re.compile(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+/")

# Handles that are people or bots we do not want in a chat-candidate list.
_REF_STOPWORDS = frozenset({"joinchat", "share", "addstickers", "proxy", "socks"})

# --- Contact handles -------------------------------------------------------
# One pattern per channel people actually publish. Instagram and Facebook are
# taken only from a full link: "@someone" in an Instagram sentence is
# indistinguishable from a Telegram handle, and guessing wrong hands the reader
# a contact that does not exist.
_CONTACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instagram",
        re.compile(r"(?:https?://)?(?:www\.)?instagram\.com/(?P<v>[A-Za-z0-9._]{2,30})", re.I),
    ),
    (
        "whatsapp",
        re.compile(r"(?:https?://)?(?:chat\.whatsapp\.com|wa\.me)/(?P<v>[\w+-]{5,40})", re.I),
    ),
    ("zalo", re.compile(r"(?:https?://)?zalo\.me/(?P<v>[\w+-]{5,40})", re.I)),
    (
        "facebook",
        re.compile(
            r"(?:https?://)?(?:www\.|m\.)?(?:facebook\.com|fb\.me|m\.me)/(?P<v>[A-Za-z0-9._-]{3,40})",
            re.I,
        ),
    ),
    (
        "maps",
        re.compile(
            r"(?:https?://)?(?:maps\.app\.goo\.gl/[\w-]+"
            r"|goo\.gl/maps/[\w-]+"
            r"|(?:www\.)?google\.[a-z.]{2,6}/maps/[^\s)>\]]+)",
            re.I,
        ),
    ),
)

# Phone numbers, and only the two shapes that are unambiguous here: an
# international number written with its plus, and a Vietnamese mobile in local
# form. Anything looser matches prices -- this corpus is full of "500.000",
# "1 500 000 VND" and apartment sizes -- and a wrong phone number is worse than
# no phone number, because the reader cannot tell it is wrong until they dial.
# Measured over the 65k-message corpus: these two shapes yield 211 distinct
# numbers and no price. A trailing currency guard was tried alongside them and
# fired zero times, so the shape of the pattern is what does the work here.
_PHONE_INTL_RE = re.compile(r"(?<![\w+])\+(?P<v>\d[\d\s().-]{7,17}\d)(?![\w])")
_PHONE_VN_RE = re.compile(r"(?<![\w+])0(?P<v>[35789]\d[\d\s.-]{6,10}\d)(?![\w])")


def is_extraction_candidate(text: str) -> bool:
    """Return whether non-empty text can carry an identifiable local entity.

    The old length/vocabulary gate hid terse self-promotion and made every new
    category a code change. The current gate is category-neutral: two Unicode letters are
    enough, while a deterministically mined contact also admits contact-only
    provider cards.
    """
    value = text.strip()
    if not value:
        return False
    if sum(character.isalpha() for character in value) >= 2:
        return True
    return bool(extract_contacts(value))


def content_digest(text: str | None) -> str | None:
    """Return the crosspost-dedup digest for a message body.

    Deliberately identical to :meth:`storage.scout.ScoutDatabase.store_message`
    so a message synced from either store lands on the same hash.
    """
    if not text:
        return None
    return hashlib.sha256(" ".join(text.split()).lower().encode("utf-8")).hexdigest()


def fold_ascii(value: str) -> str:
    """Fold stylized branding to a plain-ASCII lookup key.

    Nightlife venues brand themselves with letters that look like ASCII but are
    not: SYNCHOUSE is written SYNCHØUSE, with U+00D8. FTS5 does not fold that --
    ``remove_diacritics`` only strips combining marks, and U+00D8 is a distinct
    base letter -- so a search for the name a person would actually type finds
    nothing. Every venue gets an ASCII-folded key alongside its real name.
    """
    swaps = {
        "ø": "o",
        "Ø": "o",
        "đ": "d",
        "Đ": "d",
        "ł": "l",
        "Ł": "l",
        "æ": "ae",
        "Æ": "ae",
        "œ": "oe",
        "Œ": "oe",
        "ß": "ss",
        "þ": "th",
    }
    swapped = "".join(swaps.get(ch, ch) for ch in value)
    decomposed = unicodedata.normalize("NFKD", swapped)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.lower().split())


# Attempts a message gets before extraction gives up on it for good.
MAX_EXTRACTION_ATTEMPTS = 3

# Provider faults worth another call. Anything else -- a refusal, a truncated
# completion, malformed output -- is a property of this particular message and
# will reproduce exactly on a retry.
_TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "ratelimit",
    "rate limit",
    "connection",
    "apiconnection",
    "internalserver",
    "service unavailable",
    "overloaded",
    "502",
    "503",
    "504",
)


def is_transient_error(error: str) -> bool:
    """Whether an extraction failure is worth retrying later."""
    lowered = error.lower()
    return any(marker in lowered for marker in _TRANSIENT_ERROR_MARKERS)


def is_venue_name(name: str) -> bool:
    """Reject values that are not names of places.

    Extraction is an LLM reading messy chat text, and a message that names a
    venue usually also carries its map link. Asked for the name, the model
    sometimes hands back the link. One such row is not merely noise: it becomes
    a permanent entry in the venue index that no later pass removes, and it is
    what the agent reads back to the user as a place to hold a concert.
    """
    value = name.strip()
    if len(value) < 2 or len(value) > 120:
        return False
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "www.", "@")):
        return False
    if "://" in lowered or _URLISH_RE.search(lowered):
        return False
    # A name has to contain a letter; "+84 905 123 456" and "18:30" do not.
    return any(ch.isalpha() for ch in value)


def is_entity_label(name: str, *, entity_kind: str, contact_displays: Sequence[str] = ()) -> bool:
    """Validate a place name or an exact deterministic contact display.

    Places retain the old letter-bearing boundary byte-for-byte. A person or
    organization may use a phone/handle as its only identity, but only when the
    contact miner found that exact display in the source message.
    """
    if entity_kind == "place":
        return is_venue_name(name)
    value = name.strip()
    if len(value) < 2 or len(value) > 120 or "://" in value.lower() or _URLISH_RE.search(value):
        return False
    if any(character.isalpha() for character in value) and not value.startswith("@"):
        return True
    return value in contact_displays


def normalize_descriptor(value: str) -> str:
    """Normalize an open category phrase without translating or stemming it."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


# Versioned compatibility mapping. Unknown open text remains searchable and
# simply projects to ``other``/``[]`` for legacy clients.
_PLACE_TYPE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dentist", ("dentist", "dental", "стоматолог", "nha khoa")),
    ("pharmacy", ("pharmacy", "аптек", "nhà thuốc")),
    ("hospital", ("hospital", "больниц", "bệnh viện")),
    ("clinic", ("clinic", "клиник", "phòng khám")),
    ("repair", ("repair", "ремонт", "мастер", "sửa chữa")),
    ("salon", ("salon", "barber", "beauty", "салон", "барбер", "парикмах")),
    ("gym", ("gym", "fitness", "спортзал", "фитнес")),
    ("school", ("school", "course", "tutor", "школ", "курс", "репетитор")),
    ("laundry", ("laundry", "dry cleaning", "прачеч", "химчист")),
    ("cafe", ("cafe", "coffee", "кафе", "кофейн")),
    ("restaurant", ("restaurant", "ресторан")),
    ("bar", (" bar", "bar ", "бар", "pub")),
    ("club", ("club", "клуб")),
    ("hotel", ("hotel", "отел", "гостиниц")),
    ("yoga", ("yoga", "йог")),
    ("studio", ("studio", "студи")),
    ("gallery", ("gallery", "галере")),
    ("coworking", ("coworking", "коворкинг")),
    ("hostel", ("hostel", "хостел")),
    ("theatre", ("theatre", "theater", "театр")),
)

_EVENT_TYPE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("live_music", ("live music", "живая музыка", "лайв")),
    ("concert", ("concert", "концерт")),
    ("dj_set", ("dj set", "диджей", "дидже")),
    ("open_mic", ("open mic", "открытый микрофон")),
    ("jam", ("jam", "джем")),
    ("party", ("party", "вечерин")),
    ("festival", ("festival", "фестивал")),
    ("quiz", ("quiz", "квиз")),
    ("board_games", ("board game", "настольн")),
    ("film_screening", ("film screening", "кинопоказ")),
    ("meetup", ("meetup", "митап", "встреч")),
    ("workshop", ("workshop", "мастер-класс")),
    ("lecture", ("lecture", "лекци")),
    ("language_club", ("language club", "языковой клуб")),
    ("market", ("market", "маркет", "ярмарк")),
    ("yoga", ("yoga", "йог")),
    ("meditation", ("meditation", "медитац")),
    ("sound_healing", ("sound healing", "саундхилинг")),
    ("ecstatic_dance", ("ecstatic dance", "экстатик")),
    ("retreat", ("retreat", "ретрит")),
    ("sport", ("sport", "спорт")),
    ("food", ("food", "еда", "ужин", "завтрак")),
)


def derive_legacy_facets(descriptor: str, offerings: Sequence[str]) -> tuple[str, list[str]]:
    """Project open entity text into the legacy closed facets."""
    haystack = f" {normalize_descriptor(descriptor)} " + " ".join(
        normalize_descriptor(value) for value in offerings
    )
    place_type = next(
        (facet for facet, terms in _PLACE_TYPE_TERMS if any(term in haystack for term in terms)),
        "other",
    )
    event_types = sorted(
        facet for facet, terms in _EVENT_TYPE_TERMS if any(term in haystack for term in terms)
    )
    return place_type, event_types


def escape_fts_term(term: str) -> str:
    """Quote one bare term for an FTS5 MATCH expression.

    User and model input reaches MATCH directly. Unquoted, a stray hyphen or
    quote is not a bad search, it is a syntax error that fails the whole call.
    """
    return '"' + term.replace('"', '""') + '"'


# Words that carry no selectivity. A prefix match on "в" or "и" hits nearly
# every message in a Russian corpus, so ORing them into a query does not widen
# the net -- it drowns the terms that mattered, because BM25 then ranks on words
# the asker never cared about.
# Written as prose and split once at import: a hundred-element list literal is
# what the linter would prefer and is unreadable for a word list humans edit.
_STOPWORD_TEXT = (
    "а и но или да же ли бы что как где когда куда чтобы если то это этот эта эти "
    "в во на за под над при по из от до для с со у к ко о об про без через между "
    "я ты он она оно мы вы они мне тебе нам вам им его её их себя свой своя "
    "не ни уж вот там тут здесь тоже также ещё еще уже только просто очень "
    "есть быть был была было были будет буду будем можно нужно надо "
    "a an the and or but if then this that these those is are was were be been "
    "in on at by for to from of with without about into over under "
    "i you he she it we they me him her us them my your his its our their "
    "not no yes very just only also too some any all"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

# Anything shorter than this is either a stopword we missed or a fragment whose
# prefix match would sweep in most of the corpus.
_MIN_TERM_LENGTH = 3


def content_terms(text: str) -> list[str]:
    """Extract the words worth searching for from a natural-language query.

    Splitting a sentence on whitespace and ORing the pieces is how a question
    like "площадка в Дананге где проводят живую музыку" becomes
    ``"площадка"* OR "в"* OR "где"* OR ...`` -- a query whose top results are
    ranked by how often a message says "в".
    """
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    return [
        word for word in words if len(word) >= _MIN_TERM_LENGTH and word.lower() not in _STOPWORDS
    ]


def build_fts_query(terms: Sequence[str], *, prefix: bool = True) -> str:
    """Build a safe OR-of-prefixes MATCH expression from plain terms.

    Prefix matching is what makes Russian work here: an exact-token index
    matches ``бар`` and misses ``бара``/``баре``/``барах``, which measured out
    at 14.6% recall on this corpus.
    """
    parts: list[str] = []
    for raw in terms:
        term = raw.strip()
        if not term:
            continue
        if any(ch in term for ch in " \t"):
            # A phrase: quote whole, no prefix star (FTS5 rejects `"a b"*`).
            parts.append(escape_fts_term(term))
            continue
        quoted = escape_fts_term(term)
        parts.append(f"{quoted}*" if prefix else quoted)
    return " OR ".join(parts)


@dataclass(slots=True)
class CorpusRow:
    """One indexed message."""

    corpus_id: int
    chat_id: int
    telegram_msg_id: int
    chat_title: str | None
    sender_name: str | None
    text: str
    date: str
    content_hash: str
    source: str
    reply_to_message_id: int | None
    score: float = 0.0
    lanes: list[str] = field(default_factory=list)
    duplicate_chat_ids: list[int] = field(default_factory=list)
    # The message this one answers, when it is an answer and the question is
    # in the corpus. Six in ten messages in a city chat are replies, and an
    # answer alone ("да, Beeppub, каждый день с 21") names neither the
    # question nor what was asked about.
    in_reply_to: dict[str, Any] | None = None

    def as_dict(self, *, text_limit: int | None = None) -> dict[str, Any]:
        body = self.text
        if text_limit is not None and len(body) > text_limit:
            body = body[:text_limit].rstrip() + "…"
        return {
            "corpus_id": self.corpus_id,
            "chat_id": self.chat_id,
            "chat_title": self.chat_title,
            "message_link": message_link(self.chat_id, self.telegram_msg_id),
            "reply_to_message_id": self.reply_to_message_id,
            "in_reply_to": self.in_reply_to,
            "sender": self.sender_name,
            "date": self.date,
            "text": body,
            "score": round(self.score, 5),
            "matched_via": self.lanes,
            "also_posted_in": self.duplicate_chat_ids,
        }


def message_link(chat_id: int, telegram_msg_id: int) -> str:
    """Return a deep link a human can click to open the source message."""
    # Supergroup/channel ids are -100 prefixed; t.me/c wants them without it.
    raw = str(chat_id)
    if raw.startswith("-100"):
        return f"https://t.me/c/{raw[4:]}/{telegram_msg_id}"
    return f"https://t.me/c/{raw.lstrip('-')}/{telegram_msg_id}"


class SearchDatabase:
    """Owns ``eidolon_search.db``: the union corpus plus its indexes."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._vector_cache: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Search database is not connected")
        return self._conn

    def connect(self, *, read_only: bool = False) -> None:
        """Open the connection, applying the schema unless read-only."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # URI mode is on in both branches, not just read-only: ATTACH only
        # honours a `file:...?mode=ro` argument when the connection that runs it
        # was itself opened with URI filenames enabled. Without this, the sync
        # job silently loses its read-only guarantee against the live stores.
        if read_only:
            self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=10.0)
        else:
            self._conn = sqlite3.connect(f"file:{self.db_path}", uri=True, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        if not read_only:
            os.chmod(self.db_path, 0o600)
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SEARCH_SCHEMA_PATH.read_text())
            self._migrate()
            self._conn.commit()
        self._conn.execute("PRAGMA busy_timeout=10000")
        logger.info("Search database connected: %s (read_only=%s)", self.db_path, read_only)

    def _migrate(self) -> None:
        """Apply column additions to an index that predates them.

        The index is rebuildable in principle, but a rebuild re-pays for the
        whole extraction pass, so an in-place migration is worth the few lines.
        """
        columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(extraction_state)")}
        if "attempts" not in columns:
            self.conn.execute(
                "ALTER TABLE extraction_state ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
            )
            logger.info("Migration: added extraction_state.attempts")

        # 'duplicate' joined the status set when crosspost dedup landed. A CHECK
        # constraint cannot be altered in place, so the table is rebuilt -- the
        # rows are cheap to move and re-paying for the extraction they record is
        # not.
        ddl = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='extraction_state'"
        ).fetchone()
        if ddl and "'duplicate'" not in ddl["sql"]:
            self.conn.executescript(
                """
                PRAGMA foreign_keys=off;
                BEGIN;
                ALTER TABLE extraction_state RENAME TO extraction_state_old;
                CREATE TABLE extraction_state (
                    corpus_id INTEGER PRIMARY KEY
                        REFERENCES corpus_messages(corpus_id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'extracted', 'no_venue', 'skipped',
                                         'error', 'duplicate')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    attempted_at TIMESTAMP,
                    error TEXT,
                    active_prompt_version TEXT NOT NULL DEFAULT 'places-v2'
                );
                INSERT INTO extraction_state (
                    corpus_id, status, attempts, attempted_at, error, active_prompt_version
                )
                    SELECT corpus_id, status, attempts, attempted_at, error, 'places-v2'
                      FROM extraction_state_old;
                DROP TABLE extraction_state_old;
                CREATE INDEX IF NOT EXISTS idx_extraction_pending
                    ON extraction_state(status, corpus_id);
                COMMIT;
                PRAGMA foreign_keys=on;
                """
            )
            logger.info("Migration: extraction_state now accepts 'duplicate'")

        self._add_column(
            "places",
            "entity_kind",
            "TEXT NOT NULL DEFAULT 'place' "
            "CHECK(entity_kind IN ('place', 'person', 'organization'))",
        )
        self._add_column(
            "places",
            "access_modes",
            "TEXT NOT NULL DEFAULT '[\"visit\"]' CHECK(json_valid(access_modes))",
        )
        self._add_column("places", "primary_descriptor", "TEXT")
        self._add_column("places", "descriptor_text", "TEXT NOT NULL DEFAULT ''")
        self._add_column("places", "offering_text", "TEXT NOT NULL DEFAULT ''")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_places_kind_city ON places(entity_kind, city_area)"
        )

        self._add_column(
            "place_mentions", "descriptor_id", "INTEGER REFERENCES descriptors(descriptor_id)"
        )
        self._add_column("place_mentions", "descriptor_raw", "TEXT")
        self._add_column(
            "place_mentions",
            "offerings_raw",
            "TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(offerings_raw))",
        )
        self._add_column("place_mentions", "entity_kind_raw", "TEXT")
        self._add_column(
            "place_mentions",
            "access_modes_raw",
            "TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(access_modes_raw))",
        )
        self._add_column("place_mentions", "extractor_version", "TEXT NOT NULL DEFAULT 'places-v2'")
        self._add_column(
            "extraction_state",
            "active_prompt_version",
            "TEXT NOT NULL DEFAULT 'places-v2'",
        )
        self._add_column("corpus_messages", "reply_to_message_id", "INTEGER")
        # Answers are grouped by chat and parent id; the table's existing
        # UNIQUE(chat_id, telegram_msg_id) index serves the parent side.
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_reply "
            "ON corpus_messages(chat_id, reply_to_message_id) "
            "WHERE reply_to_message_id IS NOT NULL"
        )
        # Descriptor stats are recomputed after every extraction; without this
        # index each recompute scans the whole mentions table per descriptor.
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_place_mentions_descriptor "
            "ON place_mentions(descriptor_id) WHERE descriptor_id IS NOT NULL"
        )

        # Legacy facets are compatibility projections only. They provide useful
        # lexical coverage before v3 backfill without pretending an English enum
        # was a phrase extracted from the source message.
        self.conn.execute(
            """
            UPDATE places
               SET descriptor_text = CASE
                       WHEN lower(COALESCE(place_type, 'other')) = 'other' THEN ''
                       ELSE lower(place_type)
                   END
             WHERE descriptor_text = ''
            """
        )
        self.conn.execute(
            """
            UPDATE places
               SET offering_text = COALESCE((
                   SELECT group_concat(value, ' ')
                     FROM (
                         SELECT DISTINCT lower(j.value) AS value
                           FROM place_mentions pm, json_each(pm.event_types) AS j
                          WHERE pm.place_id = places.place_id
                          ORDER BY value
                     )
               ), '')
             WHERE offering_text = ''
            """
        )
        self._ensure_shadow_place_fts()
        self._seed_extraction_jobs(ENTITY_EXTRACTION_VERSION)

    def _add_column(self, table: str, name: str, ddl: str) -> None:
        """Add one migration column only when the existing index lacks it."""
        columns = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if name in columns:
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
        logger.info("Migration: added %s.%s", table, name)

    def _ensure_shadow_place_fts(self) -> None:
        """Build the expanded FTS beside the active legacy name index."""
        columns = [row["name"] for row in self.conn.execute("PRAGMA table_info(place_fts_next)")]
        expected = ["name", "aliases", "descriptor_text", "offering_text"]
        if columns and columns != expected:
            self.conn.executescript(
                """
                DROP TRIGGER IF EXISTS place_fts_next_ai;
                DROP TRIGGER IF EXISTS place_fts_next_ad;
                DROP TRIGGER IF EXISTS place_fts_next_au;
                DROP TABLE place_fts_next;
                """
            )
        self.conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS place_fts_next USING fts5(
                name, aliases, descriptor_text, offering_text,
                content='places', content_rowid='place_id', tokenize='trigram'
            );
            CREATE TRIGGER IF NOT EXISTS place_fts_next_ai AFTER INSERT ON places BEGIN
                INSERT INTO place_fts_next(rowid, name, aliases, descriptor_text, offering_text)
                VALUES (new.place_id, new.name, new.aliases,
                        new.descriptor_text, new.offering_text);
            END;
            CREATE TRIGGER IF NOT EXISTS place_fts_next_ad AFTER DELETE ON places BEGIN
                INSERT INTO place_fts_next(place_fts_next, rowid, name, aliases,
                                           descriptor_text, offering_text)
                VALUES ('delete', old.place_id, old.name, old.aliases,
                        old.descriptor_text, old.offering_text);
            END;
            CREATE TRIGGER IF NOT EXISTS place_fts_next_au AFTER UPDATE ON places BEGIN
                INSERT INTO place_fts_next(place_fts_next, rowid, name, aliases,
                                           descriptor_text, offering_text)
                VALUES ('delete', old.place_id, old.name, old.aliases,
                        old.descriptor_text, old.offering_text);
                INSERT INTO place_fts_next(rowid, name, aliases,
                                           descriptor_text, offering_text)
                VALUES (new.place_id, new.name, new.aliases,
                        new.descriptor_text, new.offering_text);
            END;
            """
        )
        self.conn.execute("INSERT INTO place_fts_next(place_fts_next) VALUES('rebuild')")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._vector_cache = None

    def __enter__(self) -> SearchDatabase:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Sync from the two source stores
    # ------------------------------------------------------------------

    def sync(
        self,
        *,
        live_db: Path,
        scout_db: Path,
        batch_size: int = 5000,
        max_batches: int = 1000,
        rescan_every: int = 60,
    ) -> dict[str, int]:
        """Pull new rows from both source stores into the corpus.

        Reads happen through short, bounded transactions against the sources.
        Both are in WAL mode, so a reader cannot block the daemon's writes --
        but a *long-held* reader stops the WAL from being reclaimed, which is
        why every batch is capped and the connection detaches between ticks.
        """
        conn = self.conn
        conn.execute("ATTACH DATABASE ? AS live", (f"file:{live_db}?mode=ro",))
        conn.execute("ATTACH DATABASE ? AS scout", (f"file:{scout_db}?mode=ro",))
        try:
            live_reply_available = self._attached_column_exists(
                "live", "messages", "reply_to_message_id"
            )
            scout_reply_available = self._attached_column_exists(
                "scout", "scout_messages", "reply_to_message_id"
            )
            live_select_sql = (
                """
                SELECT id AS cursor, chat_id, telegram_msg_id, chat_title,
                       sender_id, sender_name, text, date, NULL AS content_hash,
                       reply_to_message_id
                  FROM live.messages
                 WHERE id > ? AND text IS NOT NULL AND trim(text) <> ''
                 ORDER BY id LIMIT ?
                """
                if live_reply_available
                else """
                SELECT id AS cursor, chat_id, telegram_msg_id, chat_title,
                       sender_id, sender_name, text, date, NULL AS content_hash,
                       NULL AS reply_to_message_id
                  FROM live.messages
                 WHERE id > ? AND text IS NOT NULL AND trim(text) <> ''
                 ORDER BY id LIMIT ?
                """
            )
            scout_select_sql = (
                """
                SELECT rowid AS cursor, chat_id, telegram_msg_id, NULL AS chat_title,
                       sender_id, sender_name, text, date, content_hash,
                       reply_to_message_id
                  FROM scout.scout_messages
                 WHERE rowid > ? AND text IS NOT NULL AND trim(text) <> ''
                 ORDER BY rowid LIMIT ?
                """
                if scout_reply_available
                else """
                SELECT rowid AS cursor, chat_id, telegram_msg_id, NULL AS chat_title,
                       sender_id, sender_name, text, date, content_hash,
                       NULL AS reply_to_message_id
                  FROM scout.scout_messages
                 WHERE rowid > ? AND text IS NOT NULL AND trim(text) <> ''
                 ORDER BY rowid LIMIT ?
                """
            )
            stats = {
                "live": self._sync_stream(
                    stream="live",
                    select_sql=live_select_sql,
                    # messages.id is AUTOINCREMENT, so the retention purge that
                    # runs daily against this table can never cause an id to be
                    # reused. Drift detection here would only misread a purge as
                    # a lost row and rescan the whole table every tick.
                    rowid_reuse_risk=False,
                    rescan_every=rescan_every,
                    batch_size=batch_size,
                    max_batches=max_batches,
                ),
                "scout": self._sync_stream(
                    stream="scout",
                    select_sql=scout_select_sql,
                    # scout_messages has only a composite primary key, so its
                    # rowid is reused after a delete. Nothing deletes from it
                    # today, but a cursor that silently skips a row forever is
                    # not a risk worth carrying on an assumption about tomorrow.
                    rowid_reuse_risk=True,
                    rescan_every=rescan_every,
                    batch_size=batch_size,
                    max_batches=max_batches,
                ),
            }
            stats["reply_links"] = self._reconcile_reply_links(
                live_available=live_reply_available,
                scout_available=scout_reply_available,
            )
            self._refresh_chat_metadata()
            self._seed_extraction_state()
            stats["named"] = self._resolve_sender_names()
            stats["contacts"] = self._extract_contacts()
            conn.commit()
        finally:
            conn.execute("DETACH DATABASE live")
            conn.execute("DETACH DATABASE scout")
        self._vector_cache = None
        return stats

    def _attached_column_exists(self, schema: str, table: str, column: str) -> bool:
        """Feature-detect a source column before selecting from an older store."""
        columns = {row["name"] for row in self.conn.execute(f"PRAGMA {schema}.table_info({table})")}
        return column in columns

    def _reconcile_reply_links(self, *, live_available: bool, scout_available: bool) -> int:
        """Fill links on corpus rows synced before the source backfill ran."""
        changed = 0
        if live_available:
            self.conn.execute(
                """
                UPDATE corpus_messages AS corpus
                   SET reply_to_message_id = (
                       SELECT live_message.reply_to_message_id
                         FROM live.messages AS live_message
                        WHERE live_message.chat_id = corpus.chat_id
                          AND live_message.telegram_msg_id = corpus.telegram_msg_id
                   )
                 WHERE corpus.reply_to_message_id IS NULL
                   AND EXISTS (
                       SELECT 1 FROM live.messages AS live_message
                        WHERE live_message.chat_id = corpus.chat_id
                          AND live_message.telegram_msg_id = corpus.telegram_msg_id
                          AND live_message.reply_to_message_id IS NOT NULL
                   )
                """
            )
            changed += int(self.conn.execute("SELECT changes()").fetchone()[0])
        if scout_available:
            self.conn.execute(
                """
                UPDATE corpus_messages AS corpus
                   SET reply_to_message_id = (
                       SELECT scout_message.reply_to_message_id
                         FROM scout.scout_messages AS scout_message
                        WHERE scout_message.chat_id = corpus.chat_id
                          AND scout_message.telegram_msg_id = corpus.telegram_msg_id
                   )
                 WHERE corpus.reply_to_message_id IS NULL
                   AND EXISTS (
                       SELECT 1 FROM scout.scout_messages AS scout_message
                        WHERE scout_message.chat_id = corpus.chat_id
                          AND scout_message.telegram_msg_id = corpus.telegram_msg_id
                          AND scout_message.reply_to_message_id IS NOT NULL
                   )
                """
            )
            changed += int(self.conn.execute("SELECT changes()").fetchone()[0])
        return changed

    def _sync_stream(
        self,
        *,
        stream: str,
        select_sql: str,
        rowid_reuse_risk: bool,
        rescan_every: int,
        batch_size: int,
        max_batches: int,
    ) -> int:
        """Copy new rows from one source stream, self-healing against drift.

        The scout cursor rides on ``rowid``, which SQLite reuses after a delete.
        Nothing deletes from ``scout_messages`` today, so the cursor is exact --
        but "today" is not a guarantee, and a reused rowid would silently skip a
        row forever. Comparing the source's own row count against what we have
        stored detects that in one cheap COUNT, and a detected gap restarts the
        stream from zero rather than leaving a permanent hole.
        """
        conn = self.conn
        row = conn.execute(
            "SELECT last_id, rows_synced, ticks FROM sync_state WHERE source = ?", (stream,)
        ).fetchone()
        cursor_id = int(row["last_id"]) if row else 0
        rows_seen = int(row["rows_synced"]) if row else 0
        ticks = int(row["ticks"]) if row else 0

        if rowid_reuse_risk and ticks >= rescan_every:
            # Periodic full rescan. Counting cannot detect the case this
            # defends against -- one delete plus one insert leaves the row
            # count unchanged while the new row lands below the cursor -- so
            # the honest fix is to re-read the stream on a schedule rather than
            # to reason harder about the cursor. Rows already present are
            # skipped by ON CONFLICT, so a rescan costs a scan and nothing else.
            logger.info("Full rescan of %s after %d incremental ticks", stream, ticks)
            cursor_id = 0
            rows_seen = 0
            ticks = 0

        before = self._corpus_count()
        exhausted = False
        for _ in range(max_batches):
            rows = conn.execute(select_sql, (cursor_id, batch_size)).fetchall()
            if not rows:
                exhausted = True
                break
            payload = []
            for r in rows:
                digest = r["content_hash"] or content_digest(r["text"])
                if digest is None:
                    continue
                payload.append(
                    (
                        stream,
                        r["chat_id"],
                        r["telegram_msg_id"],
                        r["chat_title"],
                        r["sender_id"],
                        r["sender_name"],
                        r["text"],
                        r["date"],
                        r["reply_to_message_id"],
                        digest,
                    )
                )
            if payload:
                conn.executemany(
                    """
                    INSERT INTO corpus_messages (
                        source, chat_id, telegram_msg_id, chat_title,
                        sender_id, sender_name, text, date, reply_to_message_id, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id, telegram_msg_id) DO UPDATE SET
                        reply_to_message_id = COALESCE(
                            excluded.reply_to_message_id,
                            corpus_messages.reply_to_message_id
                        )
                    WHERE excluded.reply_to_message_id IS NOT NULL
                      AND corpus_messages.reply_to_message_id IS NULL
                    """,
                    payload,
                )
            cursor_id = int(rows[-1]["cursor"])
            rows_seen += len(rows)
            self._save_cursor(stream, cursor_id, rows_seen, ticks)
            conn.commit()
            if len(rows) < batch_size:
                exhausted = True
                break

        if exhausted:
            self._save_cursor(stream, cursor_id, rows_seen, ticks + 1)
            conn.commit()
        return self._corpus_count() - before

    def _save_cursor(self, stream: str, cursor_id: int, rows_seen: int, ticks: int) -> None:
        self.conn.execute(
            """
            INSERT INTO sync_state (source, last_id, rows_synced, ticks, last_run_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                last_id = excluded.last_id,
                rows_synced = excluded.rows_synced,
                ticks = excluded.ticks,
                last_run_at = excluded.last_run_at
            """,
            (stream, cursor_id, rows_seen, ticks, _now()),
        )

    def _corpus_count(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM corpus_messages").fetchone()[0])

    def _refresh_chat_metadata(self) -> None:
        """Recompute per-chat counts and pull observation mode from the sources."""
        conn = self.conn
        conn.execute(
            """
            INSERT INTO corpus_chats (chat_id, message_count, first_message_at, last_message_at, updated_at)
            SELECT chat_id, count(*), min(date), max(date), ?
              FROM corpus_messages GROUP BY chat_id
            ON CONFLICT(chat_id) DO UPDATE SET
                message_count = excluded.message_count,
                first_message_at = excluded.first_message_at,
                last_message_at = excluded.last_message_at,
                updated_at = excluded.updated_at
            """,
            (_now(),),
        )
        # Titles: prefer the observation registry, fall back to whatever the
        # live ingest recorded, then to the backfill target label.
        conn.execute(
            """
            UPDATE corpus_chats SET
                title = COALESCE(
                    (SELECT title FROM live.observed_chats o WHERE o.chat_id = corpus_chats.chat_id),
                    (SELECT title FROM live.chats c WHERE c.chat_id = corpus_chats.chat_id),
                    (SELECT label FROM scout.backfill_targets b WHERE b.chat_id = corpus_chats.chat_id),
                    title
                ),
                mode = COALESCE(
                    (SELECT mode FROM live.observed_chats o WHERE o.chat_id = corpus_chats.chat_id),
                    'history-only'
                )
            """
        )

    def _seed_extraction_state(self) -> None:
        """Create compatibility state and one current-version job per eligible message."""
        new_rows = self.conn.execute(
            """
            SELECT corpus_id, text
              FROM corpus_messages
             WHERE corpus_id NOT IN (SELECT corpus_id FROM extraction_state)
            """
        ).fetchall()
        self.conn.executemany(
            "INSERT INTO extraction_state (corpus_id, status) VALUES (?, ?)",
            (
                (row["corpus_id"], "pending" if is_extraction_candidate(row["text"]) else "skipped")
                for row in new_rows
            ),
        )
        skipped_rows = self.conn.execute(
            """
            SELECT s.corpus_id, m.text
              FROM extraction_state s JOIN corpus_messages m USING(corpus_id)
             WHERE s.status = 'skipped'
            """
        ).fetchall()
        self.conn.executemany(
            "UPDATE extraction_state SET status = 'pending' WHERE corpus_id = ?",
            ((row["corpus_id"],) for row in skipped_rows if is_extraction_candidate(row["text"])),
        )
        self._mark_crosspost_duplicates()
        self._seed_extraction_jobs(ENTITY_EXTRACTION_VERSION)

    def _seed_extraction_jobs(self, prompt_version: str) -> int:
        """Idempotently create one replacement job per message and prompt version."""
        rows = self.conn.execute(
            """
            SELECT m.corpus_id, m.text
              FROM corpus_messages m
              JOIN extraction_state s USING(corpus_id)
             WHERE s.active_prompt_version <> ?
               AND NOT EXISTS (
                   SELECT 1 FROM extraction_jobs j
                    WHERE j.corpus_id = m.corpus_id AND j.prompt_version = ?
               )
            """,
            (prompt_version, prompt_version),
        ).fetchall()
        payload = [
            (row["corpus_id"], prompt_version)
            for row in rows
            if is_extraction_candidate(str(row["text"]))
        ]
        if payload:
            self.conn.executemany(
                """
                INSERT INTO extraction_jobs (corpus_id, prompt_version)
                VALUES (?, ?) ON CONFLICT(corpus_id, prompt_version) DO NOTHING
                """,
                payload,
            )
        return len(payload)

    def _mark_crosspost_duplicates(self) -> None:
        """Send one message per identical text to the model, not all of them.

        A quarter of this corpus is the same announcement posted into several
        chats. Identical text yields an identical answer, so paying for each
        copy buys nothing -- measured at 4,808 of 34,018 settled calls, 14%.
        The copies are settled from the original by ``propagate_duplicates``
        once it returns, which keeps every crosspost's own place_mentions row
        and leaves the venue index byte-identical to what per-message
        extraction produced.
        """
        self.conn.execute(
            """
            UPDATE extraction_state SET status = 'duplicate'
             WHERE status = 'pending'
               AND corpus_id NOT IN (
                   SELECT min(m.corpus_id) FROM corpus_messages m
                     JOIN extraction_state e USING(corpus_id)
                    WHERE e.status IN ('pending', 'extracted', 'no_venue')
                    GROUP BY m.content_hash,
                             COALESCE('id:' || m.sender_id,
                                      'name:' || m.sender_name,
                                      'anonymous')
               )
            """
        )

    def propagate_duplicates(self, *, prompt_version: str = ENTITY_EXTRACTION_VERSION) -> int:
        """Activate crossposts from the one copy sent to the model."""
        conn = self.conn
        twins = conn.execute(
            """
            SELECT dj.corpus_id AS dup_id, min(sj.corpus_id) AS src_id,
                   source_state.status AS status
              FROM extraction_jobs dj
              JOIN corpus_messages duplicate_message ON duplicate_message.corpus_id = dj.corpus_id
              JOIN corpus_messages source_message
                ON source_message.content_hash = duplicate_message.content_hash
               AND COALESCE('id:' || source_message.sender_id,
                            'name:' || source_message.sender_name,
                            'anonymous')
                   = COALESCE('id:' || duplicate_message.sender_id,
                              'name:' || duplicate_message.sender_name,
                              'anonymous')
              JOIN extraction_jobs sj ON sj.corpus_id = source_message.corpus_id
              JOIN extraction_state source_state ON source_state.corpus_id = sj.corpus_id
             WHERE dj.prompt_version = ? AND dj.status IN ('pending', 'error')
               AND sj.prompt_version = dj.prompt_version AND sj.status = 'succeeded'
               AND sj.corpus_id < dj.corpus_id
             GROUP BY dj.corpus_id
            """,
            (prompt_version,),
        ).fetchall()
        settled = 0
        for row in twins:
            with conn:
                old_ids = {
                    int(item["place_id"])
                    for item in conn.execute(
                        "SELECT place_id FROM place_mentions WHERE corpus_id = ?",
                        (row["dup_id"],),
                    )
                }
                # Same bookkeeping as record_extraction: descriptors of the rows
                # deleted here are invisible to the scoped recompute afterwards.
                touched_descriptors = {
                    int(item["descriptor_id"])
                    for item in conn.execute(
                        "SELECT descriptor_id FROM place_mentions "
                        "WHERE corpus_id = ? AND descriptor_id IS NOT NULL",
                        (row["dup_id"],),
                    )
                }
                conn.execute("DELETE FROM place_mentions WHERE corpus_id = ?", (row["dup_id"],))
                conn.execute(
                    """
                    INSERT INTO place_mentions (
                        place_id, corpus_id, event_types, evidence_quote, confidence,
                        extracted_by, descriptor_id, descriptor_raw, offerings_raw,
                        entity_kind_raw, access_modes_raw, extractor_version
                    )
                    SELECT place_id, ?, event_types, evidence_quote, confidence,
                           extracted_by, descriptor_id, descriptor_raw, offerings_raw,
                           entity_kind_raw, access_modes_raw, extractor_version
                      FROM place_mentions WHERE corpus_id = ?
                    """,
                    (row["dup_id"], row["src_id"]),
                )
                new_ids = {
                    int(item["place_id"])
                    for item in conn.execute(
                        "SELECT place_id FROM place_mentions WHERE corpus_id = ?",
                        (row["dup_id"],),
                    )
                }
                affected = old_ids | new_ids
                self._refresh_entity_aggregates(affected, descriptor_ids=touched_descriptors)
                if affected:
                    marks = ",".join("?" for _ in affected)
                    conn.execute(
                        f"DELETE FROM places WHERE mention_count=0 AND place_id IN ({marks})",  # noqa: S608
                        tuple(affected),
                    )
                now = _now()
                conn.execute(
                    """
                    UPDATE extraction_state
                       SET status=?, attempted_at=?, error=NULL, active_prompt_version=?
                     WHERE corpus_id=?
                    """,
                    (row["status"], now, prompt_version, row["dup_id"]),
                )
                conn.execute(
                    """
                    UPDATE extraction_jobs
                       SET status='succeeded', attempted_at=?, completed_at=?, error=NULL
                     WHERE corpus_id=? AND prompt_version=?
                    """,
                    (now, now, row["dup_id"], prompt_version),
                )
            settled += 1
        return settled

    def _resolve_sender_names(self) -> int:
        """Give a name to archived messages whose sender is named elsewhere.

        History pages used to arrive without their senders' names, so most of
        the archive is anonymous while the same people are named in the live
        stream. One id seen with a name anywhere names every one of that
        person's messages, which is what makes "who keeps announcing events
        here" answerable over two years instead of over last week.

        The crawler now records the name at capture time, so this shrinks to
        nothing for anything fetched from here on.

        The map is built once and applied per sender. Expressed instead as a
        correlated subquery -- one lookup into this same table per row -- it is
        quadratic over sixty-five thousand rows and does not finish inside the
        indexer's window, which reads from the outside as "recovered nothing".
        """
        conn = self.conn
        names = conn.execute(
            """
            SELECT sender_id,
                   COALESCE(
                       max(CASE WHEN sender_name LIKE '@%' THEN sender_name END),
                       max(sender_name)
                   ) AS sender_name
              FROM corpus_messages
             WHERE sender_name IS NOT NULL AND sender_id IS NOT NULL
             GROUP BY sender_id
            """
        ).fetchall()
        if not names:
            return 0
        updated = 0
        for row in names:
            cursor = conn.execute(
                """
                UPDATE corpus_messages SET sender_name = ?
                 WHERE sender_id = ?
                   AND (
                       sender_name IS NULL
                       OR (? LIKE '@%' AND sender_name NOT LIKE '@%')
                   )
                """,
                (row["sender_name"], row["sender_id"], row["sender_name"]),
            )
            updated += max(cursor.rowcount, 0)
        return updated

    def _extract_contacts(self, *, batch_size: int = 20000) -> int:
        """Mine contact handles out of every message not yet scanned.

        The cursor is the highest corpus_id already scanned, kept in the same
        table as the ingest cursors. A message's text never changes after
        insert, so a scanned row can never gain a contact later and rescanning
        would only re-derive what is already stored.
        """
        conn = self.conn
        row = conn.execute("SELECT last_id FROM sync_state WHERE source = 'contacts'").fetchone()
        cursor_id = int(row["last_id"]) if row else 0
        written = 0
        while True:
            rows = conn.execute(
                """
                SELECT corpus_id, text FROM corpus_messages
                 WHERE corpus_id > ? ORDER BY corpus_id LIMIT ?
                """,
                (cursor_id, batch_size),
            ).fetchall()
            if not rows:
                break
            payload = [
                (r["corpus_id"], kind, value, display)
                for r in rows
                for kind, value, display in extract_contacts(r["text"])
            ]
            if payload:
                conn.executemany(
                    """
                    INSERT INTO message_contacts (corpus_id, kind, value, display)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(corpus_id, kind, value) DO NOTHING
                    """,
                    payload,
                )
                written += len(payload)
            cursor_id = int(rows[-1]["corpus_id"])
            self._save_cursor("contacts", cursor_id, written, 0)
            conn.commit()
            if len(rows) < batch_size:
                break
        return written

    def _refresh_place_counts(self) -> None:
        """Recount mentions and re-derive the seen-dates for every place."""
        self.conn.execute(
            """
            UPDATE places SET
                mention_count = (SELECT count(*) FROM place_mentions pm
                                  WHERE pm.place_id = places.place_id),
                first_seen_at = (SELECT min(m.date) FROM place_mentions pm
                                   JOIN corpus_messages m USING(corpus_id)
                                  WHERE pm.place_id = places.place_id),
                last_seen_at  = (SELECT max(m.date) FROM place_mentions pm
                                   JOIN corpus_messages m USING(corpus_id)
                                  WHERE pm.place_id = places.place_id)
            """
        )

    def record_extraction_cost(
        self,
        *,
        model: str,
        calls: int,
        messages: int,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Store what one extraction run actually spent."""
        self.conn.execute(
            """
            INSERT INTO extraction_cost (model, calls, messages, input_tokens,
                                         cached_input_tokens, output_tokens)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model, calls, messages, input_tokens, cached_input_tokens, output_tokens),
        )
        self.conn.commit()

    def contacts_for_places(
        self,
        place_ids: Sequence[int],
        *,
        limit_per_place: int = 8,
    ) -> dict[int, list[dict[str, Any]]]:
        """Contacts published in the messages that mention each place.

        Ranked by how many distinct messages carry them: a handle repeated
        across five announcements is the venue's own, a handle appearing once
        is whoever happened to comment that day.
        """
        if not place_ids:
            return {}
        marks = ",".join("?" * len(place_ids))
        rows = self.conn.execute(
            f"""
            SELECT pm.place_id, mc.kind, mc.value,
                   count(DISTINCT mc.corpus_id) AS seen,
                   max(m.date) AS last_seen,
                   min(mc.display) AS display
              FROM place_mentions pm
              JOIN message_contacts mc ON mc.corpus_id = pm.corpus_id
              JOIN corpus_messages m ON m.corpus_id = mc.corpus_id
             WHERE pm.place_id IN ({marks})
             GROUP BY pm.place_id, mc.kind, mc.value
             ORDER BY pm.place_id, seen DESC, last_seen DESC
            """,  # noqa: S608 -- `marks` is generated placeholders, values are bound
            list(place_ids),
        ).fetchall()
        out: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            bucket = out.setdefault(row["place_id"], [])
            if len(bucket) < limit_per_place:
                bucket.append(
                    {
                        "kind": row["kind"],
                        "value": row["value"],
                        "display": row["display"],
                        "seen_in_messages": row["seen"],
                        "last_seen": row["last_seen"],
                    }
                )
        return out

    def authors_for_places(
        self,
        place_ids: Sequence[int],
        *,
        limit_per_place: int = 6,
    ) -> dict[int, list[dict[str, Any]]]:
        """Who posted the messages that mention each place.

        The poster is a contact the message body never states: whoever keeps
        announcing a venue's events is reachable through the chat itself, and
        that is often the only route to an organiser who published no handle.
        """
        if not place_ids:
            return {}
        marks = ",".join("?" * len(place_ids))
        # The link has to come from ONE message. Aggregating chat_id and
        # telegram_msg_id independently pairs the largest chat id with the
        # largest message id, which for an author who posted in two chats is a
        # link to a message that does not exist -- and 93 of this corpus's
        # author/place pairs span more than one chat.
        rows = self.conn.execute(
            f"""
            WITH mentioned AS (
                SELECT pm.place_id, m.sender_id, m.sender_name, m.date,
                       m.chat_id, m.telegram_msg_id,
                       -- Identity for grouping: the numeric id when there is
                       -- one, the name otherwise. SQLite groups all NULLs
                       -- together, which would merge every senderless channel
                       -- post into a single invented author.
                       COALESCE('id:' || m.sender_id, 'name:' || m.sender_name) AS who,
                       ROW_NUMBER() OVER (
                           PARTITION BY pm.place_id,
                                        COALESCE('id:' || m.sender_id, 'name:' || m.sender_name)
                           ORDER BY m.date DESC, m.corpus_id DESC
                       ) AS recency
                  FROM place_mentions pm
                  JOIN corpus_messages m ON m.corpus_id = pm.corpus_id
                 WHERE pm.place_id IN ({marks}) AND m.sender_name IS NOT NULL
            )
            SELECT place_id, who,
                   count(*) AS posts,
                   max(date) AS last_post,
                   max(CASE WHEN recency = 1 THEN sender_id END) AS sender_id,
                   max(CASE WHEN recency = 1 THEN sender_name END) AS sender_name,
                   max(CASE WHEN recency = 1 THEN chat_id END) AS chat_id,
                   max(CASE WHEN recency = 1 THEN telegram_msg_id END) AS telegram_msg_id
              FROM mentioned
             GROUP BY place_id, who
             ORDER BY place_id, posts DESC, last_post DESC
            """,  # noqa: S608 -- `marks` is generated placeholders, values are bound
            list(place_ids),
        ).fetchall()
        out: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            bucket = out.setdefault(row["place_id"], [])
            if len(bucket) < limit_per_place:
                bucket.append(
                    {
                        "name": row["sender_name"],
                        "sender_id": row["sender_id"],
                        "posts": row["posts"],
                        "last_post": row["last_post"],
                        "last_message_link": message_link(row["chat_id"], row["telegram_msg_id"]),
                    }
                )
        return out

    # ------------------------------------------------------------------
    # Lexical lane
    # ------------------------------------------------------------------

    def lexical_search(
        self,
        match_query: str,
        *,
        limit: int = 200,
        chat_ids: Sequence[int] | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[CorpusRow]:
        """Rank messages by BM25 against an FTS5 MATCH expression."""
        sql = [
            """
            SELECT m.corpus_id, m.chat_id, m.telegram_msg_id,
                   COALESCE(m.chat_title, c.title) AS chat_title, m.sender_name,
                   m.text, m.date, m.content_hash, m.source, m.reply_to_message_id,
                   bm25(corpus_fts) AS rank
              FROM corpus_fts
              JOIN corpus_messages m ON m.corpus_id = corpus_fts.rowid
              LEFT JOIN corpus_chats c ON c.chat_id = m.chat_id
             WHERE corpus_fts MATCH ?
            """
        ]
        params: list[Any] = [match_query]
        sql, params = _apply_filters(sql, params, chat_ids, since, until)
        sql.append(" ORDER BY rank LIMIT ?")
        params.append(limit)
        try:
            rows = self.conn.execute("".join(sql), params).fetchall()
        except sqlite3.OperationalError as exc:
            raise ValueError(f"invalid full-text query: {exc}") from exc
        return [_row_to_corpus(r, lane="lexical") for r in rows]

    # ------------------------------------------------------------------
    # Semantic lane
    # ------------------------------------------------------------------

    def _load_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (corpus_ids, unit-normalized matrix), cached per connection."""
        if self._vector_cache is not None:
            return self._vector_cache
        rows = self.conn.execute(
            "SELECT corpus_id, dim, vec FROM corpus_embeddings ORDER BY corpus_id"
        ).fetchall()
        if not rows:
            empty = (np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32))
            self._vector_cache = empty
            return empty
        dim = int(rows[0]["dim"])
        ids = np.fromiter((r["corpus_id"] for r in rows), dtype=np.int64, count=len(rows))
        flat = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float32)
        matrix = flat.reshape(len(rows), dim)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vector_cache = (ids, matrix / norms)
        return self._vector_cache

    def semantic_search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 200,
        chat_ids: Sequence[int] | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[CorpusRow]:
        """Rank messages by cosine similarity to a query embedding.

        Brute force over the whole matrix: one matmul across tens of thousands
        of vectors costs single-digit milliseconds, which is far below the
        latency of the call that produced the query embedding in the first
        place. An ANN index would add a dependency to optimize a step that is
        not the bottleneck.
        """
        ids, matrix = self._load_vectors()
        if ids.size == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        scores = matrix @ (query / norm)

        # Over-fetch before filtering: the filter runs in SQL against the ids
        # we return, so taking exactly `limit` here would let a chat/date
        # filter empty the result even when matching messages exist.
        take = min(len(ids), max(limit * 5, limit + 200))
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        ordered_ids = [int(ids[i]) for i in top]
        score_by_id = {int(ids[i]): float(scores[i]) for i in top}

        # json_each keeps the id list a bound parameter instead of splicing a
        # run of placeholders into the SQL text.
        sql = [
            """
            SELECT m.corpus_id, m.chat_id, m.telegram_msg_id,
                   COALESCE(m.chat_title, c.title) AS chat_title, m.sender_name,
                   m.text, m.date, m.content_hash, m.source, m.reply_to_message_id
              FROM corpus_messages m
              LEFT JOIN corpus_chats c ON c.chat_id = m.chat_id
             WHERE m.corpus_id IN (SELECT value FROM json_each(?))
            """
        ]
        params: list[Any] = [json.dumps(ordered_ids)]
        sql, params = _apply_filters(sql, params, chat_ids, since, until)
        rows = self.conn.execute("".join(sql), params).fetchall()

        out = []
        for r in rows:
            item = _row_to_corpus(r, lane="semantic")
            item.score = score_by_id.get(item.corpus_id, 0.0)
            out.append(item)
        out.sort(key=lambda x: -x.score)
        return out[:limit]

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        *,
        match_query: str | None,
        query_vector: Sequence[float] | None,
        limit: int = 40,
        pool: int = 200,
        chat_ids: Sequence[int] | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[CorpusRow]:
        """Fuse the two lanes with RRF, then roll up crossposts.

        RRF rather than a weighted blend of the raw scores: BM25 and cosine
        similarity are not on comparable scales, and rank fusion does not
        require them to be. A message that is strong in only one lane still
        surfaces, which is the whole point -- lexical is the precision anchor
        and semantic is the recall net for posts that use none of the expected
        vocabulary.
        """
        lanes: list[tuple[str, list[CorpusRow]]] = []
        if match_query:
            lanes.append(
                (
                    "lexical",
                    self.lexical_search(
                        match_query, limit=pool, chat_ids=chat_ids, since=since, until=until
                    ),
                )
            )
        if query_vector is not None:
            lanes.append(
                (
                    "semantic",
                    self.semantic_search(
                        query_vector, limit=pool, chat_ids=chat_ids, since=since, until=until
                    ),
                )
            )
        if not lanes:
            return []

        fused: dict[int, CorpusRow] = {}
        for lane_name, hits in lanes:
            for rank, item in enumerate(hits, start=1):
                existing = fused.get(item.corpus_id)
                if existing is None:
                    item.score = 0.0
                    item.lanes = []
                    fused[item.corpus_id] = existing = item
                existing.score += 1.0 / (RRF_K + rank)
                if lane_name not in existing.lanes:
                    existing.lanes.append(lane_name)

        rows = _rollup_duplicates(sorted(fused.values(), key=lambda x: -x.score), limit)
        self.attach_reply_context(rows)
        return rows

    def attach_reply_context(self, rows: Sequence[CorpusRow], *, text_limit: int = 300) -> None:
        """Fill ``in_reply_to`` on every row that answers a message we hold.

        One query for the whole page rather than one per row: a page is at
        most sixty rows, and the parent lookup is by the (chat, message id)
        unique index, so the cost is one index probe per reply.
        """
        wanted = [
            (row.chat_id, row.reply_to_message_id)
            for row in rows
            if row.reply_to_message_id is not None
        ]
        if not wanted:
            return
        placeholders = " OR ".join("(chat_id = ? AND telegram_msg_id = ?)" for _ in wanted)
        params = [value for pair in wanted for value in pair]
        parents = {
            (int(r["chat_id"]), int(r["telegram_msg_id"])): r
            for r in self.conn.execute(
                # Only "?" placeholders are interpolated; every value stays bound.
                f"SELECT chat_id, telegram_msg_id, sender_name, date, text "  # noqa: S608
                f"FROM corpus_messages WHERE {placeholders}",
                params,
            )
        }
        for row in rows:
            if row.reply_to_message_id is None:
                continue
            parent = parents.get((row.chat_id, row.reply_to_message_id))
            if parent is None:
                continue
            body = str(parent["text"])
            if len(body) > text_limit:
                body = body[:text_limit].rstrip() + "…"
            row.in_reply_to = {
                "sender": parent["sender_name"],
                "date": parent["date"],
                "text": body,
                "message_link": message_link(row.chat_id, row.reply_to_message_id),
            }

    # ------------------------------------------------------------------
    # Embedding bookkeeping
    # ------------------------------------------------------------------

    def pending_embeddings(self, *, limit: int, min_length: int = 1) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT m.corpus_id, m.text FROM corpus_messages m
             LEFT JOIN corpus_embeddings e ON e.corpus_id = m.corpus_id
             WHERE e.corpus_id IS NULL AND length(m.text) >= ?
             ORDER BY m.corpus_id LIMIT ?
            """,
            (min_length, limit),
        ).fetchall()

    def store_embeddings(self, items: Iterable[tuple[int, Sequence[float]]], *, model: str) -> int:
        payload = []
        for corpus_id, vector in items:
            arr = np.asarray(vector, dtype=np.float32)
            payload.append((corpus_id, model, int(arr.shape[0]), arr.tobytes()))
        if not payload:
            return 0
        self.conn.executemany(
            """
            INSERT INTO corpus_embeddings (corpus_id, model, dim, vec)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(corpus_id) DO UPDATE SET
                model = excluded.model, dim = excluded.dim, vec = excluded.vec,
                embedded_at = CURRENT_TIMESTAMP
            """,
            payload,
        )
        self.conn.commit()
        self._vector_cache = None
        return len(payload)

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def pending_extractions(
        self,
        limit: int,
        *,
        prompt_version: str = ENTITY_EXTRACTION_VERSION,
        retry_after_hours: int = 6,
        max_attempts: int = MAX_EXTRACTION_ATTEMPTS,
    ) -> list[sqlite3.Row]:
        """Return canonical versioned jobs, including retryable failures."""
        self._seed_extraction_jobs(prompt_version)
        return self.conn.execute(
            """
            SELECT m.corpus_id, m.text, m.date, m.chat_id, c.title AS chat_title,
                   m.sender_id, m.sender_name, s.status AS source_status
              FROM extraction_jobs j
              JOIN extraction_state s ON s.corpus_id = j.corpus_id
              JOIN corpus_messages m ON m.corpus_id = j.corpus_id
              LEFT JOIN corpus_chats c ON c.chat_id = m.chat_id
             WHERE j.prompt_version = ?
               AND (
                    j.status = 'pending'
                    OR (
                    j.status = 'error'
                    AND j.attempts < ?
                    AND (
                        j.not_before IS NULL OR datetime(j.not_before) <= datetime('now')
                    )
                    AND (
                        j.attempted_at IS NULL
                        OR datetime(j.attempted_at) <= datetime('now', ?)
                    )
                )
                    OR (
                        j.status = 'running'
                        AND j.attempts < ?
                        AND datetime(j.attempted_at) <= datetime('now', ?)
                    )
               )
               AND NOT EXISTS (
                   SELECT 1
                     FROM corpus_messages earlier
                     JOIN extraction_jobs ej ON ej.corpus_id = earlier.corpus_id
                    WHERE earlier.content_hash = m.content_hash
                      AND COALESCE('id:' || earlier.sender_id,
                                   'name:' || earlier.sender_name,
                                   'anonymous')
                          = COALESCE('id:' || m.sender_id,
                                     'name:' || m.sender_name,
                                     'anonymous')
                      AND ej.prompt_version = j.prompt_version
                      AND earlier.corpus_id < m.corpus_id
               )
             ORDER BY j.status = 'error', m.date DESC LIMIT ?
            """,
            (
                prompt_version,
                max_attempts,
                f"-{retry_after_hours} hours",
                max_attempts,
                f"-{retry_after_hours} hours",
                limit,
            ),
        ).fetchall()

    def mark_extractions_running(self, corpus_ids: Sequence[int], *, prompt_version: str) -> None:
        """Lease a bounded job chunk before making provider calls."""
        if not corpus_ids:
            return
        now = _now()
        with self.conn:
            self.conn.executemany(
                """
                UPDATE extraction_jobs SET status='running', attempted_at=?
                 WHERE corpus_id=? AND prompt_version=? AND status IN ('pending', 'error', 'running')
                """,
                ((now, corpus_id, prompt_version) for corpus_id in corpus_ids),
            )

    def extractions_for_statuses(
        self,
        statuses: Sequence[str],
        *,
        limit: int,
        prompt_version: str = ENTITY_EXTRACTION_VERSION,
    ) -> list[sqlite3.Row]:
        """Return one bounded snapshot of explicitly selected settled states."""
        selected = tuple(dict.fromkeys(statuses))
        invalid = set(selected) - REEXTRACTABLE_STATUSES
        if not selected or invalid:
            allowed = ", ".join(sorted(REEXTRACTABLE_STATUSES))
            bad = ", ".join(sorted(invalid)) or "empty selection"
            raise ValueError(f"unsupported extraction status ({bad}); choose from: {allowed}")
        if limit <= 0:
            return []
        self._seed_extraction_jobs(prompt_version)
        marks = ",".join("?" for _ in selected)
        per_status = (limit + len(selected) - 1) // len(selected)
        return self.conn.execute(
            f"""
            WITH candidates AS (
                SELECT m.corpus_id, m.text, m.date, m.chat_id,
                       c.title AS chat_title, m.sender_id, m.sender_name,
                       s.status AS source_status,
                       row_number() OVER (
                           PARTITION BY s.status ORDER BY m.date DESC, m.corpus_id DESC
                       ) AS stratum_rank
                  FROM extraction_jobs j
                  JOIN extraction_state s ON s.corpus_id = j.corpus_id
                  JOIN corpus_messages m ON m.corpus_id = s.corpus_id
                  LEFT JOIN corpus_chats c ON c.chat_id = m.chat_id
                 WHERE j.prompt_version = ? AND j.status IN ('pending', 'error')
                   AND s.status IN ({marks})
                   AND NOT EXISTS (
                       SELECT 1
                         FROM corpus_messages earlier
                         JOIN extraction_jobs ej ON ej.corpus_id = earlier.corpus_id
                        WHERE earlier.content_hash = m.content_hash
                          AND COALESCE('id:' || earlier.sender_id,
                                       'name:' || earlier.sender_name,
                                       'anonymous')
                              = COALESCE('id:' || m.sender_id,
                                         'name:' || m.sender_name,
                                         'anonymous')
                          AND ej.prompt_version = j.prompt_version
                          AND earlier.corpus_id < m.corpus_id
                   )
            )
            SELECT corpus_id, text, date, chat_id, chat_title, sender_id, sender_name,
                   source_status
              FROM candidates WHERE stratum_rank <= ?
             ORDER BY stratum_rank, date DESC, corpus_id DESC LIMIT ?
            """,  # noqa: S608 -- generated placeholders; all values are bound
            (prompt_version, *selected, per_status, limit),
        ).fetchall()

    def record_extraction(
        self,
        corpus_id: int,
        places: Sequence[dict[str, Any]],
        *,
        model: str,
        prompt_version: str = ENTITY_EXTRACTION_VERSION,
        descriptor_embedding_model: str | None = None,
        error: str | None = None,
    ) -> int:
        """Atomically replace one message's active entity extraction."""
        conn = self.conn
        if error is not None:
            attempts = 1 if is_transient_error(error) else MAX_EXTRACTION_ATTEMPTS
            with conn:
                conn.execute(
                    """
                    INSERT INTO extraction_jobs (corpus_id, prompt_version)
                    VALUES (?, ?) ON CONFLICT(corpus_id, prompt_version) DO NOTHING
                    """,
                    (corpus_id, prompt_version),
                )
                conn.execute(
                    """
                    UPDATE extraction_jobs
                       SET status='error', attempts=MAX(attempts + 1, ?),
                           attempted_at=?, error=?
                     WHERE corpus_id=? AND prompt_version=?
                    """,
                    (attempts, _now(), error[:500], corpus_id, prompt_version),
                )
                conn.execute(
                    "UPDATE extraction_state SET error=? WHERE corpus_id=?",
                    (error[:500], corpus_id),
                )
                conn.execute(
                    """
                    UPDATE extraction_jobs
                       SET status='error', attempts=?, attempted_at=?, error=?
                     WHERE prompt_version=? AND corpus_id <> ?
                       AND corpus_id IN (
                           SELECT duplicate.corpus_id
                             FROM corpus_messages source
                             JOIN corpus_messages duplicate
                               ON duplicate.content_hash = source.content_hash
                              AND COALESCE('id:' || duplicate.sender_id,
                                           'name:' || duplicate.sender_name,
                                           'anonymous')
                                  = COALESCE('id:' || source.sender_id,
                                             'name:' || source.sender_name,
                                             'anonymous')
                            WHERE source.corpus_id = ?
                       )
                       AND (SELECT attempts FROM extraction_jobs
                             WHERE corpus_id=? AND prompt_version=?) >= ?
                    """,
                    (
                        MAX_EXTRACTION_ATTEMPTS,
                        _now(),
                        f"duplicate source failed: {error}"[:500],
                        prompt_version,
                        corpus_id,
                        corpus_id,
                        corpus_id,
                        prompt_version,
                        MAX_EXTRACTION_ATTEMPTS,
                    ),
                )
            return 0

        validated = self._validate_entities(corpus_id, places)
        with conn:
            conn.execute(
                """
                INSERT INTO extraction_jobs (corpus_id, prompt_version)
                VALUES (?, ?) ON CONFLICT(corpus_id, prompt_version) DO NOTHING
                """,
                (corpus_id, prompt_version),
            )
            old_ids = {
                int(row["place_id"])
                for row in conn.execute(
                    "SELECT place_id FROM place_mentions WHERE corpus_id = ?", (corpus_id,)
                )
            }
            # Descriptors of the mentions about to be deleted: after the DELETE
            # nothing else remembers that their counts must shrink.
            touched_descriptors = {
                int(row["descriptor_id"])
                for row in conn.execute(
                    "SELECT descriptor_id FROM place_mentions "
                    "WHERE corpus_id = ? AND descriptor_id IS NOT NULL",
                    (corpus_id,),
                )
            }
            conn.execute("DELETE FROM place_mentions WHERE corpus_id = ?", (corpus_id,))
            new_ids: set[int] = set()
            for entity in validated:
                place_id = self._upsert_entity(corpus_id, entity)
                # The extractor sometimes returns the same entity twice for one message, and both
                # copies fold to one canonical. The DELETE above already cleared this message's
                # old mentions, so a second insert here can only be that self-collision -- which
                # aborted a whole production run on UNIQUE(place_id, corpus_id) (2026-08-12,
                # after 452 messages). Skipping is also the right semantics: one message
                # vouching for one place is one mention, however many times it says the name.
                if place_id in new_ids:
                    continue
                new_ids.add(place_id)
                descriptor_id = self._upsert_descriptor(
                    entity, embedding_model=descriptor_embedding_model
                )
                if descriptor_id is not None:
                    touched_descriptors.add(int(descriptor_id))
                conn.execute(
                    """
                    INSERT INTO place_mentions (
                        place_id, corpus_id, event_types, evidence_quote, confidence,
                        extracted_by, descriptor_id, descriptor_raw, offerings_raw,
                        entity_kind_raw, access_modes_raw, extractor_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        place_id,
                        corpus_id,
                        json.dumps(entity["event_types"], ensure_ascii=False),
                        entity["evidence"],
                        entity["confidence"],
                        model,
                        descriptor_id,
                        entity["descriptor"],
                        json.dumps(entity["offerings"], ensure_ascii=False),
                        entity["entity_kind"],
                        json.dumps(entity["access_modes"], ensure_ascii=False),
                        prompt_version,
                    ),
                )
            affected = old_ids | new_ids
            self._refresh_entity_aggregates(affected, descriptor_ids=touched_descriptors)
            if affected:
                marks = ",".join("?" for _ in affected)
                conn.execute(
                    f"DELETE FROM places WHERE mention_count = 0 AND place_id IN ({marks})",  # noqa: S608
                    tuple(affected),
                )
            state = "extracted" if validated else "no_venue"
            now = _now()
            conn.execute(
                """
                UPDATE extraction_state
                   SET status=?, attempts=attempts+1, attempted_at=?, error=NULL,
                       active_prompt_version=?
                 WHERE corpus_id=?
                """,
                (state, now, prompt_version, corpus_id),
            )
            conn.execute(
                """
                UPDATE extraction_jobs
                   SET status='succeeded', attempts=attempts+1, attempted_at=?,
                       completed_at=?, error=NULL
                 WHERE corpus_id=? AND prompt_version=?
                """,
                (now, now, corpus_id, prompt_version),
            )
        return len(validated)

    def _validate_entities(
        self, corpus_id: int, entities: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Validate and normalize a provider result without changing state."""
        message = self.conn.execute(
            "SELECT text, sender_name FROM corpus_messages WHERE corpus_id = ?", (corpus_id,)
        ).fetchone()
        if message is None:
            raise ValueError(f"unknown corpus_id {corpus_id}")
        contacts = {
            (str(row["kind"]), str(row["value"]), str(row["display"]))
            for row in self.conn.execute(
                "SELECT kind, value, display FROM message_contacts WHERE corpus_id = ?",
                (corpus_id,),
            )
        }
        contacts.update(extract_contacts(str(message["text"])))
        contacts.update(extract_contacts(str(message["sender_name"] or "")))
        contact_displays = tuple(contact[2] for contact in contacts)

        validated: list[dict[str, Any]] = []
        for raw in entities:
            legacy = "entity_kind" not in raw
            name = str(raw.get("name") or "").strip()
            kind = str(raw.get("entity_kind") or "place")
            if kind not in ENTITY_KINDS:
                raise ValueError(f"invalid entity_kind: {kind}")
            aliases_value = raw.get("aliases") or []
            if not isinstance(aliases_value, list):
                raise ValueError("aliases must be an array")
            aliases = sorted(
                {
                    str(value).strip()
                    for value in aliases_value
                    if 2 <= len(str(value).strip()) <= 120
                }
            )
            identity_text = "\n".join([name, *aliases, str(raw.get("evidence") or "")])
            matched_contacts = sorted(
                contact for contact in contacts if contact[2] and contact[2] in identity_text
            )
            if not is_entity_label(name, entity_kind=kind, contact_displays=contact_displays):
                logger.debug("Rejected extracted entity label %r from %s", name, corpus_id)
                continue

            city = str(raw.get("city_area") or "").strip() or "unknown"
            folded_name = fold_ascii(name)
            if name.casefold() != folded_name and folded_name not in aliases:
                aliases.append(folded_name)
            if kind == "place":
                canonical = folded_name
                fallback_canonical = canonical
            elif matched_contacts:
                contact_kind, contact_value, _display = matched_contacts[0]
                canonical = f"{kind}:contact:{contact_kind}:{contact_value}"
                fallback_canonical = f"{kind}:name:{folded_name}:city:{fold_ascii(city)}"
            else:
                if city.casefold() == "unknown":
                    logger.debug(
                        "Rejected contactless non-place %r without a city from %s",
                        name,
                        corpus_id,
                    )
                    continue
                canonical = f"{kind}:name:{folded_name}:city:{fold_ascii(city)}"
                fallback_canonical = canonical
            if not folded_name or len(canonical) < 2:
                continue

            descriptor = str(raw.get("descriptor") or raw.get("place_type") or "").strip()
            if not descriptor or len(descriptor) > 120:
                if legacy:
                    descriptor = "place"
                else:
                    raise ValueError("descriptor must contain 1-120 characters")
            offerings_value = raw.get("offerings", raw.get("event_types", [])) or []
            if not isinstance(offerings_value, list):
                raise ValueError("offerings must be an array")
            offerings = [str(value).strip() for value in offerings_value]
            if any(not 2 <= len(value) <= 120 for value in offerings):
                raise ValueError("offering phrases must contain 2-120 characters")
            modes_value = raw.get("access_modes") or (["visit"] if kind == "place" else ["unknown"])
            if not isinstance(modes_value, list):
                raise ValueError("access_modes must be an array")
            access_modes = sorted({str(value) for value in modes_value})
            if not access_modes or set(access_modes) - ACCESS_MODES:
                raise ValueError(f"invalid access_modes: {access_modes}")
            evidence = str(raw.get("evidence") or "")
            if len(evidence) > 200:
                raise ValueError("evidence exceeds 200 characters")
            confidence = float(raw.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            derived_type, derived_events = derive_legacy_facets(descriptor, offerings)
            if legacy:
                derived_type = str(raw.get("place_type") or derived_type)
                derived_events = sorted({str(value) for value in raw.get("event_types") or []})
            validated.append(
                {
                    "name": name,
                    "aliases": sorted(set(aliases)),
                    "entity_kind": kind,
                    "access_modes": access_modes,
                    "descriptor": descriptor,
                    "descriptor_normalized": normalize_descriptor(descriptor),
                    "descriptor_language": str(raw.get("descriptor_language") or "und"),
                    "offerings": offerings,
                    "city_area": city,
                    "evidence": evidence,
                    "confidence": confidence,
                    "canonical": canonical,
                    "fallback_canonical": fallback_canonical,
                    "place_type": derived_type,
                    "event_types": derived_events,
                }
            )
        return validated

    def _upsert_entity(self, corpus_id: int, entity: dict[str, Any]) -> int:
        """Upsert one aggregate, merging a unique name+city row into its contact key."""
        canonical = str(entity["canonical"])
        row = self.conn.execute(
            "SELECT place_id, name, aliases FROM places WHERE canonical = ?", (canonical,)
        ).fetchone()
        fallback = str(entity["fallback_canonical"])
        if row is None and fallback != canonical:
            candidate = self.conn.execute(
                "SELECT place_id, name, aliases FROM places WHERE canonical = ?", (fallback,)
            ).fetchone()
            if candidate is not None:
                self.conn.execute(
                    "UPDATE places SET canonical = ? WHERE place_id = ?",
                    (canonical, candidate["place_id"]),
                )
                row = candidate

        aliases = set(entity["aliases"])
        if row is None:
            cursor = self.conn.execute(
                """
                INSERT INTO places (
                    canonical, name, aliases, city_area, place_type,
                    first_seen_at, last_seen_at, entity_kind, access_modes
                ) VALUES (?, ?, ?, ?, ?,
                    (SELECT date FROM corpus_messages WHERE corpus_id = ?),
                    (SELECT date FROM corpus_messages WHERE corpus_id = ?), ?, ?)
                """,
                (
                    canonical,
                    entity["name"],
                    json.dumps(sorted(aliases), ensure_ascii=False),
                    entity["city_area"],
                    entity["place_type"],
                    corpus_id,
                    corpus_id,
                    entity["entity_kind"],
                    json.dumps(entity["access_modes"], ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid or 0)

        place_id = int(row["place_id"])
        aliases.update(json.loads(row["aliases"] or "[]"))
        aliases.add(str(entity["name"]))
        existing_name = str(row["name"])
        use_new_name = existing_name.startswith("@") or not any(
            character.isalpha() for character in existing_name
        )
        self.conn.execute(
            """
            UPDATE places SET
                name = CASE WHEN ? THEN ? ELSE name END,
                aliases = ?, city_area = COALESCE(NULLIF(city_area, 'unknown'), ?),
                entity_kind = ?,
                last_seen_at = MAX(COALESCE(last_seen_at, ''),
                    (SELECT date FROM corpus_messages WHERE corpus_id = ?))
             WHERE place_id = ?
            """,
            (
                use_new_name,
                entity["name"],
                json.dumps(sorted(aliases), ensure_ascii=False),
                entity["city_area"],
                entity["entity_kind"],
                corpus_id,
                place_id,
            ),
        )
        return place_id

    def _upsert_descriptor(self, entity: dict[str, Any], *, embedding_model: str | None) -> int:
        normalized = str(entity["descriptor_normalized"])
        self.conn.execute(
            """
            INSERT INTO descriptors (normalized, display_text, language)
            VALUES (?, ?, ?)
            ON CONFLICT(normalized) DO UPDATE SET
                language = COALESCE(descriptors.language, excluded.language),
                updated_at = CURRENT_TIMESTAMP
            """,
            (normalized, entity["descriptor"], entity["descriptor_language"]),
        )
        row = self.conn.execute(
            "SELECT descriptor_id FROM descriptors WHERE normalized = ?", (normalized,)
        ).fetchone()
        if row is None:
            raise RuntimeError("descriptor upsert did not return a row")
        descriptor_id = int(row["descriptor_id"])
        if embedding_model:
            self.conn.execute(
                """
                INSERT INTO descriptor_embeddings (descriptor_id, model)
                VALUES (?, ?) ON CONFLICT(descriptor_id, model) DO NOTHING
                """,
                (descriptor_id, embedding_model),
            )
        return descriptor_id

    def _refresh_entity_aggregates(
        self, place_ids: set[int], *, descriptor_ids: set[int] | None = None
    ) -> None:
        """Rebuild every projection touched by one atomic extraction replacement.

        ``descriptor_ids`` names descriptors whose mentions were REMOVED, which
        the mention rows can no longer reveal. Descriptors still attached to
        the affected places are collected on the walk below. Recomputing the
        whole descriptors table here instead used to burn fourteen minutes of
        CPU per index tick once the corpus passed 130k messages: every message
        recorded meant three correlated subqueries for each of 1 800
        descriptors (measured 2026-08-24, every tick killed by TimeoutStartSec
        for a day while the backlog stood still).
        """
        touched_descriptors: set[int] = set(descriptor_ids or ())
        for place_id in place_ids:
            mentions = self.conn.execute(
                """
                SELECT pm.descriptor_id, pm.offerings_raw, pm.access_modes_raw,
                       pm.event_types, pm.entity_kind_raw, pm.extractor_version,
                       m.date, d.normalized, d.display_text
                  FROM place_mentions pm
                  JOIN corpus_messages m USING(corpus_id)
                  LEFT JOIN descriptors d USING(descriptor_id)
                 WHERE pm.place_id = ?
                """,
                (place_id,),
            ).fetchall()
            if not mentions:
                self.conn.execute("UPDATE places SET mention_count=0 WHERE place_id=?", (place_id,))
                self.conn.execute("DELETE FROM place_descriptors WHERE place_id=?", (place_id,))
                continue

            descriptor_stats: dict[int, dict[str, Any]] = {}
            offerings: set[str] = set()
            access_modes: set[str] = set()
            legacy_place_type = self.conn.execute(
                "SELECT place_type FROM places WHERE place_id=?", (place_id,)
            ).fetchone()[0]
            has_legacy_descriptor = False
            for mention in mentions:
                descriptor_id = mention["descriptor_id"]
                if descriptor_id is not None:
                    touched_descriptors.add(int(descriptor_id))
                    stats = descriptor_stats.setdefault(
                        int(descriptor_id),
                        {
                            "count": 0,
                            "first": mention["date"],
                            "last": mention["date"],
                            "normalized": mention["normalized"],
                            "display": mention["display_text"],
                        },
                    )
                    stats["count"] += 1
                    stats["first"] = min(stats["first"], mention["date"])
                    stats["last"] = max(stats["last"], mention["date"])
                else:
                    has_legacy_descriptor = True
                raw_offerings = json.loads(mention["offerings_raw"] or "[]")
                if not raw_offerings and mention["extractor_version"] == "places-v2":
                    raw_offerings = json.loads(mention["event_types"] or "[]")
                offerings.update(normalize_descriptor(str(value)) for value in raw_offerings)
                access_modes.update(json.loads(mention["access_modes_raw"] or "[]"))

            self.conn.execute("DELETE FROM place_descriptors WHERE place_id=?", (place_id,))
            self.conn.executemany(
                """
                INSERT INTO place_descriptors (
                    place_id, descriptor_id, mention_count, first_seen_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (place_id, descriptor_id, data["count"], data["first"], data["last"])
                    for descriptor_id, data in descriptor_stats.items()
                ),
            )
            ordered = sorted(descriptor_stats.values(), key=lambda data: str(data["normalized"]))
            ordered.sort(key=lambda data: str(data["last"]), reverse=True)
            ordered.sort(key=lambda data: int(data["count"]), reverse=True)
            primary = str(ordered[0]["display"]) if ordered else None
            descriptor_text = " ".join(
                sorted(str(data["normalized"]) for data in descriptor_stats.values())
            )
            if (
                has_legacy_descriptor
                and legacy_place_type
                and str(legacy_place_type).casefold() != "other"
            ):
                descriptor_text = " ".join(
                    sorted({descriptor_text, normalize_descriptor(str(legacy_place_type))} - {""})
                )
            mode_values = sorted(access_modes) or ["unknown"]
            place_type, _events = derive_legacy_facets(descriptor_text, sorted(offerings))
            self.conn.execute(
                """
                UPDATE places SET mention_count=?, first_seen_at=?, last_seen_at=?,
                    primary_descriptor=?, descriptor_text=?, offering_text=?,
                    access_modes=?, place_type=?
                 WHERE place_id=?
                """,
                (
                    len(mentions),
                    min(str(mention["date"]) for mention in mentions),
                    max(str(mention["date"]) for mention in mentions),
                    primary,
                    descriptor_text,
                    " ".join(sorted(offerings)),
                    json.dumps(mode_values, ensure_ascii=False),
                    place_type,
                    place_id,
                ),
            )

        if not touched_descriptors:
            return
        marks = ",".join("?" for _ in touched_descriptors)
        self.conn.execute(
            f"""
            UPDATE descriptors SET
                mention_count = (SELECT count(*) FROM place_mentions pm
                                  WHERE pm.descriptor_id = descriptors.descriptor_id),
                first_seen_at = (SELECT min(m.date) FROM place_mentions pm
                                  JOIN corpus_messages m USING(corpus_id)
                                  WHERE pm.descriptor_id = descriptors.descriptor_id),
                last_seen_at = (SELECT max(m.date) FROM place_mentions pm
                                 JOIN corpus_messages m USING(corpus_id)
                                 WHERE pm.descriptor_id = descriptors.descriptor_id),
                updated_at = CURRENT_TIMESTAMP
             WHERE descriptor_id IN ({marks})
            """,  # noqa: S608 -- `marks` is generated placeholders, values are bound
            tuple(touched_descriptors),
        )

    def search_places(
        self,
        *,
        name_query: str | None = None,
        query: str | None = None,
        city_area: str | None = None,
        place_type: str | None = None,
        event_types: Sequence[str] | None = None,
        entity_kind: str | None = None,
        access_mode: str | None = None,
        min_mentions: int = 1,
        limit: int = 40,
        include_contacts: bool = True,
        expanded_fts: bool = False,
        semantic_enabled: bool = False,
        query_vector: Sequence[float] | None = None,
        embedding_model: str | None = None,
        semantic_cutoff: float = 0.55,
    ) -> list[dict[str, Any]]:
        """Query entities while preserving the legacy reader as the default."""
        name_query = name_query.strip() if name_query and name_query.strip() else None
        query = query.strip() if query and query.strip() else None
        if entity_kind is not None and entity_kind not in ENTITY_KINDS:
            raise ValueError(f"invalid entity_kind: {entity_kind}")
        if access_mode is not None and access_mode not in ACCESS_MODES:
            raise ValueError(f"invalid access_mode: {access_mode}")

        name_ids: set[int] | None = None
        if name_query:
            fts_table = "place_fts_next" if expanded_fts else "place_fts"
            expression = escape_fts_term(fold_ascii(name_query))
            if expanded_fts:
                expression = f"{{name aliases}} : {expression}"
            name_ids = {
                int(row["rowid"])
                for row in self.conn.execute(
                    f"SELECT rowid FROM {fts_table} WHERE {fts_table} MATCH ?",  # noqa: S608
                    (expression,),
                )
            }
            if not name_ids:
                return []

        lane_ranks: dict[str, list[int]] = {}
        if query and expanded_fts:
            terms = content_terms(query)
            if terms:
                expression = (
                    "{descriptor_text offering_text} : ("
                    + build_fts_query(terms, prefix=False)
                    + ")"
                )
                lane_ranks["lexical"] = [
                    int(row["rowid"])
                    for row in self.conn.execute(
                        """
                        SELECT rowid, bm25(place_fts_next, 8.0, 6.0, 3.0, 2.0) AS rank
                          FROM place_fts_next WHERE place_fts_next MATCH ?
                         ORDER BY rank LIMIT 100
                        """,
                        (expression,),
                    )
                ]
        if query and semantic_enabled and query_vector is not None and embedding_model:
            semantic_ids = self._semantic_place_ids(
                query_vector,
                model=embedding_model,
                cutoff=semantic_cutoff,
                descriptor_limit=50,
            )
            if semantic_ids:
                lane_ranks["semantic"] = semantic_ids

        scores: dict[int, float] = {}
        lanes_by_id: dict[int, list[str]] = {}
        for lane, place_ids in lane_ranks.items():
            for rank, place_id in enumerate(place_ids, start=1):
                scores[place_id] = scores.get(place_id, 0.0) + 1.0 / (RRF_K + rank)
                lanes_by_id.setdefault(place_id, []).append(lane)
        if query and not scores:
            return []

        sql = [
            """
            SELECT p.place_id, p.canonical, p.name, p.aliases, p.city_area, p.place_type,
                   p.mention_count, p.first_seen_at, p.last_seen_at, p.entity_kind,
                   p.access_modes, p.primary_descriptor, p.descriptor_text, p.offering_text
              FROM places p
             WHERE p.mention_count >= ?
            """
        ]
        params: list[Any] = [min_mentions]
        if name_ids is not None:
            marks = ",".join("?" for _ in name_ids)
            sql.append(f" AND p.place_id IN ({marks})")
            params.extend(name_ids)
        if query:
            candidate_ids = set(scores)
            marks = ",".join("?" for _ in candidate_ids)
            sql.append(f" AND p.place_id IN ({marks})")
            params.extend(candidate_ids)
        if city_area:
            sql.append(" AND lower(COALESCE(p.city_area,'')) LIKE ?")
            params.append(f"%{city_area.lower()}%")
        if place_type:
            sql.append(" AND lower(COALESCE(p.place_type,'')) LIKE ?")
            params.append(f"%{place_type.lower()}%")
        if event_types:
            clauses = []
            for et in event_types:
                clauses.append(
                    "EXISTS (SELECT 1 FROM place_mentions pm WHERE pm.place_id = p.place_id "
                    "AND lower(pm.event_types) LIKE ?)"
                )
                params.append(f"%{et.lower()}%")
            sql.append(" AND (" + " OR ".join(clauses) + ")")
        if entity_kind:
            sql.append(" AND p.entity_kind = ?")
            params.append(entity_kind)
        if access_mode:
            sql.append(" AND EXISTS (SELECT 1 FROM json_each(p.access_modes) WHERE value = ?)")
            params.append(access_mode)
        if query:
            rows = self.conn.execute("".join(sql), params).fetchall()
            eligible_ids = {int(row["place_id"]) for row in rows}
            scores = {}
            lanes_by_id = {}
            for lane, place_ids in lane_ranks.items():
                filtered_ids = [place_id for place_id in place_ids if place_id in eligible_ids]
                for rank, place_id in enumerate(filtered_ids, start=1):
                    scores[place_id] = scores.get(place_id, 0.0) + 1.0 / (RRF_K + rank)
                    lanes_by_id.setdefault(place_id, []).append(lane)
            rows = sorted(rows, key=lambda row: str(row["canonical"]))
            rows.sort(key=lambda row: str(row["last_seen_at"] or ""), reverse=True)
            rows.sort(key=lambda row: int(row["mention_count"]), reverse=True)
            rows.sort(key=lambda row: scores[int(row["place_id"])], reverse=True)
            rows = rows[:limit]
        else:
            sql.append(" ORDER BY p.mention_count DESC, p.last_seen_at DESC LIMIT ?")
            params.append(limit)
            rows = self.conn.execute("".join(sql), params).fetchall()

        if name_query and not query:
            lanes_by_id = {int(row["place_id"]): ["lexical"] for row in rows}
        return self._render_places(rows, include_contacts=include_contacts, lanes=lanes_by_id)

    def _semantic_place_ids(
        self,
        query_vector: Sequence[float],
        *,
        model: str,
        cutoff: float,
        descriptor_limit: int,
    ) -> list[int]:
        """Rank entities by their best descriptor, enforcing an honest-empty cutoff."""
        query = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if not norm:
            return []
        ranked_descriptors: list[tuple[int, float]] = []
        for row in self.conn.execute(
            """
            SELECT descriptor_id, dim, vec FROM descriptor_embeddings
             WHERE model=? AND status='ready' AND dim=?
            """,
            (model, int(query.shape[0])),
        ):
            vector = np.frombuffer(row["vec"], dtype=np.float32)
            vector_norm = float(np.linalg.norm(vector))
            if not vector_norm:
                continue
            score = float(np.dot(vector, query) / (vector_norm * norm))
            if score >= cutoff:
                ranked_descriptors.append((int(row["descriptor_id"]), score))
        ranked_descriptors.sort(key=lambda item: (-item[1], item[0]))
        descriptor_rank = {
            descriptor_id: rank
            for rank, (descriptor_id, _score) in enumerate(
                ranked_descriptors[:descriptor_limit], start=1
            )
        }
        if not descriptor_rank:
            return []
        marks = ",".join("?" for _ in descriptor_rank)
        rows = self.conn.execute(
            f"SELECT place_id, descriptor_id FROM place_descriptors "  # noqa: S608
            f"WHERE descriptor_id IN ({marks})",
            tuple(descriptor_rank),
        ).fetchall()
        best: dict[int, int] = {}
        for row in rows:
            place_id = int(row["place_id"])
            rank = descriptor_rank[int(row["descriptor_id"])]
            best[place_id] = min(rank, best.get(place_id, rank))
        return [
            place_id
            for place_id, _rank in sorted(best.items(), key=lambda item: (item[1], item[0]))
        ]

    def store_descriptor_embeddings(
        self,
        items: Iterable[tuple[int, Sequence[float]]],
        *,
        model: str,
    ) -> int:
        """Store precomputed descriptor vectors; this method never calls a provider."""
        payload: list[tuple[int, str, int, bytes]] = []
        for descriptor_id, vector in items:
            array = np.asarray(vector, dtype=np.float32)
            payload.append((descriptor_id, model, int(array.shape[0]), array.tobytes()))
        if not payload:
            return 0
        self.conn.executemany(
            """
            INSERT INTO descriptor_embeddings (descriptor_id, model, status, dim, vec, embedded_at)
            VALUES (?, ?, 'ready', ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(descriptor_id, model) DO UPDATE SET
                status='ready', dim=excluded.dim, vec=excluded.vec,
                error=NULL, embedded_at=CURRENT_TIMESTAMP
            """,
            payload,
        )
        self.conn.commit()
        return len(payload)

    def _render_places(
        self,
        rows: Sequence[sqlite3.Row],
        *,
        include_contacts: bool,
        lanes: dict[int, list[str]],
    ) -> list[dict[str, Any]]:
        """Render old response fields plus additive entity fields."""
        place_ids = [row["place_id"] for row in rows]
        contacts = self.contacts_for_places(place_ids) if include_contacts else {}
        authors = self.authors_for_places(place_ids) if include_contacts else {}

        out = []
        for row in rows:
            evidence = self.conn.execute(
                """
                SELECT pm.evidence_quote, pm.event_types, pm.offerings_raw, pm.confidence,
                       m.date, m.chat_id, m.telegram_msg_id, c.title AS chat_title
                  FROM place_mentions pm
                  JOIN corpus_messages m ON m.corpus_id = pm.corpus_id
                  LEFT JOIN corpus_chats c ON c.chat_id = m.chat_id
                 WHERE pm.place_id = ? ORDER BY m.date DESC LIMIT 4
                """,
                (row["place_id"],),
            ).fetchall()
            out.append(
                {
                    "entity_key": f"{row['entity_kind']}|{row['canonical']}",
                    "name": row["name"],
                    "aliases": json.loads(row["aliases"] or "[]"),
                    "city_area": row["city_area"],
                    "place_type": row["place_type"],
                    "entity_kind": row["entity_kind"],
                    "access_modes": json.loads(row["access_modes"] or "[]"),
                    "descriptor": row["primary_descriptor"],
                    "offerings": sorted(
                        {
                            str(value)
                            for evidence_row in evidence
                            for value in json.loads(evidence_row["offerings_raw"] or "[]")
                        }
                    ),
                    "matched_via": lanes.get(int(row["place_id"]), []),
                    "mentions": row["mention_count"],
                    "first_seen": row["first_seen_at"],
                    "last_seen": row["last_seen_at"],
                    "contacts": contacts.get(row["place_id"], []),
                    "posted_by": authors.get(row["place_id"], []),
                    "evidence": [
                        {
                            "quote": e["evidence_quote"],
                            "event_types": json.loads(e["event_types"] or "[]"),
                            "date": e["date"],
                            "chat": e["chat_title"],
                            "message_link": message_link(e["chat_id"], e["telegram_msg_id"]),
                        }
                        for e in evidence
                    ],
                }
            )
        return out

    # ------------------------------------------------------------------
    # Chat references mined from message text
    # ------------------------------------------------------------------

    def refresh_chat_references(self, *, known: dict[str, str] | None = None) -> int:
        """Mine t.me links and @handles out of the corpus into a candidate list.

        These cost nothing: the messages are already downloaded, and collecting
        a reference spends no Telegram action budget, unlike every discovery
        primitive the account could call instead.
        """
        known = {k.lower(): v for k, v in (known or {}).items()}
        seen: dict[str, dict[str, Any]] = {}
        for row in self.conn.execute(
            "SELECT corpus_id, text, date FROM corpus_messages WHERE text LIKE '%t.me/%' OR text LIKE '%@%'"
        ):
            for ref, kind in _extract_refs(row["text"]):
                entry = seen.setdefault(
                    ref,
                    {
                        "kind": kind,
                        "count": 0,
                        "first": row["date"],
                        "last": row["date"],
                        "sample": row["corpus_id"],
                    },
                )
                entry["count"] += 1
                entry["first"] = min(entry["first"], row["date"])
                entry["last"] = max(entry["last"], row["date"])
        if not seen:
            return 0
        self.conn.executemany(
            """
            INSERT INTO chat_references (ref, kind, mention_count, first_seen_at, last_seen_at,
                                         known_state, sample_corpus_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ref) DO UPDATE SET
                mention_count = excluded.mention_count,
                last_seen_at = excluded.last_seen_at,
                known_state = excluded.known_state
            """,
            [
                (
                    ref,
                    e["kind"],
                    e["count"],
                    e["first"],
                    e["last"],
                    known.get(ref.lower()),
                    e["sample"],
                )
                for ref, e in seen.items()
            ],
        )
        self.conn.commit()
        return len(seen)

    def chat_references(
        self,
        *,
        only_unknown: bool = True,
        min_mentions: int = 1,
        limit: int = 50,
        include_handles: bool = False,
    ) -> list[dict[str, Any]]:
        sql = ["SELECT * FROM chat_references WHERE mention_count >= ?"]
        params: list[Any] = [min_mentions]
        if not include_handles:
            # A bare @handle is far more often a person, a shop or an ad account
            # than a chat worth joining; an explicit t.me link is a deliberate
            # pointer at a group or channel.
            sql.append(" AND kind IN ('username', 'invite')")
        if only_unknown:
            sql.append(" AND known_state IS NULL")
        sql.append(" ORDER BY mention_count DESC, last_seen_at DESC LIMIT ?")
        params.append(limit)
        return [
            {
                "ref": r["ref"],
                "kind": r["kind"],
                "mentions": r["mention_count"],
                "first_seen": r["first_seen_at"],
                "last_seen": r["last_seen_at"],
                "already": r["known_state"],
            }
            for r in self.conn.execute("".join(sql), params).fetchall()
        ]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def chats(self) -> list[dict[str, Any]]:
        return [
            {
                "chat_id": r["chat_id"],
                "title": r["title"],
                "mode": r["mode"],
                "messages": r["message_count"],
                "first_message": r["first_message_at"],
                "last_message": r["last_message_at"],
            }
            for r in self.conn.execute(
                "SELECT * FROM corpus_chats ORDER BY message_count DESC"
            ).fetchall()
        ]

    def parent_text_for_reply(self, message: CorpusRow) -> str | None:
        """Return a reply's parent text, or None for non-replies and corpus gaps."""
        if message.reply_to_message_id is None:
            return None
        row = self.conn.execute(
            """
            SELECT text FROM corpus_messages
             WHERE chat_id = ? AND telegram_msg_id = ?
            """,
            (message.chat_id, message.reply_to_message_id),
        ).fetchone()
        return str(row["text"]) if row is not None else None

    def reply_linkage_stats(self, *, chat_id: int | None = None) -> dict[str, object]:
        """Measure reply coverage with explicit populations and question heuristic."""
        row = self.conn.execute(
            """
            SELECT
                count(*) AS rows_total,
                count(answer.reply_to_message_id) AS reply_rows,
                count(parent.corpus_id) AS replies_with_parent,
                count(CASE
                    WHEN parent.corpus_id IS NOT NULL
                     AND (instr(parent.text, '?') > 0 OR instr(parent.text, '？') > 0)
                    THEN 1
                END) AS replies_whose_parent_is_question
              FROM corpus_messages AS answer
              LEFT JOIN corpus_messages AS parent
                ON parent.chat_id = answer.chat_id
               AND parent.telegram_msg_id = answer.reply_to_message_id
             WHERE (? IS NULL OR answer.chat_id = ?)
            """,
            (chat_id, chat_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("reply linkage measurement returned no row")
        return {
            "scope": "whole_corpus" if chat_id is None else "chat",
            "chat_id": chat_id,
            "rows_total": int(row["rows_total"]),
            "reply_rows": {
                "count": int(row["reply_rows"]),
                "population": "rows_total",
            },
            "replies_with_parent": {
                "count": int(row["replies_with_parent"]),
                "population": "reply_rows",
            },
            "replies_whose_parent_is_question": {
                "count": int(row["replies_whose_parent_is_question"]),
                "population": "replies_with_parent",
                "question_rule": "parent text contains ? or full-width ？",
            },
        }

    def recent(
        self,
        *,
        chat_id: int | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 50,
    ) -> list[CorpusRow]:
        sql = [
            """
            SELECT m.corpus_id, m.chat_id, m.telegram_msg_id,
                   COALESCE(m.chat_title, c.title) AS chat_title, m.sender_name,
                   m.text, m.date, m.content_hash, m.source, m.reply_to_message_id
              FROM corpus_messages m
              LEFT JOIN corpus_chats c ON c.chat_id = m.chat_id
             WHERE 1=1
            """
        ]
        params: list[Any] = []
        sql, params = _apply_filters(sql, params, [chat_id] if chat_id else None, since, until)
        sql.append(" ORDER BY m.date DESC LIMIT ?")
        params.append(limit)
        return [_row_to_corpus(r, lane="recent") for r in self.conn.execute("".join(sql), params)]

    def status(self) -> dict[str, Any]:
        c = self.conn

        def one(sql: str, params: Sequence[object] = ()) -> Any:
            return c.execute(sql, params).fetchone()[0]

        return {
            "messages_indexed": one("SELECT count(*) FROM corpus_messages"),
            "chats": one("SELECT count(*) FROM corpus_chats"),
            "oldest_message": one("SELECT min(date) FROM corpus_messages"),
            "newest_message": one("SELECT max(date) FROM corpus_messages"),
            "embeddings": one("SELECT count(*) FROM corpus_embeddings"),
            "embedding_backlog": one(
                "SELECT count(*) FROM corpus_messages m LEFT JOIN corpus_embeddings e"
                " ON e.corpus_id = m.corpus_id WHERE e.corpus_id IS NULL"
            ),
            "places": one("SELECT count(*) FROM places"),
            "place_mentions": one("SELECT count(*) FROM place_mentions"),
            "extraction_backlog": one(
                "SELECT count(*) FROM extraction_jobs WHERE prompt_version = ? "
                "AND status IN ('pending', 'running')",
                (ENTITY_EXTRACTION_VERSION,),
            ),
            "extraction_errors": one(
                "SELECT count(*) FROM extraction_jobs WHERE prompt_version = ? "
                "AND status = 'error'",
                (ENTITY_EXTRACTION_VERSION,),
            ),
            "active_prompt_version": one(
                "SELECT COALESCE((SELECT active_prompt_version FROM extraction_state "
                "GROUP BY active_prompt_version ORDER BY count(*) DESC LIMIT 1), 'places-v2')"
            ),
            "entities_v3_active": one(
                "SELECT count(*) FROM extraction_state WHERE active_prompt_version = 'entities-v3'"
            ),
            "entities_current_active": one(
                "SELECT count(*) FROM extraction_state WHERE active_prompt_version = ?",
                (ENTITY_EXTRACTION_VERSION,),
            ),
            "descriptors": one("SELECT count(*) FROM descriptors"),
            "descriptor_embedding_backlog": one(
                "SELECT count(*) FROM descriptor_embeddings WHERE status <> 'ready'"
            ),
            "place_fts_rows": one("SELECT count(*) FROM place_fts"),
            "place_fts_next_rows": one("SELECT count(*) FROM place_fts_next"),
            "chat_references": one("SELECT count(*) FROM chat_references"),
            "sync": [
                {"source": r["source"], "cursor": r["last_id"], "last_run": r["last_run_at"]}
                for r in c.execute("SELECT * FROM sync_state").fetchall()
            ],
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row_to_corpus(row: sqlite3.Row, *, lane: str) -> CorpusRow:
    return CorpusRow(
        corpus_id=row["corpus_id"],
        chat_id=row["chat_id"],
        telegram_msg_id=row["telegram_msg_id"],
        chat_title=row["chat_title"],
        sender_name=row["sender_name"],
        text=row["text"],
        date=row["date"],
        content_hash=row["content_hash"],
        source=row["source"],
        reply_to_message_id=row["reply_to_message_id"],
        lanes=[lane] if lane in {"lexical", "semantic"} else [],
    )


def _apply_filters(
    sql: list[str],
    params: list[Any],
    chat_ids: Sequence[int] | None,
    since: str | None,
    until: str | None,
) -> tuple[list[str], list[Any]]:
    """Append the shared chat and date filters.

    Every caller aliases ``corpus_messages`` as ``m``, so the column prefix is a
    constant here rather than a value spliced into the SQL text.
    """
    if chat_ids:
        sql.append(" AND m.chat_id IN (SELECT value FROM json_each(?))")
        params.append(json.dumps(list(chat_ids)))
    if since:
        sql.append(" AND m.date >= ?")
        params.append(since)
    if until:
        sql.append(" AND m.date <= ?")
        params.append(until)
    return sql, params


def _rollup_duplicates(ranked: Sequence[CorpusRow], limit: int) -> list[CorpusRow]:
    """Collapse identical crossposts into one result carrying its other chats.

    A quarter of this corpus is the same announcement posted to several chats.
    Without this, one popular event fills half an answer with copies of itself,
    and the fact that it was posted in five chats -- a real relevance signal --
    is lost in the noise instead of being reported.
    """
    out: list[CorpusRow] = []
    by_hash: dict[str, CorpusRow] = {}
    for item in ranked:
        head = by_hash.get(item.content_hash)
        if head is None:
            by_hash[item.content_hash] = item
            out.append(item)
            if len(out) >= limit:
                break
        elif item.chat_id != head.chat_id and item.chat_id not in head.duplicate_chat_ids:
            head.duplicate_chat_ids.append(item.chat_id)
    return out


def extract_contacts(text: str) -> list[tuple[str, str, str]]:
    """Find every reachable handle in one message as ``(kind, value, display)``.

    ``value`` is the comparable form -- lowercased handle, digits-only phone --
    so the same organiser written as ``@aum_danang`` and ``t.me/AUM_danang``
    collapses to one contact instead of ranking as two.
    """
    found: dict[tuple[str, str], str] = {}

    def add(kind: str, value: str, display: str) -> None:
        found.setdefault((kind, value), display)

    for match in _TME_RE.finditer(text):
        body = match.group("body")
        if body.lower().startswith("joinchat/") or body.startswith("+"):
            add("telegram", body.lower(), f"t.me/{body}")
        elif body.lower() not in _REF_STOPWORDS:
            add("telegram", body.lower(), f"@{body}")
    for match in _AT_HANDLE_RE.finditer(text):
        name = match.group("name")
        if name.lower() not in _REF_STOPWORDS:
            add("telegram", name.lower(), f"@{name}")

    for kind, pattern in _CONTACT_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group("v") if "v" in pattern.groupindex else match.group(0)
            add(kind, raw.lower().rstrip("/"), match.group(0))

    for pattern, local in ((_PHONE_INTL_RE, False), (_PHONE_VN_RE, True)):
        for match in pattern.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))
            # A local Vietnamese number is the same phone as its +84 form, so
            # both have to normalize to one value or the venue's owner shows up
            # twice with two spellings of the same number.
            if local:
                digits = "84" + digits[1:]
            if not 9 <= len(digits) <= 15:
                continue
            add("phone", "+" + digits, match.group(0).strip())
    return [(kind, value, display) for (kind, value), display in found.items()]


def _extract_refs(text: str) -> Iterator[tuple[str, str]]:
    for match in _TME_RE.finditer(text):
        body = match.group("body")
        if body.lower().startswith(("+", "joinchat/")):
            yield body, "invite"
        elif body.lower() not in _REF_STOPWORDS:
            yield body.lower(), "username"
    for match in _AT_HANDLE_RE.finditer(text):
        name = match.group("name").lower()
        if name not in _REF_STOPWORDS:
            yield name, "unknown"
