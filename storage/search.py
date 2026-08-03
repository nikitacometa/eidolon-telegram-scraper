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


def escape_fts_term(term: str) -> str:
    """Quote one bare term for an FTS5 MATCH expression.

    User and model input reaches MATCH directly. Unquoted, a stray hyphen or
    quote is not a bad search, it is a syntax error that fails the whole call.
    """
    return '"' + term.replace('"', '""') + '"'


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
    score: float = 0.0
    lanes: list[str] = field(default_factory=list)
    duplicate_chat_ids: list[int] = field(default_factory=list)

    def as_dict(self, *, text_limit: int | None = None) -> dict[str, Any]:
        body = self.text
        if text_limit is not None and len(body) > text_limit:
            body = body[:text_limit].rstrip() + "…"
        return {
            "corpus_id": self.corpus_id,
            "chat_id": self.chat_id,
            "chat_title": self.chat_title,
            "message_link": message_link(self.chat_id, self.telegram_msg_id),
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
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SEARCH_SCHEMA_PATH.read_text())
            self._conn.commit()
        self._conn.execute("PRAGMA busy_timeout=10000")
        logger.info("Search database connected: %s (read_only=%s)", self.db_path, read_only)

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
            stats = {
                "live": self._sync_stream(
                    stream="live",
                    select_sql="""
                        SELECT id AS cursor, chat_id, telegram_msg_id, chat_title,
                               sender_id, sender_name, text, date, NULL AS content_hash
                          FROM live.messages
                         WHERE id > ? AND text IS NOT NULL AND trim(text) <> ''
                         ORDER BY id LIMIT ?
                    """,
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
                    select_sql="""
                        SELECT rowid AS cursor, chat_id, telegram_msg_id, NULL AS chat_title,
                               sender_id, sender_name, text, date, content_hash
                          FROM scout.scout_messages
                         WHERE rowid > ? AND text IS NOT NULL AND trim(text) <> ''
                         ORDER BY rowid LIMIT ?
                    """,
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
            self._refresh_chat_metadata()
            self._seed_extraction_state()
            conn.commit()
        finally:
            conn.execute("DETACH DATABASE live")
            conn.execute("DETACH DATABASE scout")
        self._vector_cache = None
        return stats

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
                        digest,
                    )
                )
            if payload:
                conn.executemany(
                    """
                    INSERT INTO corpus_messages (
                        source, chat_id, telegram_msg_id, chat_title,
                        sender_id, sender_name, text, date, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id, telegram_msg_id) DO NOTHING
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
        """Queue every substantive new message for entity extraction.

        The gate is length, not vocabulary. A keyword pre-filter was measured
        against this corpus and missed most real venue posts -- the announcement
        for a rave at a named hotel is written in English party slang that no
        Russian keyword list anticipates. Length is the only cheap filter that
        does not encode a guess about wording.
        """
        self.conn.execute(
            """
            INSERT INTO extraction_state (corpus_id, status)
            SELECT corpus_id, CASE WHEN length(text) >= 80 THEN 'pending' ELSE 'skipped' END
              FROM corpus_messages
             WHERE corpus_id NOT IN (SELECT corpus_id FROM extraction_state)
            """
        )

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
                   m.text, m.date, m.content_hash, m.source, bm25(corpus_fts) AS rank
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
                   m.text, m.date, m.content_hash, m.source
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

        return _rollup_duplicates(sorted(fused.values(), key=lambda x: -x.score), limit)

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

    def pending_extractions(self, limit: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT m.corpus_id, m.text, m.date, m.chat_id, c.title AS chat_title
              FROM extraction_state s
              JOIN corpus_messages m ON m.corpus_id = s.corpus_id
              LEFT JOIN corpus_chats c ON c.chat_id = m.chat_id
             WHERE s.status = 'pending'
             ORDER BY m.date DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def record_extraction(
        self,
        corpus_id: int,
        places: Sequence[dict[str, Any]],
        *,
        model: str,
        error: str | None = None,
    ) -> int:
        """Persist one message's extracted venues and close out its state row."""
        conn = self.conn
        if error is not None:
            conn.execute(
                "UPDATE extraction_state SET status='error', attempted_at=?, error=? WHERE corpus_id=?",
                (_now(), error[:500], corpus_id),
            )
            conn.commit()
            return 0

        stored = 0
        for entry in places:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            if not is_venue_name(name):
                logger.debug("Rejected extracted place name %r from %s", name, corpus_id)
                continue
            canonical = fold_ascii(name)
            if not canonical or len(canonical) < 2:
                continue
            aliases = entry.get("aliases") or []
            if name.lower() != canonical and canonical not in aliases:
                aliases = [*aliases, canonical]
            row = conn.execute(
                "SELECT place_id, aliases FROM places WHERE canonical = ?", (canonical,)
            ).fetchone()
            if row is None:
                cur = conn.execute(
                    """
                    INSERT INTO places (canonical, name, aliases, city_area, place_type,
                                        first_seen_at, last_seen_at)
                    VALUES (?, ?, ?, ?, ?,
                            (SELECT date FROM corpus_messages WHERE corpus_id = ?),
                            (SELECT date FROM corpus_messages WHERE corpus_id = ?))
                    """,
                    (
                        canonical,
                        name,
                        json.dumps(sorted(set(aliases)), ensure_ascii=False),
                        entry.get("city_area"),
                        entry.get("place_type"),
                        corpus_id,
                        corpus_id,
                    ),
                )
                place_id = int(cur.lastrowid or 0)
            else:
                place_id = int(row["place_id"])
                merged = sorted(set(json.loads(row["aliases"] or "[]")) | set(aliases) | {name})
                conn.execute(
                    """
                    UPDATE places SET
                        aliases = ?,
                        city_area = COALESCE(city_area, ?),
                        place_type = COALESCE(place_type, ?),
                        last_seen_at = MAX(COALESCE(last_seen_at, ''),
                                           (SELECT date FROM corpus_messages WHERE corpus_id = ?))
                     WHERE place_id = ?
                    """,
                    (
                        json.dumps(merged, ensure_ascii=False),
                        entry.get("city_area"),
                        entry.get("place_type"),
                        corpus_id,
                        place_id,
                    ),
                )
            conn.execute(
                """
                INSERT INTO place_mentions (place_id, corpus_id, event_types, evidence_quote,
                                            confidence, extracted_by)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(place_id, corpus_id) DO UPDATE SET
                    event_types = excluded.event_types,
                    evidence_quote = excluded.evidence_quote,
                    confidence = excluded.confidence,
                    extracted_by = excluded.extracted_by
                """,
                (
                    place_id,
                    corpus_id,
                    json.dumps(entry.get("event_types") or [], ensure_ascii=False),
                    str(entry.get("evidence") or "")[:600],
                    entry.get("confidence"),
                    model,
                ),
            )
            stored += 1

        conn.execute(
            "UPDATE extraction_state SET status=?, attempted_at=?, error=NULL WHERE corpus_id=?",
            ("extracted" if stored else "no_venue", _now(), corpus_id),
        )
        conn.execute(
            """
            UPDATE places SET mention_count =
                (SELECT count(*) FROM place_mentions pm WHERE pm.place_id = places.place_id)
             WHERE place_id IN (SELECT place_id FROM place_mentions WHERE corpus_id = ?)
            """,
            (corpus_id,),
        )
        conn.commit()
        return stored

    def search_places(
        self,
        *,
        name_query: str | None = None,
        city_area: str | None = None,
        place_type: str | None = None,
        event_types: Sequence[str] | None = None,
        min_mentions: int = 1,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Query the extracted venue index."""
        sql = [
            """
            SELECT p.place_id, p.name, p.aliases, p.city_area, p.place_type,
                   p.mention_count, p.first_seen_at, p.last_seen_at
              FROM places p
             WHERE p.mention_count >= ?
            """
        ]
        params: list[Any] = [min_mentions]
        if name_query:
            sql.append(" AND p.place_id IN (SELECT rowid FROM place_fts WHERE place_fts MATCH ?)")
            params.append(escape_fts_term(fold_ascii(name_query)))
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
        sql.append(" ORDER BY p.mention_count DESC, p.last_seen_at DESC LIMIT ?")
        params.append(limit)

        out = []
        for row in self.conn.execute("".join(sql), params).fetchall():
            evidence = self.conn.execute(
                """
                SELECT pm.evidence_quote, pm.event_types, pm.confidence, m.date, m.chat_id,
                       m.telegram_msg_id, c.title AS chat_title
                  FROM place_mentions pm
                  JOIN corpus_messages m ON m.corpus_id = pm.corpus_id
                  LEFT JOIN corpus_chats c ON c.chat_id = m.chat_id
                 WHERE pm.place_id = ? ORDER BY m.date DESC LIMIT 4
                """,
                (row["place_id"],),
            ).fetchall()
            out.append(
                {
                    "name": row["name"],
                    "aliases": json.loads(row["aliases"] or "[]"),
                    "city_area": row["city_area"],
                    "place_type": row["place_type"],
                    "mentions": row["mention_count"],
                    "first_seen": row["first_seen_at"],
                    "last_seen": row["last_seen_at"],
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
                   m.text, m.date, m.content_hash, m.source
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

        def one(sql: str) -> Any:
            return c.execute(sql).fetchone()[0]

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
                "SELECT count(*) FROM extraction_state WHERE status = 'pending'"
            ),
            "extraction_errors": one(
                "SELECT count(*) FROM extraction_state WHERE status = 'error'"
            ),
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
