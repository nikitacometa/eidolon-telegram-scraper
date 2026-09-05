"""Durable state for reconnaissance jobs, held in its own SQLite file.

The live monitoring :class:`storage.db.Database` serializes every statement
behind a single connection lock. Crawl traffic is orders of magnitude heavier
than monitoring traffic, so it gets a separate file, connection, and lock; a
slow bulk write here can never delay live alert delivery.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

import aiosqlite

from pipeline.recon_models import (
    ACTIVE_JOB_STATUSES,
    DEFAULT_BUDGET_POLICY,
    TERMINAL_JOB_STATUSES,
    ActionKind,
    ActionOutcome,
    ActionReservation,
    BackfillState,
    BackfillTarget,
    BudgetDenial,
    BudgetRule,
    BudgetScope,
    Candidate,
    CandidateState,
    ChatIdentity,
    ChatMembership,
    ChatVisibility,
    Evidence,
    FrontierItem,
    JobRequest,
    JoinQueueState,
    QueuedJoin,
    ReconJob,
    ReconJobStatus,
    ReservationOutcome,
    ScoutChat,
    ScoutMessage,
    normalize_username,
)

logger = logging.getLogger(__name__)

SCOUT_SCHEMA_PATH = Path(__file__).parent / "scout_schema.sql"


def invite_fingerprint(invite_hash: str, *, secret: bytes) -> str:
    """Return a stable, non-reversible handle for an invite hash.

    A raw invite hash is a working key to a chat. Reconnaissance needs to
    recognise a link it has seen before, which a keyed digest provides without
    persisting the credential itself.
    """
    normalized = invite_hash.strip().strip("/")
    for prefix in ("https://t.me/", "http://t.me/", "t.me/", "joinchat/", "+"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    digest = hmac.new(secret, normalized.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:32]


class ScoutDatabase:
    """Async SQLite wrapper for reconnaissance state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        # Mirrors storage.db.Database: one aiosqlite connection cannot host
        # overlapping transactions, so writes are serialized in-process. This
        # lock is deliberately a different object from the live database's.
        self._write_lock = asyncio.Lock()

    @property
    def conn(self) -> aiosqlite.Connection:
        """Return the open connection or fail loudly."""
        if self._conn is None:
            raise RuntimeError("Scout database is not connected")
        return self._conn

    async def connect(self) -> None:
        """Open the connection and apply the schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.db_path.parent, 0o700)
        self._conn = await aiosqlite.connect(self.db_path)
        os.chmod(self.db_path, 0o600)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        stranded = await (
            await self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='telegram_actions_legacy'"
            )
        ).fetchone()
        await self._conn.executescript(SCOUT_SCHEMA_PATH.read_text())
        if stranded is not None:
            # A previous connect died between renaming the ledger aside and
            # copying it back. The schema above has just created an empty
            # ledger next to the full one; without this the account's whole
            # rate-limit history would sit in a table nothing reads.
            await self._conn.execute(
                "INSERT OR IGNORE INTO telegram_actions SELECT * FROM telegram_actions_legacy"
            )
            await self._conn.execute("DROP TABLE telegram_actions_legacy")
            await self._conn.executescript(SCOUT_SCHEMA_PATH.read_text())
            logger.info("Migration: finished an interrupted telegram_actions rebuild")
        columns = {
            row[1]
            for row in await (
                await self._conn.execute("PRAGMA table_info(scout_messages)")
            ).fetchall()
        }
        if "reply_to_message_id" not in columns:
            await self._conn.execute(
                "ALTER TABLE scout_messages ADD COLUMN reply_to_message_id INTEGER"
            )
            logger.info("Migration: added scout_messages.reply_to_message_id")
        ledger_sql = await (
            await self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='telegram_actions'"
            )
        ).fetchone()
        missing_kinds = [
            kind.value
            for kind in ActionKind
            if ledger_sql is not None and f"'{kind.value}'" not in str(ledger_sql[0])
        ]
        if missing_kinds:
            # The kind CHECK predates some action kinds, and a CHECK cannot be
            # altered in place: every reservation of a missing kind raises
            # IntegrityError and the worker behind it never runs. This has
            # happened once already (the media download kinds), so the test is
            # now "every kind the code knows is in the CHECK", not one name.
            # Rebuild the table; nothing in the schema references it by FK.
            await self._conn.execute(
                "ALTER TABLE telegram_actions RENAME TO telegram_actions_legacy"
            )
            await self._conn.executescript(SCOUT_SCHEMA_PATH.read_text())
            await self._conn.execute(
                "INSERT INTO telegram_actions SELECT * FROM telegram_actions_legacy"
            )
            await self._conn.execute("DROP TABLE telegram_actions_legacy")
            # The rename dragged idx_actions_budget along to the legacy table,
            # where the DROP destroyed it; IF NOT EXISTS skipped the name while
            # it was still taken. Re-running the schema now recreates it.
            await self._conn.executescript(SCOUT_SCHEMA_PATH.read_text())
            logger.info(
                "Migration: rebuilt telegram_actions with kinds %s", ", ".join(missing_kinds)
            )
        await self._conn.commit()
        logger.info("Scout database connected: %s", self.db_path)

    async def close(self) -> None:
        """Close the connection if it is open."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    async def create_job(self, request: JobRequest) -> ReconJob:
        """Create a job, or return the existing one for a repeated key.

        The caller is an ephemeral agent that cannot tell a lost response from
        a lost request, so a repeated submit must never start a second crawl.
        """
        async with self._write_lock:
            existing = await self._job_by_key(request.idempotency_key)
            if existing is not None:
                return existing

            job_id = uuid.uuid4().hex
            await self.conn.execute(
                """
                INSERT INTO recon_jobs (
                    id, idempotency_key, account_id, topic, location, languages, seeds,
                    status, max_waves, lookback_days, max_candidates, max_join_attempts,
                    deadline_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', ?))
                """,
                (
                    job_id,
                    request.idempotency_key,
                    request.account_id,
                    request.topic,
                    request.location,
                    json.dumps(list(request.languages), separators=(",", ":")),
                    json.dumps(list(request.seeds), separators=(",", ":")),
                    ReconJobStatus.QUEUED.value,
                    request.max_waves,
                    request.lookback_days,
                    request.max_candidates,
                    request.max_join_attempts,
                    f"+{request.deadline_hours} hours",
                ),
            )
            await self.conn.commit()

        created = await self.get_job(job_id)
        if created is None:
            raise RuntimeError("job insert did not produce a readable row")
        return created

    async def get_job(self, job_id: str) -> ReconJob | None:
        """Return one job by id."""
        async with self.conn.execute(
            "SELECT * FROM recon_jobs WHERE id = ?",
            (job_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return _job_from_row(row) if row is not None else None

    async def _job_by_key(self, idempotency_key: str) -> ReconJob | None:
        async with self.conn.execute(
            "SELECT * FROM recon_jobs WHERE idempotency_key = ?",
            (idempotency_key,),
        ) as cursor:
            row = await cursor.fetchone()
        return _job_from_row(row) if row is not None else None

    async def next_queued_job(self) -> ReconJob | None:
        """The oldest job nobody has started, for the in-daemon discovery worker."""
        async with self.conn.execute(
            # created_at has second resolution; two jobs queued in one second keep
            # their insertion order through rowid rather than a random uuid.
            "SELECT * FROM recon_jobs WHERE status = ? ORDER BY created_at, rowid LIMIT 1",
            (ReconJobStatus.QUEUED.value,),
        ) as cursor:
            row = await cursor.fetchone()
        return _job_from_row(row) if row is not None else None

    async def active_jobs(self) -> list[ReconJob]:
        """Return jobs that a restarted scheduler must pick back up."""
        placeholders = ",".join("?" for _ in ACTIVE_JOB_STATUSES)
        statuses = [status.value for status in sorted(ACTIVE_JOB_STATUSES)]
        async with self.conn.execute(
            # Only "?" placeholders are interpolated; every value stays bound.
            f"SELECT * FROM recon_jobs WHERE status IN ({placeholders}) ORDER BY created_at",  # noqa: S608
            statuses,
        ) as cursor:
            rows = await cursor.fetchall()
        return [_job_from_row(row) for row in rows]

    async def update_job_status(
        self,
        job_id: str,
        status: ReconJobStatus,
        *,
        stop_reason: str | None = None,
        error_code: str | None = None,
        resume_after_seconds: int | None = None,
        current_wave: int | None = None,
    ) -> bool:
        """Move a job to a new status unless it already ended.

        Terminal states are immutable: an agent polling a finished job must
        never observe it regress into running.
        """
        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                async with self.conn.execute(
                    "SELECT status FROM recon_jobs WHERE id = ?",
                    (job_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    await self.conn.rollback()
                    return False
                if ReconJobStatus(str(row["status"])) in TERMINAL_JOB_STATUSES:
                    await self.conn.rollback()
                    return False

                completed = status in TERMINAL_JOB_STATUSES
                resume_modifier = (
                    f"+{resume_after_seconds} seconds" if resume_after_seconds is not None else None
                )
                await self.conn.execute(
                    """
                    UPDATE recon_jobs
                    SET status = ?,
                        stop_reason = COALESCE(?, stop_reason),
                        error_code = COALESCE(?, error_code),
                        current_wave = COALESCE(?, current_wave),
                        resume_at = CASE
                            WHEN ? IS NULL THEN resume_at
                            ELSE datetime('now', ?)
                        END,
                        updated_at = CURRENT_TIMESTAMP,
                        completed_at = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE completed_at END
                    WHERE id = ?
                    """,
                    (
                        status.value,
                        stop_reason,
                        error_code,
                        current_wave,
                        resume_modifier,
                        resume_modifier,
                        1 if completed else 0,
                        job_id,
                    ),
                )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise
        return True

    # ------------------------------------------------------------------
    # Chat identity
    # ------------------------------------------------------------------

    async def resolve_chat(self, identity: ChatIdentity) -> str:
        """Return the canonical chat uuid for an identity, creating or merging.

        Discovery finds the same chat under a username, a peer id, and an
        invite link at different times. Whichever alias arrives first creates
        the record; a later alias that proves two records are the same chat
        merges them instead of leaving a duplicate node in the graph.
        """
        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                known: list[str] = []
                for kind, value in identity.aliases:
                    async with self.conn.execute(
                        "SELECT chat_uuid FROM chat_aliases WHERE alias_kind = ? AND alias_value = ?",
                        (kind.value, value),
                    ) as cursor:
                        row = await cursor.fetchone()
                    if row is not None:
                        chat_uuid = str(row["chat_uuid"])
                        if chat_uuid not in known:
                            known.append(chat_uuid)

                if not known:
                    survivor = uuid.uuid4().hex
                    await self.conn.execute(
                        """
                        INSERT INTO scout_chats (
                            id, telegram_chat_id, username, title, chat_type
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            survivor,
                            identity.peer_id,
                            identity.username,
                            identity.title,
                            identity.chat_type,
                        ),
                    )
                else:
                    survivor = await self._pick_survivor(known)
                    for loser in known:
                        if loser != survivor:
                            await self._merge_chat(loser, survivor)
                    await self.conn.execute(
                        """
                        UPDATE scout_chats
                        SET telegram_chat_id = COALESCE(?, telegram_chat_id),
                            username = COALESCE(?, username),
                            title = COALESCE(?, title),
                            chat_type = COALESCE(?, chat_type),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (
                            identity.peer_id,
                            identity.username,
                            identity.title,
                            identity.chat_type,
                            survivor,
                        ),
                    )

                for kind, value in identity.aliases:
                    await self.conn.execute(
                        """
                        INSERT INTO chat_aliases (alias_kind, alias_value, chat_uuid)
                        VALUES (?, ?, ?)
                        ON CONFLICT(alias_kind, alias_value)
                        DO UPDATE SET last_seen_at = CURRENT_TIMESTAMP
                        """,
                        (kind.value, value, survivor),
                    )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise
        return survivor

    async def _pick_survivor(self, candidates: Sequence[str]) -> str:
        """Prefer the record that already carries a resolved Telegram peer."""
        placeholders = ",".join("?" for _ in candidates)
        async with self.conn.execute(
            # Only "?" placeholders are interpolated; every value stays bound.
            f"""
            SELECT id FROM scout_chats
            WHERE id IN ({placeholders})
            ORDER BY (telegram_chat_id IS NULL), created_at, id
            LIMIT 1
            """,  # noqa: S608
            list(candidates),
        ) as cursor:
            row = await cursor.fetchone()
        return str(row["id"]) if row is not None else candidates[0]

    async def _merge_chat(self, loser: str, survivor: str) -> None:
        """Fold one duplicate chat record into another inside a transaction."""
        await self.conn.execute(
            "UPDATE chat_aliases SET chat_uuid = ? WHERE chat_uuid = ?",
            (survivor, loser),
        )

        # Where both records were candidates of the same job, keep the
        # surviving candidate and move the duplicate's signals onto it.
        # OR IGNORE drops a signal that is already recorded, which is the
        # correct reading of the same origin arriving twice.
        async with self.conn.execute(
            """
            SELECT loser.id AS loser_id, keeper.id AS keeper_id
            FROM job_candidates AS loser
            JOIN job_candidates AS keeper
              ON keeper.job_id = loser.job_id AND keeper.chat_uuid = ?
            WHERE loser.chat_uuid = ?
            """,
            (survivor, loser),
        ) as cursor:
            collisions = [
                (int(row["loser_id"]), int(row["keeper_id"])) for row in await cursor.fetchall()
            ]

        for loser_id, keeper_id in collisions:
            await self.conn.execute(
                "UPDATE OR IGNORE candidate_evidence SET candidate_id = ? WHERE candidate_id = ?",
                (keeper_id, loser_id),
            )
            await self.conn.execute(
                "UPDATE telegram_actions SET candidate_id = ? WHERE candidate_id = ?",
                (keeper_id, loser_id),
            )
            await self.conn.execute("DELETE FROM job_candidates WHERE id = ?", (loser_id,))

        await self.conn.execute(
            "UPDATE job_candidates SET chat_uuid = ? WHERE chat_uuid = ?",
            (survivor, loser),
        )
        await self.conn.execute("DELETE FROM scout_chats WHERE id = ?", (loser,))

        # independent_sources is denormalized, so every candidate that absorbed
        # signals has to be recounted before anything scores it again.
        await self.conn.execute(
            """
            UPDATE job_candidates
            SET independent_sources = (
                    SELECT COUNT(DISTINCT origin_key)
                    FROM candidate_evidence
                    WHERE candidate_evidence.candidate_id = job_candidates.id
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE chat_uuid = ?
            """,
            (survivor,),
        )

    async def get_chat(self, chat_uuid: str) -> ScoutChat | None:
        """Return one canonical chat record."""
        async with self.conn.execute(
            "SELECT * FROM scout_chats WHERE id = ?",
            (chat_uuid,),
        ) as cursor:
            row = await cursor.fetchone()
        return _chat_from_row(row) if row is not None else None

    async def record_chat_profile(
        self,
        chat_uuid: str,
        *,
        participants: int | None = None,
        risk_flags: Iterable[str] = (),
        chat_type: str | None = None,
        title: str | None = None,
    ) -> None:
        """Store the profile fields scoring depends on.

        Scoring runs from the stored record rather than from the search
        response, so anything the scorer reads has to survive the write.
        """
        flags = json.dumps(sorted(set(risk_flags)), separators=(",", ":"))
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE scout_chats
                SET participants = COALESCE(?, participants),
                    risk_flags = CASE WHEN ? = '[]' THEN risk_flags ELSE ? END,
                    chat_type = COALESCE(?, chat_type),
                    title = COALESCE(?, title),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (participants, flags, flags, chat_type, title, chat_uuid),
            )
            await self.conn.commit()

    async def set_chat_visibility(
        self,
        chat_uuid: str,
        visibility: ChatVisibility,
        *,
        source: str,
    ) -> None:
        """Record proof of a chat's public scope, with where the proof came from."""
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE scout_chats
                SET visibility = ?,
                    visibility_source = ?,
                    visibility_checked_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (visibility.value, source, chat_uuid),
            )
            await self.conn.commit()

    async def set_chat_membership(
        self,
        chat_uuid: str,
        membership: ChatMembership,
    ) -> None:
        """Record membership, stamping ``joined_at`` only on confirmed entry."""
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE scout_chats
                SET membership = ?,
                    joined_at = CASE
                        WHEN ? = 'member' AND joined_at IS NULL THEN CURRENT_TIMESTAMP
                        ELSE joined_at
                    END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (membership.value, membership.value, chat_uuid),
            )
            await self.conn.commit()

    # ------------------------------------------------------------------
    # Candidates and evidence
    # ------------------------------------------------------------------

    async def add_candidate(
        self,
        *,
        job_id: str,
        chat_uuid: str,
        wave: int,
        evidence: Evidence,
        priority: float = 0.0,
    ) -> int:
        """Attach a chat to a job and record the signal that surfaced it.

        Returns the candidate id whether it was created now or seen before:
        rediscovery through another route is extra evidence, not a new node.
        """
        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                await self.conn.execute(
                    """
                    INSERT INTO job_candidates (job_id, chat_uuid, wave)
                    VALUES (?, ?, ?)
                    ON CONFLICT(job_id, chat_uuid) DO NOTHING
                    """,
                    (job_id, chat_uuid, wave),
                )
                async with self.conn.execute(
                    "SELECT id FROM job_candidates WHERE job_id = ? AND chat_uuid = ?",
                    (job_id, chat_uuid),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise RuntimeError("candidate upsert did not produce a readable row")
                candidate_id = int(row["id"])

                await self.conn.execute(
                    """
                    INSERT INTO candidate_evidence (
                        candidate_id, source, origin_key, source_chat_id,
                        source_message_id, snippet
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(candidate_id, source, origin_key) DO NOTHING
                    """,
                    (
                        candidate_id,
                        evidence.source.value,
                        evidence.origin_key,
                        evidence.source_chat_id,
                        evidence.source_message_id,
                        evidence.snippet,
                    ),
                )
                await self._recount_evidence(candidate_id)
                await self.conn.execute(
                    """
                    INSERT INTO recon_frontier (candidate_id, job_id, priority)
                    VALUES (?, ?, ?)
                    ON CONFLICT(candidate_id) DO UPDATE
                    SET priority = MAX(recon_frontier.priority, excluded.priority),
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (candidate_id, job_id, priority),
                )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise
        return candidate_id

    async def _recount_evidence(self, candidate_id: int) -> None:
        """Refresh the independent-signal count from distinct origins."""
        await self.conn.execute(
            """
            UPDATE job_candidates
            SET independent_sources = (
                    SELECT COUNT(DISTINCT origin_key)
                    FROM candidate_evidence
                    WHERE candidate_id = ?
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (candidate_id, candidate_id),
        )

    async def get_candidate(self, candidate_id: int) -> Candidate | None:
        """Return one candidate."""
        async with self.conn.execute(
            "SELECT * FROM job_candidates WHERE id = ?",
            (candidate_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return _candidate_from_row(row) if row is not None else None

    async def score_candidate(
        self,
        candidate_id: int,
        *,
        policy_score: float,
        llm_score: float | None = None,
        risk_flags: Iterable[str] = (),
        state: CandidateState = CandidateState.SCORED,
    ) -> int:
        """Store a score and bump the candidate version.

        The version is what makes a stale approval detectable: a button pressed
        after a rescore refers to a candidate that no longer exists as scored.
        """
        flags = json.dumps(sorted(set(risk_flags)), separators=(",", ":"))
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE job_candidates
                SET policy_score = ?,
                    llm_score = ?,
                    risk_flags = ?,
                    state = ?,
                    version = version + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (policy_score, llm_score, flags, state.value, candidate_id),
            )
            await self.conn.commit()
            async with self.conn.execute(
                "SELECT version FROM job_candidates WHERE id = ?",
                (candidate_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return int(row["version"]) if row is not None else 0

    async def transition_candidate(
        self,
        candidate_id: int,
        state: CandidateState,
        *,
        expected_version: int | None = None,
    ) -> bool:
        """Move a candidate to a new state, optionally guarding on version.

        A candidate blocked for unproven public scope is never reopened here:
        approval must not be able to override the privacy gate.
        """
        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                async with self.conn.execute(
                    "SELECT state, version FROM job_candidates WHERE id = ?",
                    (candidate_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    await self.conn.rollback()
                    return False
                if CandidateState(str(row["state"])) is CandidateState.BLOCKED_PRIVATE:
                    await self.conn.rollback()
                    return False
                if expected_version is not None and int(row["version"]) != expected_version:
                    await self.conn.rollback()
                    return False

                await self.conn.execute(
                    """
                    UPDATE job_candidates
                    SET state = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (state.value, candidate_id),
                )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise
        return True

    async def candidates_for_job(
        self,
        job_id: str,
        *,
        states: Iterable[CandidateState] | None = None,
    ) -> list[Candidate]:
        """List a job's candidates, newest score first."""
        query = "SELECT * FROM job_candidates WHERE job_id = ?"
        params: list[str] = [job_id]
        if states is not None:
            wanted = [state.value for state in states]
            if not wanted:
                return []
            query += f" AND state IN ({','.join('?' for _ in wanted)})"
            params.extend(wanted)
        query += " ORDER BY COALESCE(policy_score, -1) DESC, id"
        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [_candidate_from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Frontier
    # ------------------------------------------------------------------

    async def claim_frontier(
        self,
        job_id: str,
        *,
        limit: int = 1,
        lease_seconds: int = 300,
    ) -> list[FrontierItem]:
        """Claim the highest-priority pending work for a job.

        Priority order rather than insertion order: hubs and owner seeds must
        be crawled before whatever a noisy chat mentioned first.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be at least 1")

        claim_token = secrets.token_urlsafe(24)
        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self.conn.execute(
                    """
                    WITH claimable AS (
                        SELECT candidate_id
                        FROM recon_frontier
                        WHERE job_id = ?
                          AND state IN ('pending', 'claimed')
                          AND (not_before IS NULL OR datetime(not_before) <= CURRENT_TIMESTAMP)
                          AND (claimed_until IS NULL OR datetime(claimed_until) <= CURRENT_TIMESTAMP)
                        ORDER BY priority DESC, candidate_id
                        LIMIT ?
                    )
                    UPDATE recon_frontier
                    SET state = 'claimed',
                        claimed_until = datetime('now', ?),
                        lease_owner = ?,
                        attempts = attempts + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE candidate_id IN (SELECT candidate_id FROM claimable)
                    RETURNING candidate_id
                    """,
                    (job_id, limit, f"+{lease_seconds} seconds", claim_token),
                )
                claimed = [int(row[0]) for row in await cursor.fetchall()]
                if not claimed:
                    await self.conn.commit()
                    return []

                ids_json = json.dumps(claimed, separators=(",", ":"))
                cursor = await self.conn.execute(
                    """
                    SELECT f.candidate_id, f.job_id, f.priority, f.attempts,
                           c.chat_uuid, c.wave
                    FROM recon_frontier AS f
                    JOIN job_candidates AS c ON c.id = f.candidate_id
                    WHERE f.candidate_id IN (
                        SELECT CAST(value AS INTEGER) FROM json_each(?)
                    )
                    ORDER BY f.priority DESC, f.candidate_id
                    """,
                    (ids_json,),
                )
                rows = await cursor.fetchall()
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise

        return [
            FrontierItem(
                candidate_id=int(row["candidate_id"]),
                job_id=str(row["job_id"]),
                chat_uuid=str(row["chat_uuid"]),
                wave=int(row["wave"]),
                priority=float(row["priority"]),
                attempts=int(row["attempts"]),
                claim_token=claim_token,
            )
            for row in rows
        ]

    async def settle_frontier(
        self,
        candidate_id: int,
        *,
        claim_token: str,
        state: str = "done",
        retry_after_seconds: int | None = None,
        error: str | None = None,
    ) -> bool:
        """Release a claimed frontier item, honouring the lease owner.

        A worker whose lease already expired must not overwrite the state that
        a later worker recorded.
        """
        if state not in {"pending", "done", "skipped"}:
            raise ValueError("state must be pending, done, or skipped")

        retry_modifier = (
            f"+{retry_after_seconds} seconds" if retry_after_seconds is not None else None
        )
        async with self._write_lock:
            cursor = await self.conn.execute(
                """
                UPDATE recon_frontier
                SET state = ?,
                    claimed_until = NULL,
                    lease_owner = NULL,
                    last_error = ?,
                    not_before = CASE
                        WHEN ? IS NULL THEN not_before
                        ELSE datetime('now', ?)
                    END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE candidate_id = ? AND lease_owner = ?
                RETURNING candidate_id
                """,
                (state, error, retry_modifier, retry_modifier, candidate_id, claim_token),
            )
            settled = await cursor.fetchone()
            await self.conn.commit()
        return settled is not None

    # ------------------------------------------------------------------
    # Action budgets
    # ------------------------------------------------------------------

    async def reserve_action(
        self,
        *,
        account_id: str,
        kind: ActionKind,
        idempotency_key: str,
        job_id: str | None = None,
        candidate_id: int | None = None,
        policy: dict[ActionKind, BudgetRule] | None = None,
    ) -> ReservationOutcome:
        """Reserve one slot for an MTProto call, or explain the refusal.

        Reservation happens before the call and is never refunded. An attempt
        whose outcome is unknown after a crash must cost its slot: repeating it
        is how a single ambiguous join turns into two.
        """
        rules = policy or DEFAULT_BUDGET_POLICY
        rule = rules.get(kind, BudgetRule())

        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                async with self.conn.execute(
                    "SELECT id, kind FROM telegram_actions WHERE idempotency_key = ?",
                    (idempotency_key,),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is not None:
                    await self.conn.commit()
                    return ActionReservation(
                        action_id=int(existing["id"]),
                        kind=ActionKind(str(existing["kind"])),
                        idempotency_key=idempotency_key,
                        replayed=True,
                    )

                cooldown = await self._active_cooldown_locked(account_id, kind)
                if cooldown is not None:
                    await self.conn.commit()
                    return cooldown

                denial = await self._budget_denial_locked(account_id, kind, rule)
                if denial is not None:
                    await self.conn.commit()
                    return denial

                cursor = await self.conn.execute(
                    """
                    INSERT INTO telegram_actions (
                        account_id, kind, idempotency_key, job_id, candidate_id, outcome
                    )
                    VALUES (?, ?, ?, ?, ?, 'reserved')
                    RETURNING id
                    """,
                    (account_id, kind.value, idempotency_key, job_id, candidate_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RuntimeError("action reservation did not produce a readable row")
                action_id = int(row[0])
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise

        return ActionReservation(
            action_id=action_id,
            kind=kind,
            idempotency_key=idempotency_key,
        )

    async def _active_cooldown_locked(
        self,
        account_id: str,
        kind: ActionKind,
    ) -> BudgetDenial | None:
        """Return a denial when the account or action class is paused."""
        async with self.conn.execute(
            """
            SELECT scope, reason, manual_resume_required,
                   CAST(
                       (julianday(blocked_until) - julianday('now')) * 86400 AS INTEGER
                   ) AS remaining
            FROM account_cooldowns
            WHERE account_id = ?
              AND scope IN (?, 'all')
              AND datetime(blocked_until) > CURRENT_TIMESTAMP
            ORDER BY datetime(blocked_until) DESC
            LIMIT 1
            """,
            (account_id, kind.value),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        remaining = int(row["remaining"] or 0)
        manual = bool(int(row["manual_resume_required"]))
        return BudgetDenial(
            kind=kind,
            scope=BudgetScope.COOLDOWN,
            used=0,
            cap=None,
            retry_after_seconds=None if manual else max(remaining, 1),
            reason=str(row["reason"]),
        )

    async def _budget_denial_locked(
        self,
        account_id: str,
        kind: ActionKind,
        rule: BudgetRule,
    ) -> BudgetDenial | None:
        """Return a denial when a rolling window is already full."""
        if rule.min_interval_seconds:
            async with self.conn.execute(
                """
                SELECT CAST(
                           (julianday('now') - julianday(MAX(reserved_at))) * 86400 AS INTEGER
                       ) AS since
                FROM telegram_actions
                WHERE account_id = ? AND kind = ?
                """,
                (account_id, kind.value),
            ) as cursor:
                row = await cursor.fetchone()
            since = None if row is None or row["since"] is None else int(row["since"])
            if since is not None and since < rule.min_interval_seconds:
                return BudgetDenial(
                    kind=kind,
                    scope=BudgetScope.COOLDOWN,
                    used=1,
                    cap=None,
                    retry_after_seconds=max(rule.min_interval_seconds - since, 1),
                    reason=(
                        f"{kind.value} is paced at one per "
                        f"{rule.min_interval_seconds}s; last one was {since}s ago"
                    ),
                )

        for scope, cap, window in (
            (BudgetScope.HOUR, rule.per_hour, "-1 hour"),
            (BudgetScope.DAY, rule.per_day, "-24 hours"),
        ):
            if cap is None:
                continue
            async with self.conn.execute(
                """
                SELECT COUNT(*) AS used,
                       CAST(
                           (julianday(MIN(reserved_at)) - julianday('now', ?)) * 86400 AS INTEGER
                       ) AS retry_after
                FROM telegram_actions
                WHERE account_id = ?
                  AND kind = ?
                  AND datetime(reserved_at) > datetime('now', ?)
                """,
                (window, account_id, kind.value, window),
            ) as cursor:
                row = await cursor.fetchone()
            used = int(row["used"]) if row is not None else 0
            if used >= cap:
                retry_after = int(row["retry_after"] or 0) if row is not None else 0
                return BudgetDenial(
                    kind=kind,
                    scope=scope,
                    used=used,
                    cap=cap,
                    retry_after_seconds=max(retry_after, 1),
                    reason=f"{kind.value} budget exhausted for the rolling {scope.value}",
                )
        return None

    async def settle_action(
        self,
        action_id: int,
        *,
        outcome: ActionOutcome,
        duration_ms: float | None = None,
        flood_wait_seconds: int | None = None,
        error_code: str | None = None,
    ) -> None:
        """Record how a reserved action ended.

        ``duration_ms`` is kept even on success: Telethon sleeps through short
        FloodWaits inside the call, so an unexpectedly slow call is the only
        signal that the account is being throttled below the visible threshold.
        """
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE telegram_actions
                SET outcome = ?,
                    duration_ms = ?,
                    flood_wait_seconds = ?,
                    error_code = ?,
                    settled_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (outcome.value, duration_ms, flood_wait_seconds, error_code, action_id),
            )
            await self.conn.commit()

    async def set_cooldown(
        self,
        *,
        account_id: str,
        scope: str,
        seconds: int,
        reason: str,
        manual_resume_required: bool = False,
    ) -> None:
        """Pause an action class, or everything when scope is ``all``."""
        if seconds < 1:
            raise ValueError("seconds must be at least 1")
        async with self._write_lock:
            await self.conn.execute(
                """
                INSERT INTO account_cooldowns (
                    account_id, scope, blocked_until, reason, manual_resume_required
                )
                VALUES (?, ?, datetime('now', ?), ?, ?)
                ON CONFLICT(account_id, scope) DO UPDATE SET
                    blocked_until = MAX(
                        account_cooldowns.blocked_until,
                        excluded.blocked_until
                    ),
                    reason = excluded.reason,
                    manual_resume_required = MAX(
                        account_cooldowns.manual_resume_required,
                        excluded.manual_resume_required
                    )
                """,
                (
                    account_id,
                    scope,
                    f"+{seconds} seconds",
                    reason,
                    1 if manual_resume_required else 0,
                ),
            )
            await self.conn.commit()

    async def clear_cooldown(self, *, account_id: str, scope: str) -> None:
        """Lift a pause, which for a safety halt only an operator may do."""
        async with self._write_lock:
            await self.conn.execute(
                "DELETE FROM account_cooldowns WHERE account_id = ? AND scope = ?",
                (account_id, scope),
            )
            await self.conn.commit()

    # ------------------------------------------------------------------
    # Captured messages
    # ------------------------------------------------------------------

    async def store_message(self, message: ScoutMessage) -> bool:
        """Persist one crawled message, reporting whether it was new.

        Live capture and backfill converge on the same rows, so the primary key
        is what makes the seam between them safe: whichever arrives second is
        ignored instead of duplicating the message.
        """
        content_hash = (
            hashlib.sha256(" ".join(message.text.split()).lower().encode("utf-8")).hexdigest()
            if message.text
            else None
        )
        async with self._write_lock:
            cursor = await self.conn.execute(
                """
                INSERT INTO scout_messages (
                    chat_id, telegram_msg_id, sender_id, sender_name, text, date,
                    entities_json, forward_chat_id, forward_message_id, reply_to_message_id,
                    content_hash, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, telegram_msg_id) DO NOTHING
                RETURNING telegram_msg_id
                """,
                (
                    message.chat_id,
                    message.telegram_msg_id,
                    message.sender_id,
                    message.sender_name,
                    message.text,
                    message.date,
                    json.dumps(message.entities, separators=(",", ":"))
                    if message.entities
                    else None,
                    message.forward_chat_id,
                    message.forward_message_id,
                    message.reply_to_message_id,
                    content_hash,
                    message.source,
                ),
            )
            stored = await cursor.fetchone()
            if stored is None and message.reply_to_message_id is not None:
                await self.conn.execute(
                    """
                    UPDATE scout_messages
                       SET reply_to_message_id = ?
                     WHERE chat_id = ? AND telegram_msg_id = ?
                       AND reply_to_message_id IS NULL
                    """,
                    (
                        message.reply_to_message_id,
                        message.chat_id,
                        message.telegram_msg_id,
                    ),
                )
            await self.conn.commit()
        return stored is not None

    async def store_messages(self, messages: Sequence[ScoutMessage]) -> int:
        """Persist a backfill page in one transaction, returning new rows.

        Pages are written whole so that a crawl makes progress in units the
        cursor can resume from, and so the lock is taken once per page rather
        than once per message.
        """
        if not messages:
            return 0
        stored = 0
        async with self._write_lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                for message in messages:
                    content_hash = (
                        hashlib.sha256(
                            " ".join(message.text.split()).lower().encode("utf-8")
                        ).hexdigest()
                        if message.text
                        else None
                    )
                    cursor = await self.conn.execute(
                        """
                        INSERT INTO scout_messages (
                            chat_id, telegram_msg_id, sender_id, sender_name, text, date,
                            entities_json, forward_chat_id, forward_message_id,
                            reply_to_message_id, content_hash, source
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(chat_id, telegram_msg_id) DO NOTHING
                        RETURNING telegram_msg_id
                        """,
                        (
                            message.chat_id,
                            message.telegram_msg_id,
                            message.sender_id,
                            message.sender_name,
                            message.text,
                            message.date,
                            json.dumps(message.entities, separators=(",", ":"))
                            if message.entities
                            else None,
                            message.forward_chat_id,
                            message.forward_message_id,
                            message.reply_to_message_id,
                            content_hash,
                            message.source,
                        ),
                    )
                    inserted = await cursor.fetchone()
                    if inserted is not None:
                        stored += 1
                    elif message.reply_to_message_id is not None:
                        await self.conn.execute(
                            """
                            UPDATE scout_messages
                               SET reply_to_message_id = ?
                             WHERE chat_id = ? AND telegram_msg_id = ?
                               AND reply_to_message_id IS NULL
                            """,
                            (
                                message.reply_to_message_id,
                                message.chat_id,
                                message.telegram_msg_id,
                            ),
                        )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise
        return stored

    async def message_count(self, chat_id: int) -> int:
        """Return how many messages are stored for a chat."""
        async with self.conn.execute(
            "SELECT COUNT(*) FROM scout_messages WHERE chat_id = ?",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    # ------------------------------------------------------------------
    # Join queue
    # ------------------------------------------------------------------

    async def enqueue_join(
        self,
        *,
        chat_ref: str,
        label: str | None = None,
        watcher_name: str | None = None,
        target_days: int = 730,
    ) -> None:
        """Queue a public chat to be joined when the budget allows.

        Re-queueing a chat that was already joined, or whose request awaits an
        admin, does nothing: the queue is an intent to join once. Re-queueing
        one whose earlier attempt FAILED is a fresh instruction and revives
        the row — without this, a failed chat_ref was a silent no-op through
        every interface and could never be retried at all. The attempt counter
        is bumped so the retry runs under a fresh idempotency key.
        """
        normalized = normalize_username(chat_ref)
        if not normalized:
            raise ValueError("chat_ref must not be empty")
        async with self._write_lock:
            await self.conn.execute(
                """
                INSERT INTO join_queue (chat_ref, label, watcher_name, target_days)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_ref) DO UPDATE SET
                    label = COALESCE(excluded.label, join_queue.label),
                    watcher_name = COALESCE(excluded.watcher_name, join_queue.watcher_name),
                    target_days = MAX(join_queue.target_days, excluded.target_days),
                    state = CASE WHEN join_queue.state IN ('failed', 'skipped')
                                 THEN 'pending' ELSE join_queue.state END,
                    attempts = CASE WHEN join_queue.state IN ('failed', 'skipped')
                                    THEN join_queue.attempts + 1
                                    ELSE join_queue.attempts END,
                    not_before = CASE WHEN join_queue.state IN ('failed', 'skipped')
                                      THEN NULL ELSE join_queue.not_before END,
                    last_error = CASE WHEN join_queue.state IN ('failed', 'skipped')
                                      THEN NULL ELSE join_queue.last_error END,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (normalized, label, watcher_name, target_days),
            )
            await self.conn.commit()

    async def next_queued_join(self) -> QueuedJoin | None:
        """Return the next chat due to be joined, oldest request first."""
        async with self.conn.execute(
            """
            SELECT * FROM join_queue
            WHERE state = 'pending'
              AND (not_before IS NULL OR datetime(not_before) <= CURRENT_TIMESTAMP)
            ORDER BY created_at, chat_ref
            LIMIT 1
            """
        ) as cursor:
            row = await cursor.fetchone()
        return _queued_join_from_row(row) if row is not None else None

    async def settle_queued_join(
        self,
        chat_ref: str,
        *,
        state: JoinQueueState,
        error: str | None = None,
        joined_chat_id: int | None = None,
    ) -> None:
        """Record the terminal outcome of one queued join."""
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE join_queue
                SET state = ?,
                    attempts = attempts + 1,
                    last_error = ?,
                    joined_chat_id = COALESCE(?, joined_chat_id),
                    not_before = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chat_ref = ?
                """,
                (state.value, error, joined_chat_id, normalize_username(chat_ref)),
            )
            await self.conn.commit()

    async def defer_queued_join(
        self,
        chat_ref: str,
        *,
        seconds: int,
        error: str | None = None,
        count_attempt: bool = False,
    ) -> None:
        """Push a join back, counting it as an attempt only when one was made.

        Being refused by the budget is not a failed join: nothing reached
        Telegram, so the chat keeps its place in the queue and its attempt
        counter. A FloodWait DID reach Telegram — that attempt is spent, and
        counting it moves the retry onto a fresh idempotency key; without the
        bump the retry replays the settled reservation and the row wedges.
        """
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE join_queue
                SET not_before = datetime('now', ?),
                    last_error = ?,
                    attempts = attempts + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chat_ref = ?
                """,
                (
                    f"+{max(seconds, 1)} seconds",
                    error,
                    1 if count_attempt else 0,
                    normalize_username(chat_ref),
                ),
            )
            await self.conn.commit()

    async def joins_awaiting_reconciliation(self) -> list[QueuedJoin]:
        """Joins whose real outcome only Telegram knows.

        Two shapes land here: a request an admin may since have approved
        (state 'requested'), and a join whose process died mid-call, leaving
        an unsettled reservation behind ('pending' with already_attempted).
        Neither is ever re-checked by the join path itself — next_queued_join
        deliberately refuses the second — so a periodic reconciliation pass
        reads this list and compares it against the account's live dialogs.
        """
        async with self.conn.execute(
            """
            SELECT * FROM join_queue
            WHERE (state = 'requested'
                   OR (state = 'pending' AND last_error = 'already_attempted'))
              AND (not_before IS NULL OR datetime(not_before) <= CURRENT_TIMESTAMP)
            ORDER BY updated_at
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return [_queued_join_from_row(row) for row in rows]

    async def requeue_join_attempt(self, chat_ref: str) -> None:
        """Return a wedged join to the queue under a fresh attempt number.

        Called when reconciliation has positively established that the
        account is NOT a member — the interrupted attempt evidently never
        went through, so trying again is safe, and the bumped counter gives
        the retry an idempotency key of its own.
        """
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE join_queue
                SET state = 'pending',
                    attempts = attempts + 1,
                    last_error = 'reconciled_not_member',
                    not_before = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chat_ref = ?
                """,
                (normalize_username(chat_ref),),
            )
            await self.conn.commit()

    async def join_queue(self) -> list[QueuedJoin]:
        """Return the whole queue for reporting."""
        async with self.conn.execute(
            "SELECT * FROM join_queue ORDER BY state, created_at, chat_ref"
        ) as cursor:
            rows = await cursor.fetchall()
        return [_queued_join_from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Background history archive
    # ------------------------------------------------------------------

    async def add_backfill_target(
        self,
        *,
        chat_id: int,
        label: str | None = None,
        target_days: int = 730,
    ) -> None:
        """Register a chat whose history should be walked back over time.

        Re-registering an existing target only raises its horizon: a chat
        already archived two years back must not restart because someone
        later asked for one year.
        """
        if target_days < 1:
            raise ValueError("target_days must be at least 1")
        async with self._write_lock:
            await self.conn.execute(
                """
                INSERT INTO backfill_targets (chat_id, label, target_days)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    label = COALESCE(excluded.label, backfill_targets.label),
                    target_days = MAX(backfill_targets.target_days, excluded.target_days),
                    state = CASE
                        WHEN excluded.target_days > backfill_targets.target_days
                            AND backfill_targets.state = 'complete'
                        THEN 'pending'
                        ELSE backfill_targets.state
                    END,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, label, target_days),
            )
            await self.conn.commit()

    async def next_backfill_target(self) -> BackfillTarget | None:
        """Return the least-advanced chat that is ready for another page.

        Least-advanced first spreads the archive evenly instead of finishing
        one chat while the rest sit at zero, because the value here is
        comparable history across chats, not one deep hole.
        """
        async with self.conn.execute(
            """
            SELECT * FROM backfill_targets
            WHERE state = 'pending'
              AND (not_before IS NULL OR datetime(not_before) <= CURRENT_TIMESTAMP)
            ORDER BY pages_done, chat_id
            LIMIT 1
            """
        ) as cursor:
            row = await cursor.fetchone()
        return _backfill_from_row(row) if row is not None else None

    async def credit_archived_messages(self, chat_id: int, count: int) -> None:
        """Add discovery-phase stores to the per-chat archive counter.

        A silent no-op for a chat with no backfill target: the counter
        belongs to the target row, and a discovery crawl over a chat nobody
        asked to archive has nothing to credit.
        """
        if count <= 0:
            return
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE backfill_targets
                SET messages_stored = messages_stored + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chat_id = ?
                """,
                (count, chat_id),
            )
            await self.conn.commit()

    async def record_backfill_page(
        self,
        chat_id: int,
        *,
        stored: int,
        oldest_message_id: int | None,
        oldest_message_date: str | None,
        finished: BackfillState | None = None,
    ) -> None:
        """Advance a target's cursor after one page."""
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE backfill_targets
                SET messages_stored = messages_stored + ?,
                    pages_done = pages_done + 1,
                    oldest_message_id = COALESCE(?, oldest_message_id),
                    oldest_message_date = COALESCE(?, oldest_message_date),
                    state = COALESCE(?, state),
                    not_before = NULL,
                    last_error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chat_id = ?
                """,
                (
                    stored,
                    oldest_message_id,
                    oldest_message_date,
                    finished.value if finished is not None else None,
                    chat_id,
                ),
            )
            await self.conn.commit()

    async def defer_backfill_target(
        self,
        chat_id: int,
        *,
        seconds: int,
        error: str | None = None,
    ) -> None:
        """Hold a target back, usually because Telegram asked us to wait."""
        async with self._write_lock:
            await self.conn.execute(
                """
                UPDATE backfill_targets
                SET not_before = datetime('now', ?),
                    last_error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chat_id = ?
                """,
                (f"+{max(seconds, 1)} seconds", error, chat_id),
            )
            await self.conn.commit()

    async def backfill_targets(self) -> list[BackfillTarget]:
        """Return every target, most complete first, for reporting."""
        async with self.conn.execute(
            "SELECT * FROM backfill_targets ORDER BY messages_stored DESC, chat_id"
        ) as cursor:
            rows = await cursor.fetchall()
        return [_backfill_from_row(row) for row in rows]

    async def budget_usage(self, *, account_id: str, kind: ActionKind) -> tuple[int, int]:
        """Return actions used in the rolling hour and rolling day."""
        async with self.conn.execute(
            """
            SELECT
                SUM(datetime(reserved_at) > datetime('now', '-1 hour')) AS used_hour,
                SUM(datetime(reserved_at) > datetime('now', '-24 hours')) AS used_day
            FROM telegram_actions
            WHERE account_id = ? AND kind = ?
            """,
            (account_id, kind.value),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return 0, 0
        return int(row["used_hour"] or 0), int(row["used_day"] or 0)


def _job_from_row(row: aiosqlite.Row) -> ReconJob:
    return ReconJob(
        id=str(row["id"]),
        idempotency_key=str(row["idempotency_key"]),
        account_id=str(row["account_id"]),
        topic=str(row["topic"]),
        status=ReconJobStatus(str(row["status"])),
        location=str(row["location"]) if row["location"] is not None else None,
        languages=tuple(json.loads(str(row["languages"]))),
        seeds=tuple(json.loads(str(row["seeds"]))),
        current_wave=int(row["current_wave"]),
        max_waves=int(row["max_waves"]),
        lookback_days=int(row["lookback_days"]),
        max_candidates=int(row["max_candidates"]),
        max_join_attempts=int(row["max_join_attempts"]),
        stop_reason=str(row["stop_reason"]) if row["stop_reason"] is not None else None,
        error_code=str(row["error_code"]) if row["error_code"] is not None else None,
        resume_at=str(row["resume_at"]) if row["resume_at"] is not None else None,
        deadline_at=str(row["deadline_at"]) if row["deadline_at"] is not None else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        completed_at=str(row["completed_at"]) if row["completed_at"] is not None else None,
    )


def _queued_join_from_row(row: aiosqlite.Row) -> QueuedJoin:
    return QueuedJoin(
        chat_ref=str(row["chat_ref"]),
        label=str(row["label"]) if row["label"] is not None else None,
        watcher_name=str(row["watcher_name"]) if row["watcher_name"] is not None else None,
        target_days=int(row["target_days"]),
        state=JoinQueueState(str(row["state"])),
        attempts=int(row["attempts"]),
        last_error=str(row["last_error"]) if row["last_error"] is not None else None,
        joined_chat_id=(int(row["joined_chat_id"]) if row["joined_chat_id"] is not None else None),
    )


def _backfill_from_row(row: aiosqlite.Row) -> BackfillTarget:
    return BackfillTarget(
        chat_id=int(row["chat_id"]),
        target_days=int(row["target_days"]),
        label=str(row["label"]) if row["label"] is not None else None,
        oldest_message_id=(
            int(row["oldest_message_id"]) if row["oldest_message_id"] is not None else None
        ),
        oldest_message_date=(
            str(row["oldest_message_date"]) if row["oldest_message_date"] is not None else None
        ),
        messages_stored=int(row["messages_stored"]),
        pages_done=int(row["pages_done"]),
        state=BackfillState(str(row["state"])),
        last_error=str(row["last_error"]) if row["last_error"] is not None else None,
    )


def _chat_from_row(row: aiosqlite.Row) -> ScoutChat:
    return ScoutChat(
        id=str(row["id"]),
        telegram_chat_id=(
            int(row["telegram_chat_id"]) if row["telegram_chat_id"] is not None else None
        ),
        username=str(row["username"]) if row["username"] is not None else None,
        title=str(row["title"]) if row["title"] is not None else None,
        chat_type=str(row["chat_type"]) if row["chat_type"] is not None else None,
        participants=int(row["participants"]) if row["participants"] is not None else None,
        risk_flags=tuple(json.loads(str(row["risk_flags"]))),
        visibility=ChatVisibility(str(row["visibility"])),
        visibility_source=(
            str(row["visibility_source"]) if row["visibility_source"] is not None else None
        ),
        membership=ChatMembership(str(row["membership"])),
        joined_at=str(row["joined_at"]) if row["joined_at"] is not None else None,
    )


def _candidate_from_row(row: aiosqlite.Row) -> Candidate:
    return Candidate(
        id=int(row["id"]),
        job_id=str(row["job_id"]),
        chat_uuid=str(row["chat_uuid"]),
        wave=int(row["wave"]),
        state=CandidateState(str(row["state"])),
        policy_score=float(row["policy_score"]) if row["policy_score"] is not None else None,
        llm_score=float(row["llm_score"]) if row["llm_score"] is not None else None,
        independent_sources=int(row["independent_sources"]),
        risk_flags=tuple(json.loads(str(row["risk_flags"]))),
        version=int(row["version"]),
    )
