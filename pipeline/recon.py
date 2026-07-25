"""Running a reconnaissance job from a topic to a set of watched chats.

The runner is deliberately a straight line with hard stops rather than a
clever search. Everything expensive is budgeted by the governor, everything
irreversible is gated on a proven-public chat and an explicit decision, and
every step writes its state so that a restart resumes instead of restarting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from pipeline.crawler import TelegramCrawler
from pipeline.discovery import ChatLink, DiscoveredChat, TelegramDiscovery
from pipeline.governor import ActionResult, ActionStatus, TelegramActionGovernor
from pipeline.models import ObservationMode, ObservationSource
from pipeline.recon_models import (
    ActionKind,
    CandidateState,
    ChatIdentity,
    ChatMembership,
    ChatVisibility,
    DiscoverySource,
    Evidence,
    ReconJob,
    ReconJobStatus,
)
from pipeline.scoring import ScoringPolicy, build_policy, score_candidate
from storage.db import Database
from storage.scout import ScoutDatabase

logger = logging.getLogger(__name__)

DEFAULT_BACKFILL_PAGES = 5


@dataclass(frozen=True, slots=True)
class ChatFinding:
    """One chat as it ended up after a job."""

    chat_ref: str
    title: str | None
    score: float
    state: CandidateState
    reason: str
    chat_id: int | None = None
    messages_stored: int = 0


@dataclass(slots=True)
class ReconReport:
    """What a job found, for the owner rather than for the machine."""

    job_id: str
    status: ReconJobStatus
    stop_reason: str
    waves_completed: int = 0
    joined: list[ChatFinding] = field(default_factory=list)
    recommended: list[ChatFinding] = field(default_factory=list)
    awaiting_approval: list[ChatFinding] = field(default_factory=list)
    rejected: int = 0
    blocked_private: int = 0
    messages_stored: int = 0

    @property
    def total_candidates(self) -> int:
        """Every chat the job considered."""
        return (
            len(self.joined)
            + len(self.recommended)
            + len(self.awaiting_approval)
            + self.rejected
            + self.blocked_private
        )


class ReconRunner:
    """Drives one reconnaissance job to a terminal state."""

    def __init__(
        self,
        *,
        scout: ScoutDatabase,
        db: Database,
        client: object,
        governor: TelegramActionGovernor,
        discovery: TelegramDiscovery,
        crawler: TelegramCrawler,
        backfill_pages_per_chat: int = DEFAULT_BACKFILL_PAGES,
    ) -> None:
        self._scout = scout
        self._db = db
        self._client = client
        self._governor = governor
        self._discovery = discovery
        self._crawler = crawler
        self._backfill_pages = backfill_pages_per_chat
        # Every attempt counts, including one that only produced a pending
        # request or an ambiguous failure: each of those reached Telegram.
        self._join_attempts = 0

    async def run(self, job: ReconJob) -> ReconReport:
        """Execute a job wave by wave until a stop condition is reached."""
        policy = build_policy(topic=job.topic, location=job.location)
        report = ReconReport(job_id=job.id, status=job.status, stop_reason="")

        await self._scout.update_job_status(job.id, ReconJobStatus.DISCOVERING)
        await self._seed(job, policy, report)

        stop_reason = ""
        for wave in range(job.max_waves):
            found = await self._discover_wave(job, policy, wave, report)
            if found is not None:
                stop_reason = found
                break

            report.waves_completed = wave + 1
            if not await self._has_pending_work(job):
                stop_reason = "frontier empty"
                break
        else:
            stop_reason = "wave limit reached"

        status = (
            ReconJobStatus.COMPLETED
            if stop_reason in {"frontier empty", "wave limit reached"}
            else ReconJobStatus.COMPLETED_PARTIAL
        )
        await self._scout.update_job_status(
            job.id,
            status,
            stop_reason=stop_reason,
            current_wave=report.waves_completed,
        )
        report.status = status
        report.stop_reason = stop_reason
        await self._fill_report(job, report)
        return report

    # ------------------------------------------------------------------
    # Waves
    # ------------------------------------------------------------------

    async def _seed(self, job: ReconJob, policy: ScoringPolicy, report: ReconReport) -> None:
        """Register the chats the owner named as wave-zero candidates."""
        for seed in job.seeds:
            described = await self._resolve(job, seed)
            if described is None:
                continue
            described.evidence = Evidence(
                source=DiscoverySource.OWNER_SEED,
                origin_key=f"owner:{seed}",
            )
            await self._register(job, described, wave=0)

    async def _discover_wave(
        self,
        job: ReconJob,
        policy: ScoringPolicy,
        wave: int,
        report: ReconReport,
    ) -> str | None:
        """Run one wave. Returns a stop reason when the job cannot continue."""
        if wave == 0:
            halt = await self._search_surfaces(job, policy)
            if halt is not None:
                return halt

        candidates = await self._scout.candidates_for_job(
            job.id,
            states=[CandidateState.DISCOVERED],
        )
        if not candidates:
            return None

        for candidate in candidates:
            chat = await self._scout.get_chat(candidate.chat_uuid)
            if chat is None:
                continue
            described = _from_stored(chat)
            score = score_candidate(
                described,
                policy=policy,
                independent_sources=candidate.independent_sources,
                wave=candidate.wave,
            )
            version = await self._scout.score_candidate(
                candidate.id,
                policy_score=score.value,
                risk_flags=score.risk_flags,
            )
            await self._scout.transition_candidate(
                candidate.id,
                score.decision,
                expected_version=version,
            )

            if score.decision is not CandidateState.APPROVED:
                continue
            if job.max_join_attempts == 0:
                # Discovery-only run: everything is scored and recorded, but
                # nothing the account does is visible to anyone.
                continue
            # Counted here rather than from the report, which is only filled
            # once the wave loop is over: reading it during the loop would
            # always see zero and the cap would never bite.
            if self._join_attempts >= job.max_join_attempts:
                return "join attempt limit reached"

            halt = await self._join_and_read(job, candidate.id, chat, wave, report)
            if halt is not None:
                return halt
        return None

    async def _search_surfaces(self, job: ReconJob, policy: ScoringPolicy) -> str | None:
        """Query Telegram's public search surfaces for the job's topic."""
        queries = _search_terms(policy)
        for term in queries:
            result = await self._discovery.search_hashtag(job_id=job.id, hashtag=term)
            halt = _halt_reason(result)
            if halt is not None:
                return halt
            for described in result.value or ():
                await self._register(job, described, wave=0)

        for term in queries[:2]:
            result = await self._discovery.search_contacts(job_id=job.id, query=term)
            halt = _halt_reason(result)
            if halt is not None:
                return halt
            for described in result.value or ():
                await self._register(job, described, wave=0)
        return None

    # ------------------------------------------------------------------
    # Joining and reading
    # ------------------------------------------------------------------

    async def _join_and_read(
        self,
        job: ReconJob,
        candidate_id: int,
        chat: object,
        wave: int,
        report: ReconReport,
    ) -> str | None:
        chat_ref = str(getattr(chat, "username", None) or getattr(chat, "id", ""))
        peer_id = getattr(chat, "telegram_chat_id", None)
        entity = await self._input_entity(job, chat_ref)
        if entity is None:
            await self._scout.transition_candidate(candidate_id, CandidateState.JOIN_FAILED)
            return None

        self._join_attempts += 1
        result = await self._crawler.join(
            job_id=job.id,
            candidate_id=candidate_id,
            channel=entity,
            chat_ref=chat_ref,
        )
        if result.status is ActionStatus.REPLAYED:
            # A previous run already attempted this join. Its real outcome has
            # to be reconciled against Telegram rather than guessed at here.
            await self._scout.transition_candidate(candidate_id, CandidateState.JOIN_FAILED)
            logger.warning("Join for %s was already attempted; leaving it for review", chat_ref)
            return None
        halt = _halt_reason(result)
        if halt is not None:
            return halt

        outcome = result.value
        if outcome is None or not outcome.is_member:
            membership = outcome.membership if outcome else ChatMembership.FAILED
            await self._scout.set_chat_membership(str(getattr(chat, "id", "")), membership)
            await self._scout.transition_candidate(
                candidate_id,
                CandidateState.JOIN_REQUESTED
                if membership is ChatMembership.REQUESTED
                else CandidateState.JOIN_FAILED,
            )
            return None

        await self._scout.set_chat_membership(str(getattr(chat, "id", "")), ChatMembership.MEMBER)
        await self._scout.transition_candidate(candidate_id, CandidateState.JOINED)

        if peer_id is not None:
            # Registering before reading means everything posted from now on is
            # captured live, not only what history still serves.
            await self._db.observe_chat(
                chat_id=int(peer_id),
                mode=ObservationMode.RECON,
                source=ObservationSource.RECON,
                title=getattr(chat, "title", None),
                job_id=job.id,
            )
            stored = await self._backfill(job, int(peer_id), entity, wave)
            report.messages_stored += stored
        return None

    async def _backfill(
        self,
        job: ReconJob,
        chat_id: int,
        entity: object,
        wave: int,
    ) -> int:
        """Read a bounded amount of history and queue what it points at."""
        stored = 0
        offset_id = 0
        for _ in range(self._backfill_pages):
            result = await self._crawler.history_page(
                job_id=job.id,
                chat_id=chat_id,
                peer=entity,
                offset_id=offset_id,
                not_before=_lookback_cutoff(job.lookback_days),
            )
            if not result.ok or result.value is None:
                break

            page = result.value
            stored += await self._scout.store_messages(list(page.messages))
            if wave + 1 < job.max_waves:
                await self._queue_links(job, page.links, wave + 1)
            if page.exhausted or page.next_offset_id is None:
                break
            offset_id = page.next_offset_id
        return stored

    async def _queue_links(self, job: ReconJob, links: tuple[ChatLink, ...], wave: int) -> None:
        """Turn links found in history into next-wave candidates.

        Invite links are recorded as candidates but never resolved into a
        join: an invite is not proof that a chat is public.
        """
        for link in links:
            if link.username is None:
                continue
            identity = ChatIdentity(username=link.username)
            chat_uuid = await self._scout.resolve_chat(identity)
            await self._scout.add_candidate(
                job_id=job.id,
                chat_uuid=chat_uuid,
                wave=wave,
                evidence=Evidence(
                    source=DiscoverySource.MESSAGE_LINK,
                    origin_key=f"link:{link.username}",
                ),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _register(self, job: ReconJob, described: DiscoveredChat, wave: int) -> int | None:
        """Store a discovered chat and attach its evidence."""
        existing = await self._scout.candidates_for_job(job.id)
        if len(existing) >= job.max_candidates:
            return None

        chat_uuid = await self._scout.resolve_chat(described.identity)
        await self._scout.record_chat_profile(
            chat_uuid,
            participants=described.participants,
            risk_flags=described.flags,
            chat_type=described.chat_type,
            title=described.title,
        )
        if described.visibility is ChatVisibility.PUBLIC:
            await self._scout.set_chat_visibility(
                chat_uuid,
                ChatVisibility.PUBLIC,
                source=described.evidence.source.value,
            )
        return await self._scout.add_candidate(
            job_id=job.id,
            chat_uuid=chat_uuid,
            wave=wave,
            evidence=described.evidence,
        )

    async def _resolve(self, job: ReconJob, reference: str) -> DiscoveredChat | None:
        """Resolve a username the owner supplied into a described chat."""
        from pipeline.discovery import describe_chat

        async def call() -> object:
            return await self._client.get_entity(reference)  # type: ignore[attr-defined]

        result: ActionResult[object] = await self._governor.run(
            ActionKind.RESOLVE_USERNAME,
            f"{job.id}:resolve:{reference}",
            call,
            job_id=job.id,
        )
        if not result.ok or result.value is None:
            return None
        return describe_chat(result.value)

    async def _input_entity(self, job: ReconJob, reference: str) -> object | None:
        async def call() -> object:
            return await self._client.get_input_entity(reference)  # type: ignore[attr-defined]

        result: ActionResult[object] = await self._governor.run(
            ActionKind.RESOLVE_USERNAME,
            f"{job.id}:input:{reference}",
            call,
            job_id=job.id,
        )
        return result.value if result.ok else None

    async def _has_pending_work(self, job: ReconJob) -> bool:
        pending = await self._scout.candidates_for_job(
            job.id,
            states=[CandidateState.DISCOVERED],
        )
        return bool(pending)

    async def _fill_report(self, job: ReconJob, report: ReconReport) -> None:
        for candidate in await self._scout.candidates_for_job(job.id):
            chat = await self._scout.get_chat(candidate.chat_uuid)
            finding = ChatFinding(
                chat_ref=str(getattr(chat, "username", None) or candidate.chat_uuid),
                title=getattr(chat, "title", None),
                score=candidate.policy_score or 0.0,
                state=candidate.state,
                reason="",
                chat_id=getattr(chat, "telegram_chat_id", None),
                messages_stored=(
                    await self._scout.message_count(int(chat.telegram_chat_id))
                    if chat is not None and chat.telegram_chat_id is not None
                    else 0
                ),
            )
            if candidate.state is CandidateState.JOINED:
                report.joined.append(finding)
            elif candidate.state is CandidateState.APPROVED:
                report.recommended.append(finding)
            elif candidate.state is CandidateState.AWAITING_APPROVAL:
                report.awaiting_approval.append(finding)
            elif candidate.state is CandidateState.BLOCKED_PRIVATE:
                report.blocked_private += 1
            elif candidate.state in {CandidateState.REJECTED, CandidateState.JOIN_FAILED}:
                report.rejected += 1


def _lookback_cutoff(lookback_days: int) -> datetime:
    """The oldest message a job asked to read."""
    return datetime.now(UTC) - timedelta(days=lookback_days)


def _from_stored(chat: object) -> DiscoveredChat:
    """Rebuild a describable chat from its stored canonical record."""
    username = getattr(chat, "username", None)
    return DiscoveredChat(
        identity=ChatIdentity(
            peer_id=getattr(chat, "telegram_chat_id", None),
            username=username,
            title=getattr(chat, "title", None),
        ),
        evidence=Evidence(source=DiscoverySource.CONTACTS_SEARCH, origin_key="stored"),
        visibility=getattr(chat, "visibility", ChatVisibility.UNKNOWN),
        title=getattr(chat, "title", None),
        username=username,
        chat_type=getattr(chat, "chat_type", None),
        participants=getattr(chat, "participants", None),
        flags=tuple(getattr(chat, "risk_flags", ())),
    )


def _search_terms(policy: ScoringPolicy) -> list[str]:
    """Build the query list for wave zero, city first."""
    terms = [term.replace(" ", "") for term in policy.location_keywords[:3]]
    if policy.location_keywords and policy.topic_keywords:
        terms.append(f"{policy.location_keywords[0].replace(' ', '')}{policy.topic_keywords[0]}")
    return list(dict.fromkeys(term for term in terms if term))


def _halt_reason(result: ActionResult[object]) -> str | None:
    """Translate a governed refusal into a job-level stop reason."""
    if result.status is ActionStatus.HALTED:
        return f"halted: {result.error_code or 'safety'}"
    if result.status is ActionStatus.DENIED:
        return "budget exhausted"
    if result.status is ActionStatus.FLOOD_WAIT:
        return f"flood wait {result.flood_wait_seconds}s"
    return None
