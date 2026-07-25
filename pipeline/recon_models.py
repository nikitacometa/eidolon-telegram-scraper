"""Typed contracts for reconnaissance jobs, candidates, and throttled actions.

Reconnaissance state is deliberately separate from the live monitoring
contracts in :mod:`pipeline.models`: a crawl is a long-running, resumable
workflow with its own budgets, while monitoring is a per-message funnel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ReconJobStatus(StrEnum):
    """Lifecycle state of one reconnaissance job."""

    QUEUED = "queued"
    DISCOVERING = "discovering"
    WAITING_APPROVAL = "waiting_approval"
    JOINING = "joining"
    BACKFILLING = "backfilling"
    ANALYZING = "analyzing"
    PAUSED_RATE_LIMIT = "paused_rate_limit"
    SAFETY_HALT = "safety_halt"
    COMPLETED = "completed"
    COMPLETED_PARTIAL = "completed_partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATUSES: frozenset[ReconJobStatus] = frozenset(
    {
        ReconJobStatus.COMPLETED,
        ReconJobStatus.COMPLETED_PARTIAL,
        ReconJobStatus.FAILED,
        ReconJobStatus.CANCELLED,
    }
)

ACTIVE_JOB_STATUSES: frozenset[ReconJobStatus] = frozenset(
    {
        ReconJobStatus.QUEUED,
        ReconJobStatus.DISCOVERING,
        ReconJobStatus.WAITING_APPROVAL,
        ReconJobStatus.JOINING,
        ReconJobStatus.BACKFILLING,
        ReconJobStatus.ANALYZING,
        ReconJobStatus.PAUSED_RATE_LIMIT,
    }
)


class CandidateState(StrEnum):
    """Progress of one chat candidate inside a job.

    ``BLOCKED_PRIVATE`` is terminal by policy: an approval must never be able
    to promote a chat whose public scope was not independently proven.
    """

    DISCOVERED = "discovered"
    SCORED = "scored"
    BLOCKED_PRIVATE = "blocked_private"
    REJECTED = "rejected"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    JOIN_REQUESTED = "join_requested"
    JOINED = "joined"
    JOIN_FAILED = "join_failed"


class ChatVisibility(StrEnum):
    """Whether a chat was proven to be publicly accessible."""

    UNKNOWN = "unknown"
    PUBLIC = "public"
    PRIVATE = "private"


class ChatMembership(StrEnum):
    """Account membership state for a chat.

    ``REQUESTED`` exists because ``INVITE_REQUEST_SENT`` is not a join: the
    request awaits admin approval and must never be treated as membership.
    """

    NONE = "none"
    REQUESTED = "requested"
    MEMBER = "member"
    LEFT = "left"
    FAILED = "failed"


class AliasKind(StrEnum):
    """Identity key used to canonicalize a chat before it is resolved."""

    PEER_ID = "peer_id"
    USERNAME = "username"
    INVITE = "invite"


class DiscoverySource(StrEnum):
    """How a candidate entered the graph."""

    OWNER_SEED = "owner_seed"
    HASHTAG_SEARCH = "hashtag_search"
    FULLTEXT_SEARCH = "fulltext_search"
    CONTACTS_SEARCH = "contacts_search"
    GLOBAL_SEARCH = "global_search"
    RECOMMENDATIONS = "recommendations"
    MESSAGE_LINK = "message_link"
    MESSAGE_MENTION = "message_mention"
    FORWARD = "forward"


class ActionKind(StrEnum):
    """A rate-limited MTProto action class."""

    JOIN = "join"
    HASHTAG_SEARCH = "hashtag_search"
    FULLTEXT_SEARCH = "fulltext_search"
    CONTACTS_SEARCH = "contacts_search"
    GLOBAL_SEARCH = "global_search"
    RECOMMENDATIONS = "recommendations"
    RESOLVE_USERNAME = "resolve_username"
    INVITE_CHECK = "invite_check"
    HISTORY_PAGE = "history_page"


class ActionOutcome(StrEnum):
    """Terminal state of a reserved action.

    A reservation is never refunded: an attempt that reached Telegram consumed
    account standing regardless of what came back, and an ambiguous crash must
    cost a slot rather than risk a blind retry.
    """

    RESERVED = "reserved"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    FLOOD_WAIT = "flood_wait"
    AMBIGUOUS = "ambiguous"


class BudgetScope(StrEnum):
    """Which limit refused an action reservation."""

    HOUR = "hour"
    DAY = "day"
    COOLDOWN = "cooldown"


@dataclass(frozen=True, slots=True)
class BudgetRule:
    """Rolling caps for one action class.

    Windows roll rather than reset at midnight: a calendar boundary would let
    a full day's budget burn twice within a minute.
    """

    per_hour: int | None = None
    per_day: int | None = None

    def __post_init__(self) -> None:
        if self.per_hour is not None and self.per_hour < 0:
            raise ValueError("per_hour must not be negative")
        if self.per_day is not None and self.per_day < 0:
            raise ValueError("per_day must not be negative")


# Starting operator policy, not documented Telegram limits: no public source
# states a safe rate. These are deliberately below folklore consensus and are
# meant to be tightened from observed FloodWait telemetry.
DEFAULT_BUDGET_POLICY: dict[ActionKind, BudgetRule] = {
    ActionKind.JOIN: BudgetRule(per_hour=1, per_day=3),
    ActionKind.FULLTEXT_SEARCH: BudgetRule(per_day=5),
    ActionKind.HASHTAG_SEARCH: BudgetRule(per_hour=10, per_day=20),
    ActionKind.CONTACTS_SEARCH: BudgetRule(per_hour=10, per_day=20),
    ActionKind.GLOBAL_SEARCH: BudgetRule(per_hour=5, per_day=10),
    ActionKind.RECOMMENDATIONS: BudgetRule(per_hour=10, per_day=20),
    ActionKind.RESOLVE_USERNAME: BudgetRule(per_hour=30, per_day=100),
    ActionKind.INVITE_CHECK: BudgetRule(per_hour=10, per_day=30),
    ActionKind.HISTORY_PAGE: BudgetRule(per_hour=600, per_day=5000),
}


@dataclass(frozen=True, slots=True)
class ActionReservation:
    """A granted slot for exactly one MTProto call."""

    action_id: int
    kind: ActionKind
    idempotency_key: str
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class BudgetDenial:
    """A refused reservation and why it was refused."""

    kind: ActionKind
    scope: BudgetScope
    used: int
    cap: int | None
    retry_after_seconds: int | None
    reason: str = ""


ReservationOutcome = ActionReservation | BudgetDenial


@dataclass(frozen=True, slots=True)
class ReconJob:
    """Durable reconnaissance job record."""

    id: str
    idempotency_key: str
    account_id: str
    topic: str
    status: ReconJobStatus
    location: str | None = None
    languages: tuple[str, ...] = ()
    seeds: tuple[str, ...] = ()
    current_wave: int = 0
    max_waves: int = 2
    lookback_days: int = 30
    max_candidates: int = 100
    max_join_attempts: int = 6
    stop_reason: str | None = None
    error_code: str | None = None
    resume_at: str | None = None
    deadline_at: str | None = None
    created_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether the job reached an immutable end state."""
        return self.status in TERMINAL_JOB_STATUSES


@dataclass(frozen=True, slots=True)
class ScoutChat:
    """Canonical record for a chat observed by reconnaissance."""

    id: str
    telegram_chat_id: int | None = None
    username: str | None = None
    title: str | None = None
    chat_type: str | None = None
    visibility: ChatVisibility = ChatVisibility.UNKNOWN
    visibility_source: str | None = None
    membership: ChatMembership = ChatMembership.NONE
    joined_at: str | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    """A chat under evaluation inside one job."""

    id: int
    job_id: str
    chat_uuid: str
    wave: int
    state: CandidateState
    policy_score: float | None = None
    llm_score: float | None = None
    independent_sources: int = 0
    risk_flags: tuple[str, ...] = ()
    version: int = 1


@dataclass(frozen=True, slots=True)
class FrontierItem:
    """A claimed unit of crawl work."""

    candidate_id: int
    job_id: str
    chat_uuid: str
    wave: int
    priority: float
    attempts: int
    claim_token: str


@dataclass(frozen=True, slots=True)
class Evidence:
    """One independent signal supporting a candidate.

    ``origin_key`` is what makes a signal independent. Counting distinct source
    chats instead would let a single actor manufacture independence by
    reposting the same message into two chats the crawl already reads.
    """

    source: DiscoverySource
    origin_key: str
    source_chat_id: int | None = None
    source_message_id: int | None = None
    snippet: str | None = None


@dataclass(frozen=True, slots=True)
class JobRequest:
    """Validated intent to start reconnaissance."""

    idempotency_key: str
    topic: str
    account_id: str = "owner-primary"
    location: str | None = None
    languages: tuple[str, ...] = ()
    seeds: tuple[str, ...] = ()
    max_waves: int = 2
    lookback_days: int = 30
    max_candidates: int = 100
    max_join_attempts: int = 6
    deadline_hours: int = 72

    def __post_init__(self) -> None:
        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")
        if not self.topic.strip():
            raise ValueError("topic must not be empty")
        if not 1 <= self.max_waves <= 2:
            raise ValueError("max_waves must be 1 or 2")
        if not 1 <= self.lookback_days <= 90:
            raise ValueError("lookback_days must be between 1 and 90")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if self.max_join_attempts < 0:
            raise ValueError("max_join_attempts must not be negative")
        if self.deadline_hours < 1:
            raise ValueError("deadline_hours must be at least 1")


@dataclass(slots=True)
class ChatIdentity:
    """Aliases known for one chat, in canonicalization priority order."""

    peer_id: int | None = None
    username: str | None = None
    invite_fingerprint: str | None = None
    title: str | None = None
    chat_type: str | None = None
    aliases: list[tuple[AliasKind, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.username is not None:
            self.username = normalize_username(self.username)
        resolved: list[tuple[AliasKind, str]] = []
        if self.peer_id is not None:
            resolved.append((AliasKind.PEER_ID, str(self.peer_id)))
        if self.username:
            resolved.append((AliasKind.USERNAME, self.username))
        if self.invite_fingerprint:
            resolved.append((AliasKind.INVITE, self.invite_fingerprint))
        if not resolved:
            raise ValueError("a chat identity needs a peer id, username, or invite fingerprint")
        self.aliases = resolved


def normalize_username(value: str) -> str:
    """Return a comparable username form for aliasing and deduplication."""
    trimmed = value.strip().lower()
    for prefix in ("https://t.me/", "http://t.me/", "t.me/", "@"):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix) :]
            break
    return trimmed.split("?", 1)[0].strip("/")
