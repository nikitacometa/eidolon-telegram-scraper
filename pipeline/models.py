"""Typed contracts shared by pipeline stages and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StageStatus(StrEnum):
    """Execution status of an optional AI stage."""

    SKIPPED = "skipped"
    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"


class AlertDeliveryStatus(StrEnum):
    """Durable outbox state for an alert."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class PipelineRunStatus(StrEnum):
    """Lifecycle state of durable message/watcher processing."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class Intent(StrEnum):
    """Message intent returned by the relevance classifier."""

    OFFER = "offer"
    SEEK = "seek"
    OTHER = "other"


class ObservationMode(StrEnum):
    """How the daemon treats incoming messages from an observed chat.

    ``RECON`` exists so that a chat discovered by a crawl is captured from the
    moment it is joined. Without it, everything sent between joining a chat and
    promoting it to monitoring is lost, because the live handler only persists
    chats that already carry a policy.
    """

    MONITOR = "monitor"
    RECON = "recon"
    PAUSED = "paused"


class ObservationSource(StrEnum):
    """What put a chat or binding into the registry."""

    CONFIG = "config"
    RECON = "recon"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class ObservedChat:
    """Registry entry describing one observed chat and its policies."""

    chat_id: int
    mode: ObservationMode
    source: ObservationSource
    watcher_names: tuple[str, ...] = ()
    title: str | None = None
    job_id: str | None = None


class ModelClassification(BaseModel):
    """Strict schema produced by an LLM provider."""

    model_config = ConfigDict(extra="forbid")

    relevant: bool
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)
    evidence: str = Field(min_length=1, max_length=300)


@dataclass(frozen=True, slots=True)
class ClassificationDecision:
    """Classification plus execution provenance."""

    result: ModelClassification
    status: StageStatus
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingDecision:
    """Semantic-stage decision with explainability metadata."""

    passed: bool
    status: StageStatus
    score: float | None = None
    negative_score: float | None = None
    threshold: float | None = None
    negative_margin: float | None = None
    matched_reference: str | None = None
    matched_negative_reference: str | None = None
    reason: str = ""
    model: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """Expected outcome of one dispatcher delivery cycle.

    ``error_code`` is deliberately machine-readable: response bodies and
    exception messages may contain message content or provider details and
    must not be persisted in the outbox.
    """

    sent: bool
    retryable: bool
    error_code: str | None = None
    retry_after: int | None = None

    def __post_init__(self) -> None:
        if self.sent and (
            self.retryable or self.error_code is not None or self.retry_after is not None
        ):
            raise ValueError("a successful delivery cannot carry retry metadata")
        if not self.sent and self.error_code is None:
            raise ValueError("a failed delivery requires an error_code")
        if self.retry_after is not None and self.retry_after < 1:
            raise ValueError("retry_after must be at least one second")
        if self.retry_after is not None and not self.retryable:
            raise ValueError("retry_after requires a retryable delivery")

    @classmethod
    def success(cls) -> DeliveryResult:
        """Build a successful delivery result."""
        return cls(sent=True, retryable=False)


@dataclass(frozen=True, slots=True)
class AlertOutboxItem:
    """Claimed alert payload ready for delivery."""

    alert_id: int
    watcher_name: str
    message_id: int
    chat_title: str
    sender_name: str
    text: str
    matched_keyword: str | None
    filter_level: int
    delivery_attempts: int
    claim_token: str


@dataclass(frozen=True, slots=True)
class AlertDraft:
    """Immutable alert payload committed with its terminal pipeline outcome."""

    filter_level: int
    score: float | None = None
    llm_response: str | None = None
    matched_keyword: str | None = None


@dataclass(frozen=True, slots=True)
class StoredPipelineJob:
    """Pending pipeline work reconstructed from durable message state."""

    message_id: int
    watcher_name: str
    chat_id: int
    chat_title: str
    sender_name: str
    text: str
    watcher_config_fingerprint: str


@dataclass(slots=True)
class PipelineOutcome:
    """One durable, idempotent outcome for a message/watcher pair."""

    message_id: int
    watcher_name: str
    processing_status: PipelineRunStatus = PipelineRunStatus.COMPLETED
    rule_passed: bool = False
    embedding_status: StageStatus = StageStatus.SKIPPED
    embedding_passed: bool = False
    embedding_score: float | None = None
    embedding_negative_score: float | None = None
    embedding_threshold: float | None = None
    embedding_model: str | None = None
    embedding_latency_ms: float | None = None
    embedding_input_tokens: int = 0
    llm_status: StageStatus = StageStatus.SKIPPED
    llm_relevant: bool = False
    llm_passed: bool = False
    llm_verdict: str | None = None
    llm_confidence: float | None = None
    llm_model: str | None = None
    llm_prompt_version: str | None = None
    llm_latency_ms: float | None = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    accepted: bool = False
    alert_created: bool = False
    alert_sent: bool = False
    error_code: str | None = None
