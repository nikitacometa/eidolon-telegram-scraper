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


class Intent(StrEnum):
    """Message intent returned by the relevance classifier."""

    OFFER = "offer"
    SEEK = "seek"
    OTHER = "other"


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
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingDecision:
    """Semantic-stage decision with explainability metadata."""

    passed: bool
    status: StageStatus
    score: float | None = None
    matched_reference: str | None = None
    reason: str = ""
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
    filter_level: int
    delivery_attempts: int


@dataclass(slots=True)
class PipelineOutcome:
    """One durable, idempotent outcome for a message/watcher pair."""

    message_id: int
    watcher_name: str
    rule_passed: bool = False
    embedding_status: StageStatus = StageStatus.SKIPPED
    embedding_passed: bool = False
    embedding_score: float | None = None
    embedding_model: str | None = None
    embedding_latency_ms: float | None = None
    llm_status: StageStatus = StageStatus.SKIPPED
    llm_relevant: bool = False
    llm_verdict: str | None = None
    llm_confidence: float | None = None
    llm_model: str | None = None
    llm_prompt_version: str | None = None
    llm_latency_ms: float | None = None
    alert_created: bool = False
    alert_sent: bool = False
    error_code: str | None = None

    @property
    def accepted(self) -> bool:
        """Whether this watcher accepted the message for alerting."""
        return self.alert_created
