"""Typed orchestration for one message/watcher pipeline job."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Protocol

from config.watchers import Watcher
from pipeline.embeddings import EmbeddingFilter
from pipeline.filters import RuleFilter
from pipeline.llm import (
    PROMPT_VERSION,
    LLMClassifier,
    build_watcher_objective,
    classification_passes,
    decision_verdict,
)
from pipeline.models import AlertDraft, PipelineOutcome, PipelineRunStatus, StageStatus

logger = logging.getLogger(__name__)


class PipelineStore(Protocol):
    """Persistence operations required by the processor."""

    async def record_pipeline_outcome(
        self,
        outcome: PipelineOutcome,
        *,
        alert: AlertDraft | None = None,
    ) -> bool: ...


class MessageProcessor:
    """Execute deterministic and AI stages with one durable terminal outcome."""

    def __init__(
        self,
        *,
        store: PipelineStore,
        rule_filters: Mapping[str, RuleFilter],
        embedding_filter: EmbeddingFilter,
        llm_classifier: LLMClassifier,
    ) -> None:
        self._store = store
        self._rule_filters = rule_filters
        self._embedding_filter = embedding_filter
        self._llm_classifier = llm_classifier

    async def process(
        self,
        *,
        watcher: Watcher,
        message_id: int,
        text: str | None,
    ) -> PipelineOutcome:
        """Run one policy and persist its idempotent outcome."""
        outcome = PipelineOutcome(message_id=message_id, watcher_name=watcher.name)
        filter_level = 1
        llm_response: str | None = None
        matched_keyword: str | None = None
        alert: AlertDraft | None = None
        persist_terminal = True

        try:
            rule_result = self._rule_filters[watcher.name].check(text)
            if not rule_result:
                return outcome
            outcome.rule_passed = True
            matched_keyword = rule_result.matched_keyword

            if watcher.llm_level >= 2 and text:
                embedding = await self._embedding_filter.check(text, watcher.name)
                outcome.embedding_status = embedding.status
                outcome.embedding_passed = embedding.passed
                outcome.embedding_score = embedding.score
                outcome.embedding_negative_score = embedding.negative_score
                outcome.embedding_threshold = embedding.threshold
                outcome.embedding_model = embedding.model
                outcome.embedding_latency_ms = embedding.latency_ms
                outcome.embedding_input_tokens = embedding.input_tokens
                if embedding.error_code:
                    outcome.error_code = embedding.error_code
                if embedding.status is not StageStatus.OK and watcher.degraded_policy == "reject":
                    logger.warning(
                        "Embedding degraded and watcher rejected fallback: "
                        "watcher=%s message_id=%d code=%s",
                        watcher.name,
                        message_id,
                        embedding.error_code,
                    )
                    return outcome
                if not embedding.passed:
                    logger.info(
                        "Embedding filtered message: watcher=%s message_id=%d score=%s",
                        watcher.name,
                        message_id,
                        embedding.score,
                    )
                    return outcome
                if embedding.status is StageStatus.OK:
                    filter_level = 2

            if watcher.llm_level >= 3 and text:
                objective = build_watcher_objective(
                    description=watcher.description,
                    prompt=watcher.prompt,
                    target_intents=watcher.target_intents,
                )
                classification = await self._llm_classifier.classify(
                    text=text,
                    watcher_prompt=objective,
                )
                verdict = decision_verdict(classification)
                outcome.llm_status = classification.status
                outcome.llm_relevant = classification.result.relevant
                outcome.llm_passed = classification_passes(
                    classification,
                    watcher.target_intents,
                    watcher.degraded_policy,
                )
                outcome.llm_verdict = verdict.value
                outcome.llm_confidence = classification.result.confidence
                outcome.llm_model = classification.model
                outcome.llm_prompt_version = PROMPT_VERSION
                outcome.llm_latency_ms = classification.latency_ms
                outcome.llm_input_tokens = classification.input_tokens
                outcome.llm_output_tokens = classification.output_tokens
                if classification.error_code:
                    outcome.error_code = classification.error_code
                llm_response = classification.result.model_dump_json()
                if classification.status is StageStatus.OK:
                    filter_level = 3
                if not outcome.llm_passed:
                    logger.info(
                        "LLM policy filtered message: watcher=%s message_id=%d "
                        "intent=%s confidence=%.3f",
                        watcher.name,
                        message_id,
                        classification.result.intent.value,
                        classification.result.confidence,
                    )
                    return outcome

            outcome.accepted = True
            if watcher.alert == "immediate":
                alert = AlertDraft(
                    filter_level=filter_level,
                    score=outcome.embedding_score,
                    llm_response=llm_response,
                    matched_keyword=matched_keyword,
                )
                outcome.alert_created = True
            return outcome
        except asyncio.CancelledError:
            # Cancellation is not a business outcome. The durable pending row
            # is replayed after restart instead of recording partial success.
            persist_terminal = False
            raise
        except Exception as error:
            # Provider failures are modeled as explicit DEGRADED decisions and
            # do not reach this path. Mark unexpected faults terminal so one
            # poison row cannot block every subsequent restart.
            outcome.processing_status = PipelineRunStatus.FAILED
            outcome.error_code = _safe_exception_code(error)
            logger.error(
                "Pipeline job failed after %s: watcher=%s message_id=%d",
                type(error).__name__,
                watcher.name,
                message_id,
            )
            raise
        finally:
            if persist_terminal:
                await self._store.record_pipeline_outcome(outcome, alert=alert)


def _safe_exception_code(error: Exception) -> str:
    """Return a bounded class-only error code without provider or message text."""
    name = type(error).__name__
    normalized = "".join(
        f"_{character.lower()}" if character.isupper() else character for character in name
    ).lstrip("_")
    return f"pipeline_{normalized}"[:64]
