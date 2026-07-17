"""Integration tests for the typed message processor."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.watchers import Watcher, WatcherRules
from pipeline.filters import RuleFilter
from pipeline.models import (
    ClassificationDecision,
    EmbeddingDecision,
    Intent,
    ModelClassification,
    StageStatus,
)
from pipeline.policy import effective_policy_fingerprint
from pipeline.processor import MessageProcessor
from storage.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "processor.db")
    await database.connect()
    yield database
    await database.close()


def _watcher(*, alert: str = "immediate") -> Watcher:
    return Watcher(
        name="housing-watch",
        chats=[-100123],
        description="Island housing offers",
        rules=WatcherRules(keywords=["villa", "rent"], min_length=10),
        alert=alert,
        llm_level=3,
        prompt="Accept only genuine residential rental offers.",
    )


async def _message(db: Database, watcher: Watcher, telegram_id: int = 1) -> int:
    message_id = await db.store_message(
        telegram_msg_id=telegram_id,
        chat_id=-100123,
        chat_title="Housing Chat",
        sender_id=7,
        sender_name="Alice",
        text="Villa for rent near the beach",
        date="2026-07-17 12:00:00",
        watcher_names=(watcher.name,),
        watcher_fingerprints={watcher.name: effective_policy_fingerprint(watcher)},
    )
    assert message_id is not None
    return message_id


def _processor(
    db: Database,
    watcher: Watcher,
    *,
    relevant: bool = True,
) -> tuple[MessageProcessor, MagicMock, MagicMock]:
    embedding = MagicMock()
    embedding.check = AsyncMock(
        return_value=EmbeddingDecision(
            passed=True,
            status=StageStatus.OK,
            score=0.91,
            matched_reference="Villa available for monthly rent",
            reason="semantic match",
            model="embedding-test",
            latency_ms=8.5,
            input_tokens=7,
        )
    )
    classifier = MagicMock()
    classifier.classify = AsyncMock(
        return_value=ClassificationDecision(
            result=ModelClassification(
                relevant=relevant,
                intent=Intent.OFFER if relevant else Intent.OTHER,
                confidence=0.96,
                reason="matches watcher objective" if relevant else "not an offer",
                evidence="Villa for rent",
            ),
            status=StageStatus.OK,
            model="llm-test",
            latency_ms=32.0,
            input_tokens=25,
            output_tokens=9,
        )
    )
    return (
        MessageProcessor(
            store=db,
            rule_filters={watcher.name: RuleFilter(watcher)},
            embedding_filter=embedding,
            llm_classifier=classifier,
        ),
        embedding,
        classifier,
    )


async def test_processor_persists_explainable_alert_and_provenance(
    db: Database,
) -> None:
    watcher = _watcher()
    message_id = await _message(db, watcher)
    processor, embedding, classifier = _processor(db, watcher)

    outcome = await processor.process(
        watcher=watcher,
        message_id=message_id,
        text="Villa for rent near the beach",
    )

    assert outcome.accepted is True
    assert outcome.alert_created is True
    assert outcome.embedding_model == "embedding-test"
    assert outcome.embedding_input_tokens == 7
    assert outcome.llm_model == "llm-test"
    assert outcome.llm_input_tokens == 25
    assert outcome.llm_output_tokens == 9
    embedding.check.assert_awaited_once()
    classifier.classify.assert_awaited_once_with(
        text="Villa for rent near the beach",
        watcher_prompt=(
            "Island housing offers\n\n"
            "Accept only genuine residential rental offers.\n\n"
            "Accepted message intents: offer."
        ),
    )

    cursor = await db.conn.execute(
        """
        SELECT filter_level, matched_keyword, delivery_status
        FROM alerts
        WHERE watcher_name = ? AND message_id = ?
        """,
        (watcher.name, message_id),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == (3, "villa", "pending")


async def test_processor_records_llm_rejection_without_alert(db: Database) -> None:
    watcher = _watcher()
    message_id = await _message(db, watcher, telegram_id=2)
    processor, _, _ = _processor(db, watcher, relevant=False)

    outcome = await processor.process(
        watcher=watcher,
        message_id=message_id,
        text="Villa for rent near the beach",
    )

    assert outcome.accepted is False
    assert outcome.alert_created is False
    cursor = await db.conn.execute("SELECT COUNT(*) FROM alerts")
    assert (await cursor.fetchone())[0] == 0


async def test_digest_policy_accepts_without_creating_immediate_outbox_row(
    db: Database,
) -> None:
    watcher = _watcher(alert="digest")
    message_id = await _message(db, watcher, telegram_id=3)
    processor, _, _ = _processor(db, watcher)

    outcome = await processor.process(
        watcher=watcher,
        message_id=message_id,
        text="Villa for rent near the beach",
    )

    assert outcome.accepted is True
    assert outcome.alert_created is False
    cursor = await db.conn.execute("SELECT COUNT(*) FROM alerts")
    assert (await cursor.fetchone())[0] == 0
    digest_messages = await db.get_daily_accepted_messages(
        watcher.name,
        "2026-07-17",
    )
    assert [message["text"] for message in digest_messages] == ["Villa for rent near the beach"]


async def test_cancellation_leaves_pipeline_job_pending(db: Database) -> None:
    watcher = _watcher()
    message_id = await _message(db, watcher, telegram_id=4)
    processor, embedding, _ = _processor(db, watcher)
    embedding.check.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await processor.process(
            watcher=watcher,
            message_id=message_id,
            text="Villa for rent near the beach",
        )

    jobs = await db.get_pending_pipeline_jobs()
    assert [(job.message_id, job.watcher_name) for job in jobs] == [(message_id, watcher.name)]
    cursor = await db.conn.execute("SELECT COUNT(*) FROM alerts")
    assert (await cursor.fetchone())[0] == 0


async def test_relevant_but_disallowed_intent_is_rejected(db: Database) -> None:
    watcher = _watcher()
    message_id = await _message(db, watcher, telegram_id=5)
    processor, _, classifier = _processor(db, watcher)
    classifier.classify.return_value = ClassificationDecision(
        result=ModelClassification(
            relevant=True,
            intent=Intent.SEEK,
            confidence=0.93,
            reason="The author wants a property",
            evidence="Villa for rent",
        ),
        status=StageStatus.OK,
        model="llm-test",
        latency_ms=12,
    )

    outcome = await processor.process(
        watcher=watcher,
        message_id=message_id,
        text="Villa for rent near the beach",
    )

    assert outcome.llm_relevant is True
    assert outcome.llm_passed is False
    assert outcome.accepted is False
    cursor = await db.conn.execute("SELECT COUNT(*) FROM alerts")
    assert (await cursor.fetchone())[0] == 0


async def test_unexpected_stage_failure_is_terminal_and_sanitized(db: Database) -> None:
    watcher = _watcher()
    message_id = await _message(db, watcher, telegram_id=6)
    processor, embedding, _ = _processor(db, watcher)
    embedding.check.side_effect = RuntimeError("private provider details")

    with pytest.raises(RuntimeError, match="private provider details"):
        await processor.process(
            watcher=watcher,
            message_id=message_id,
            text="Villa for rent near the beach",
        )

    assert await db.get_pending_pipeline_jobs() == []
    cursor = await db.conn.execute(
        """
        SELECT processing_status, error_code
        FROM pipeline_runs
        WHERE message_id = ? AND watcher_name = ?
        """,
        (message_id, watcher.name),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert tuple(row) == ("failed", "pipeline_runtime_error")


async def test_degraded_llm_is_rejected_by_safe_default(db: Database) -> None:
    watcher = _watcher()
    message_id = await _message(db, watcher, telegram_id=7)
    processor, _, classifier = _processor(db, watcher)
    classifier.classify.return_value = ClassificationDecision(
        result=ModelClassification(
            relevant=True,
            intent=Intent.OFFER,
            confidence=0,
            reason="provider unavailable",
            evidence="Rules-only fallback.",
        ),
        status=StageStatus.DEGRADED,
        model="llm-test",
        latency_ms=12,
        error_code="timeout",
    )

    outcome = await processor.process(
        watcher=watcher,
        message_id=message_id,
        text="Villa for rent near the beach",
    )

    assert outcome.accepted is False
    assert outcome.llm_passed is False
    cursor = await db.conn.execute("SELECT COUNT(*) FROM alerts")
    assert (await cursor.fetchone())[0] == 0
