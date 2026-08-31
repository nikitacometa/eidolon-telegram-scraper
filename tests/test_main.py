"""Orchestration tests for durable recovery and alert delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

import main as main_module
from config.settings import settings
from config.watchers import Watcher, WatcherRules
from main import Eidolon, RuntimeConfigurationError, _validate_runtime_configuration
from pipeline.models import (
    AlertDeliveryStatus,
    AlertOutboxItem,
    DeliveryResult,
    MediaPointer,
    ObservationMode,
    ObservationSource,
    ObservedChat,
    PipelineRunStatus,
    StoredPipelineJob,
)
from pipeline.policy import effective_policy_fingerprint


def _outbox_item() -> AlertOutboxItem:
    return AlertOutboxItem(
        alert_id=7,
        watcher_name="housing-watch",
        message_id=11,
        chat_title="Housing Chat",
        sender_name="Alice",
        text="Villa for rent",
        matched_keyword="villa",
        filter_level=2,
        delivery_attempts=1,
        claim_token="claim-token",
    )


def test_digest_watcher_cannot_start_with_scheduler_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watcher = Watcher(
        name="digest-watch",
        chats=[-100123],
        rules=WatcherRules(keywords=["villa"]),
        alert="digest",
    )
    monkeypatch.setattr(settings, "summary_enabled", False)
    monkeypatch.setattr(settings, "telegram_api_id", 1)
    monkeypatch.setattr(settings, "telegram_api_hash", "hash")
    monkeypatch.setattr(settings, "telegram_session_string", "session")

    with pytest.raises(RuntimeConfigurationError, match="SUMMARY_ENABLED=false"):
        _validate_runtime_configuration([watcher])


def test_enabled_summaries_require_model_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watcher = Watcher(
        name="housing-watch",
        chats=[-100123],
        rules=WatcherRules(keywords=["villa"]),
    )
    monkeypatch.setattr(settings, "summary_enabled", True)
    monkeypatch.setattr(settings, "telegram_api_id", 1)
    monkeypatch.setattr(settings, "telegram_api_hash", "hash")
    monkeypatch.setattr(settings, "telegram_session_string", "session")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "eidolon_bot_token", "bot")
    monkeypatch.setattr(settings, "pantheon_chat_id", 1)

    with pytest.raises(RuntimeConfigurationError, match="OPENAI_API_KEY"):
        _validate_runtime_configuration([watcher])


def test_ai_watcher_requires_model_credentials_when_summaries_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watcher = Watcher(
        name="housing-watch",
        chats=[-100123],
        rules=WatcherRules(keywords=["villa"]),
        llm_level=2,
    )
    monkeypatch.setattr(settings, "summary_enabled", False)
    monkeypatch.setattr(settings, "telegram_api_id", 1)
    monkeypatch.setattr(settings, "telegram_api_hash", "hash")
    monkeypatch.setattr(settings, "telegram_session_string", "session")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "eidolon_bot_token", "bot")
    monkeypatch.setattr(settings, "pantheon_chat_id", 1)

    with pytest.raises(RuntimeConfigurationError, match="OPENAI_API_KEY"):
        _validate_runtime_configuration([watcher])


async def test_ingress_retries_transient_sqlite_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watcher = Watcher(
        name="housing-watch",
        chats=[-100123],
        rules=WatcherRules(keywords=["villa"]),
    )
    app = Eidolon.__new__(Eidolon)
    app.chat_watchers = {-100123: [watcher]}
    app.observed_chats = {
        -100123: ObservedChat(
            chat_id=-100123,
            mode=ObservationMode.MONITOR,
            source=ObservationSource.CONFIG,
            watcher_names=(watcher.name,),
        )
    }
    app.watcher_fingerprints = {watcher.name: effective_policy_fingerprint(watcher)}
    app.db = MagicMock()
    event = SimpleNamespace(
        chat_id=-100123,
        text="Villa for rent",
        # A real NewMessage always carries the message it delivered, and
        # ingestion now reads its media pointers off it.
        message=SimpleNamespace(id=7, grouped_id=None, photo=None, document=None),
    )
    ingest = AsyncMock(side_effect=[aiosqlite.OperationalError("locked"), 42])
    sleep = AsyncMock()
    monkeypatch.setattr(main_module, "ingest_message", ingest)
    monkeypatch.setattr(main_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(settings, "ingress_max_attempts", 2)
    monkeypatch.setattr(settings, "ingress_retry_base_seconds", 0.05)
    monkeypatch.setattr(settings, "debug_echo", False)

    item = await app._ingest_update(event)

    assert item is not None
    assert item.message_id == 42
    assert ingest.await_count == 2
    sleep.assert_awaited_once_with(0.05)


async def test_deliver_due_alerts_persists_atomic_success() -> None:
    app = Eidolon.__new__(Eidolon)
    app.db = MagicMock()
    app.db.claim_due_alerts = AsyncMock(side_effect=[[_outbox_item()], []])
    app.db.mark_alert_delivery_result = AsyncMock(return_value=AlertDeliveryStatus.SENT)
    app.dispatcher = MagicMock()
    app.dispatcher.deliver_alert = AsyncMock(return_value=DeliveryResult.success())

    delivered = await app._deliver_due_alerts()

    assert delivered == 1
    app.dispatcher.deliver_alert.assert_awaited_once_with(
        watcher_name="housing-watch",
        chat_title="Housing Chat",
        sender_name="Alice",
        text="Villa for rent",
        matched_keyword="villa",
        filter_level=2,
    )
    app.db.mark_alert_delivery_result.assert_awaited_once_with(
        7,
        DeliveryResult.success(),
        claim_token="claim-token",
        max_attempts=5,
    )


async def test_retryable_outbox_result_is_not_counted_as_sent() -> None:
    app = Eidolon.__new__(Eidolon)
    app.db = MagicMock()
    app.db.claim_due_alerts = AsyncMock(side_effect=[[_outbox_item()], []])
    app.db.mark_alert_delivery_result = AsyncMock(return_value=AlertDeliveryStatus.PENDING)
    app.dispatcher = MagicMock()
    app.dispatcher.deliver_alert = AsyncMock(
        return_value=DeliveryResult(
            sent=False,
            retryable=True,
            error_code="telegram_timeout",
            retry_after=2,
        )
    )

    assert await app._deliver_due_alerts() == 1
    app.db.mark_alert_delivery_result.assert_awaited_once()


async def test_pending_pipeline_job_is_replayed_from_stored_message() -> None:
    watcher = Watcher(
        name="housing-watch",
        chats=[-100123],
        rules=WatcherRules(keywords=["villa"]),
    )
    job = StoredPipelineJob(
        message_id=11,
        watcher_name=watcher.name,
        chat_id=-100123,
        chat_title="Housing Chat",
        sender_name="Alice",
        text="Villa for rent",
        watcher_config_fingerprint=effective_policy_fingerprint(watcher),
        telegram_msg_id=555,
        media=MediaPointer(has_media=True, telegram_photo_id=4242, grouped_id=909),
    )
    app = Eidolon.__new__(Eidolon)
    app.watchers_by_name = {watcher.name: watcher}
    app.watcher_fingerprints = {watcher.name: effective_policy_fingerprint(watcher)}
    app.db = MagicMock()
    app.db.get_pending_pipeline_jobs = AsyncMock(side_effect=[[job], [], []])
    app.db.record_pipeline_outcome = AsyncMock()
    app._process_watcher = AsyncMock()

    await app._recover_pending_pipeline_jobs()

    app._process_watcher.assert_awaited_once()
    call = app._process_watcher.await_args
    assert call.kwargs["watcher"] == watcher
    replayed = call.kwargs["work_item"]
    assert replayed.message_id == 11
    assert replayed.text == "Villa for rent"
    # The rebuilt item carries the identity a message-grouping subsystem needs,
    # so a restart mid-album resumes instead of losing the advertisement.
    assert replayed.chat_id == -100123
    assert replayed.telegram_msg_id == 555
    assert replayed.media.grouped_id == 909


async def test_recovery_rejects_job_from_changed_watcher_policy() -> None:
    watcher = Watcher(
        name="housing-watch",
        chats=[-100123],
        rules=WatcherRules(keywords=["villa"]),
    )
    job = StoredPipelineJob(
        message_id=12,
        watcher_name=watcher.name,
        chat_id=-100123,
        chat_title="Housing Chat",
        sender_name="Alice",
        text="Villa for rent",
        watcher_config_fingerprint="old-policy",
    )
    app = Eidolon.__new__(Eidolon)
    app.watchers_by_name = {watcher.name: watcher}
    app.watcher_fingerprints = {watcher.name: effective_policy_fingerprint(watcher)}
    app.db = MagicMock()
    app.db.get_pending_pipeline_jobs = AsyncMock(side_effect=[[job], []])
    app.db.record_pipeline_outcome = AsyncMock(return_value=True)
    app._process_watcher = AsyncMock()

    await app._recover_pending_pipeline_jobs()

    outcome = app.db.record_pipeline_outcome.await_args.args[0]
    assert outcome.processing_status is PipelineRunStatus.FAILED
    assert outcome.error_code == "watcher_policy_changed"
    app._process_watcher.assert_not_awaited()


async def test_summary_failure_is_isolated_per_watcher() -> None:
    broken = Watcher(
        name="broken-watch",
        chats=[-100123],
        rules=WatcherRules(keywords=["villa"]),
    )
    healthy = Watcher(
        name="healthy-watch",
        chats=[-100124],
        rules=WatcherRules(keywords=["house"]),
    )
    app = Eidolon.__new__(Eidolon)
    app.watchers = [broken, healthy]
    app.db = MagicMock()
    app.db.get_accepted_messages_between = AsyncMock(
        side_effect=[
            RuntimeError("broken query"),
            [{"message_id": 1, "text": "House available"}],
        ]
    )
    app.summarizer = MagicMock()
    app.summarizer.summarize = AsyncMock(return_value="• Housing: House [m:1]")
    app.dispatcher = MagicMock()
    app.dispatcher.send_summary = AsyncMock(return_value=True)

    await app._generate_summaries()

    app.summarizer.summarize.assert_awaited_once()
    assert app.summarizer.summarize.await_args.kwargs["watcher_name"] == "healthy-watch"
    app.dispatcher.send_summary.assert_awaited_once()


async def test_forward_from_a_person_is_captured_not_fatal() -> None:
    """A person's forward carries channel_post=None on the header itself.

    getattr's default never applies to an attribute that exists with value
    None, and int(None) raised straight through the ingress path — measured
    on prod 2026-08-31: 25 daemon deaths in two days, all this line.
    """
    app = Eidolon.__new__(Eidolon)
    app.scout = MagicMock()
    app.scout.store_message = AsyncMock()
    event = SimpleNamespace(
        text="переслал объявление",
        message=SimpleNamespace(
            id=11,
            date="2026-08-31 09:00:00+00:00",
            sender_id=42,
            fwd_from=SimpleNamespace(from_id=SimpleNamespace(user_id=99), channel_post=None),
            reply_to=None,
        ),
    )

    await app._capture_scout_message(event, -100555)

    stored = app.scout.store_message.await_args.args[0]
    assert stored.forward_chat_id is None
    assert stored.forward_message_id is None
    assert stored.telegram_msg_id == 11


async def test_channel_forward_keeps_its_source_ids() -> None:
    """The channel case still records where the forward came from."""
    app = Eidolon.__new__(Eidolon)
    app.scout = MagicMock()
    app.scout.store_message = AsyncMock()
    event = SimpleNamespace(
        text="repost",
        message=SimpleNamespace(
            id=12,
            date="2026-08-31 09:00:00+00:00",
            sender_id=42,
            fwd_from=SimpleNamespace(from_id=SimpleNamespace(channel_id=777), channel_post=345),
            reply_to=None,
        ),
    )

    await app._capture_scout_message(event, -100555)

    stored = app.scout.store_message.await_args.args[0]
    assert stored.forward_chat_id == 777
    assert stored.forward_message_id == 345
