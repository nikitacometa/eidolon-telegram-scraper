"""Eidolon — AI-powered Telegram monitor.

Entry point: connects to Telegram via MTProto, listens for messages
in configured chats, runs them through the filter pipeline, and
dispatches alerts.
"""

import asyncio
import logging
import signal
import sys
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import aiosqlite
from telethon import TelegramClient, events
from telethon.sessions import StringSession

from config.settings import settings
from config.watchers import (
    Watcher,
    WatcherConfigError,
    load_watchers,
)
from pipeline.agent_watchers import (
    AgentWatcherSync,
    ReconcileResult,
    merge_agent_watchers,
)
from pipeline.backfill import BackfillWorker
from pipeline.crawler import TelegramCrawler
from pipeline.discovery import TelegramDiscovery
from pipeline.discovery_worker import DiscoveryWorker
from pipeline.dispatcher import AlertDispatcher
from pipeline.embeddings import EmbeddingFilter
from pipeline.filters import RuleFilter
from pipeline.governor import TelegramActionGovernor
from pipeline.housing.worker import HousingAlertDelivery, HousingWorker
from pipeline.ingestion import (
    NewMessageEvent,
    ingest_message,
    media_pointer,
    reply_to_message_id,
)
from pipeline.joiner import JoinWorker
from pipeline.llm import LLMClassifier
from pipeline.models import (
    MediaPointer,
    ObservationMode,
    ObservedChat,
    PipelineOutcome,
    PipelineRunStatus,
)
from pipeline.policy import effective_policy_fingerprint
from pipeline.processor import MessageProcessor
from pipeline.recon import ReconRunner
from pipeline.recon_models import ScoutMessage
from pipeline.summarizer import DailySummarizer
from storage.db import Database
from storage.housing import HousingStore, unit_key_for
from storage.scout import ScoutDatabase
from storage.session_lock import SessionLock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eidolon")


# The one policy whose messages go to the housing subsystem rather than
# through the relevance ladder. Named here so the dispatch check is a
# comparison against a constant rather than a string spelled twice.
HOUSING_WATCHER_NAME = "phangan-housing"


class RuntimeConfigurationError(ValueError):
    """Raised when required daemon credentials or destinations are absent."""


@dataclass(frozen=True, slots=True)
class PipelineWorkItem:
    """Durably ingested message ready for in-memory execution."""

    message_id: int
    text: str | None
    watcher_names: tuple[str, ...]
    # Identity and media pointers, carried for subsystems that work on
    # advertisements rather than on messages. Populated always; every
    # relevance-ladder watcher ignores them.
    chat_id: int = 0
    telegram_msg_id: int = 0
    media: MediaPointer = field(default_factory=MediaPointer.unscanned)


def _validate_runtime_configuration(watchers: list[Watcher]) -> None:
    missing: list[str] = []
    if not settings.telegram_api_id:
        missing.append("TELEGRAM_API_ID")
    if not settings.telegram_api_hash:
        missing.append("TELEGRAM_API_HASH")
    if not settings.telegram_session_string:
        missing.append("TELEGRAM_SESSION_STRING")

    digest_configured = any(watcher.alert == "digest" for watcher in watchers)
    if digest_configured and not settings.summary_enabled:
        raise RuntimeConfigurationError("digest watcher configured while SUMMARY_ENABLED=false")
    ai_required = settings.summary_enabled or any(watcher.llm_level >= 2 for watcher in watchers)
    if ai_required and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")

    delivery_required = settings.summary_enabled or any(
        watcher.alert == "immediate" for watcher in watchers
    )
    if delivery_required and not (settings.eidolon_bot_token or settings.pantheon_bot_token):
        missing.append("EIDOLON_BOT_TOKEN or PANTHEON_BOT_TOKEN")
    if delivery_required and not settings.pantheon_chat_id:
        missing.append("PANTHEON_CHAT_ID")
    if missing:
        raise RuntimeConfigurationError(
            "missing required runtime configuration: " + ", ".join(missing)
        )


class Eidolon:
    """Main application: Telethon client + event loop + pipeline."""

    def __init__(self) -> None:
        self.db = Database(settings.db_path)
        self.scout = ScoutDatabase(settings.scout_db_path)
        self.client = TelegramClient(
            StringSession(settings.telegram_session_string),
            settings.telegram_api_id,
            settings.telegram_api_hash,
            sequential_updates=True,
        )
        self.dispatcher = AlertDispatcher()
        # Built in start(), once the database connection it borrows exists.
        self.housing: HousingStore | None = None
        self._housing_task: asyncio.Task[None] | None = None
        self._housing_delivery_task: asyncio.Task[None] | None = None
        self.governor = TelegramActionGovernor(scout=self.scout)
        self.crawler = TelegramCrawler(client=self.client, governor=self.governor)
        self.backfill = BackfillWorker(
            scout=self.scout,
            crawler=self.crawler,
            resolve_peer=self._input_peer,
            is_idle=lambda: self._message_queue.empty(),
        )
        self.joiner = JoinWorker(
            scout=self.scout,
            db=self.db,
            crawler=self.crawler,
            resolve_entity=self._named_entity,
            on_joined=self.reload_observation,
        )
        self.discovery = DiscoveryWorker(
            scout=self.scout,
            runner=ReconRunner(
                scout=self.scout,
                db=self.db,
                client=self.client,
                governor=self.governor,
                discovery=TelegramDiscovery(client=self.client, governor=self.governor),
                crawler=self.crawler,
            ),
        )
        self._backfill_task: asyncio.Task[None] | None = None
        self._agent_watcher_task: asyncio.Task[None] | None = None
        self._joiner_task: asyncio.Task[None] | None = None
        self._discovery_task: asyncio.Task[None] | None = None
        self.embedding_filter = EmbeddingFilter()
        self.llm_classifier = LLMClassifier()
        self.summarizer = DailySummarizer()
        self._shutdown_event = asyncio.Event()
        self._message_queue: asyncio.Queue[PipelineWorkItem] = asyncio.Queue(
            maxsize=settings.processing_queue_size
        )
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._summary_task: asyncio.Task[None] | None = None
        self._outbox_task: asyncio.Task[None] | None = None
        self._maintenance_task: asyncio.Task[None] | None = None
        self._outbox_wakeup = asyncio.Event()
        self._fatal_ingress_error: BaseException | None = None

        # Load watcher policies. Which chats they apply to comes from the
        # registry once the database is open, not from the file.
        # The git-tracked policies, kept separate from the merged view so a
        # reconciliation always rebuilds from the reviewed set rather than from
        # whatever the previous merge produced.
        self._config_watchers = load_watchers(settings.watchers_path)
        self._agent_watchers: dict[str, Watcher] = {}
        self.watchers = list(self._config_watchers)
        _validate_runtime_configuration(self.watchers)
        self.watchers_by_name = {watcher.name: watcher for watcher in self.watchers}
        self.watcher_fingerprints = {
            watcher.name: effective_policy_fingerprint(watcher) for watcher in self.watchers
        }
        self.chat_watchers: dict[int, list[Watcher]] = {}
        self.observed_chats: dict[int, ObservedChat] = {}
        self.filters: dict[str, RuleFilter] = {w.name: RuleFilter(w) for w in self.watchers}
        self.processor = MessageProcessor(
            store=self.db,
            rule_filters=self.filters,
            embedding_filter=self.embedding_filter,
            llm_classifier=self.llm_classifier,
        )

    async def start(self) -> None:
        """Connect to Telegram and database, register handlers, run until shutdown."""
        self._setup_signals()
        self._register_handlers()

        try:
            async with AsyncExitStack() as stack:
                # Taken before the client exists: a second holder of this
                # session would invalidate the authorisation for both.
                session_lock = SessionLock(settings.db_path.parent / "session.lock")
                session_lock.acquire()
                stack.callback(session_lock.release)
                await self.db.connect()
                stack.push_async_callback(self.db.close)
                # Housing shares the live connection rather than opening a
                # second one: two connections to one SQLite file queue behind
                # the same file lock anyway, and the daemon already serialises
                # its writes through the lock this borrows.
                self.housing = HousingStore(self.db.conn, self.db.write_lock)
                await self.scout.connect()
                stack.push_async_callback(self.scout.close)
                await self.reconcile_agent_watchers(apply=False)
                await self.reload_observation()
                purged = await self.db.purge_expired_data(settings.retention_days)
                if purged:
                    logger.info("Purged %d messages beyond retention window", purged)

                await self.dispatcher.start()
                stack.push_async_callback(self.dispatcher.close)
                await self.embedding_filter.start(self.watchers)
                stack.push_async_callback(self.embedding_filter.close)
                await self.llm_classifier.start()
                stack.push_async_callback(self.llm_classifier.close)
                await self.summarizer.start()
                stack.push_async_callback(self.summarizer.close)
                await self._recover_pending_pipeline_jobs()
                await self.client.start()
                stack.push_async_callback(self.client.disconnect)

                me = await self.client.get_me()
                logger.info("Connected as account_id=%s", getattr(me, "id", "unknown"))
                logger.info(
                    "Monitoring %d chats via %d watchers",
                    len(self.chat_watchers),
                    len(self.watchers),
                )

                if settings.debug_echo:
                    logger.info("DEBUG ECHO MODE — forwarding ALL messages from monitored chats")

                self._start_background_tasks()
                try:
                    logger.info(
                        "Eidolon is listening: workers=%d queue_capacity=%d",
                        settings.processing_workers,
                        settings.processing_queue_size,
                    )
                    await self._shutdown_event.wait()
                    if self._fatal_ingress_error is not None:
                        raise RuntimeError(
                            "daemon stopped after durable ingress failed"
                        ) from self._fatal_ingress_error
                    logger.info("Shutdown requested")
                finally:
                    await self._stop_background_tasks()
        finally:
            self._remove_signals()

        logger.info("Goodbye.")

    def _start_background_tasks(self) -> None:
        self._worker_tasks = [
            asyncio.create_task(
                self._message_worker(worker_id),
                name=f"message-worker-{worker_id}",
            )
            for worker_id in range(settings.processing_workers)
        ]
        self._outbox_task = asyncio.create_task(
            self._run_outbox_worker(),
            name="alert-outbox",
        )
        self._maintenance_task = asyncio.create_task(
            self._run_retention_sweeper(),
            name="retention-sweeper",
        )
        if settings.join_queue_enabled:
            self._joiner_task = asyncio.create_task(
                self.joiner.run_forever(self._shutdown_event),
                name="join-queue",
            )
            logger.info("Join queue worker enabled")
        if settings.discovery_enabled:
            self._discovery_task = asyncio.create_task(
                self.discovery.run_forever(self._shutdown_event),
                name="discovery",
            )
            logger.info("Discovery worker enabled")
        if settings.agent_watchers_enabled:
            self._agent_watcher_task = asyncio.create_task(
                AgentWatcherSync(
                    read_generation=self.db.agent_watcher_generation,
                    reconcile=self.reconcile_agent_watchers,
                    poll_seconds=settings.agent_watcher_poll_seconds,
                ).run_forever(self._shutdown_event),
                name="agent-watcher-sync",
            )
            logger.info("Agent watcher sync enabled")

        if self.housing is not None and any(
            watcher.dispatch == "external" for watcher in self.watchers
        ):
            self._housing_task = asyncio.create_task(
                HousingWorker(store=self.housing).run_forever(self._shutdown_event),
                name="housing",
            )
            self._housing_delivery_task = asyncio.create_task(
                HousingAlertDelivery(
                    store=self.housing,
                    dispatcher=self.dispatcher,
                    lease_owner=f"housing-{id(self):x}",
                ).run_forever(self._shutdown_event),
                name="housing-delivery",
            )
            logger.info("Housing subsystem enabled")

        if settings.backfill_enabled:
            self._backfill_task = asyncio.create_task(
                self.backfill.run_forever(self._shutdown_event),
                name="history-backfill",
            )
            logger.info("Background history archive enabled")
        if settings.summary_enabled:
            self._summary_task = asyncio.create_task(
                self._run_summary_scheduler(),
                name="summary-scheduler",
            )
            logger.info(
                "Summary scheduler enabled (daily at %02d:00 UTC)",
                settings.summary_hour_utc,
            )

    async def _stop_background_tasks(self) -> None:
        # Stop the archive first. It is the only task that keeps a Telegram
        # request in flight for seconds at a time, and cutting the connection
        # underneath it turns a clean pause into an ambiguous outcome.
        for task in (
            self._backfill_task,
            self._joiner_task,
            self._discovery_task,
            self._agent_watcher_task,
            self._housing_task,
            self._housing_delivery_task,
        ):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        self._backfill_task = None
        self._agent_watcher_task = None
        self._joiner_task = None
        self._discovery_task = None

        # Stop ingress before draining the bounded queue. A message committed
        # just before disconnect already has durable pending watcher jobs.
        if self.client.is_connected():
            try:
                await self.client.disconnect()
            except Exception:
                logger.exception("Telegram disconnect failed during shutdown")

        if self._summary_task is not None:
            self._summary_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._summary_task
            self._summary_task = None
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None

        try:
            await asyncio.wait_for(
                self._message_queue.join(),
                timeout=settings.shutdown_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "Queue did not drain within %d seconds; cancelling workers",
                settings.shutdown_timeout_seconds,
            )

        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

        if self._outbox_task is not None:
            try:
                await asyncio.wait_for(
                    self._outbox_task,
                    timeout=settings.shutdown_timeout_seconds,
                )
            except TimeoutError:
                self._outbox_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._outbox_task
            self._outbox_task = None

        # Flush alerts created while the message queue was draining. Future
        # retries remain durable and will be recovered on the next start.
        try:
            async with asyncio.timeout(settings.shutdown_timeout_seconds):
                await self._deliver_due_alerts()
        except TimeoutError:
            logger.warning("Final outbox flush timed out; alerts remain durable")
        except Exception:
            logger.exception("Final outbox flush failed; alerts remain durable")

    def _register_handlers(self) -> None:
        """Register Telethon event handlers."""

        @self.client.on(events.NewMessage)  # type: ignore[untyped-decorator]
        async def on_new_message(event: NewMessageEvent) -> None:
            try:
                work_item = await self._ingest_update(event)
                if work_item is not None:
                    await self._message_queue.put(work_item)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Continuing after an unpersisted Telegram update would create
                # a silent monitoring gap. Stop visibly so supervision can
                # restart the session and operators can investigate.
                self._fatal_ingress_error = error
                self._shutdown_event.set()
                logger.critical("Durable Telegram ingress failed; stopping daemon", exc_info=True)

    async def _message_worker(self, worker_id: int) -> None:
        """Process queued Telegram updates with bounded concurrency."""
        while True:
            work_item = await self._message_queue.get()
            try:
                await self._process_work_item(work_item)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Message worker %d failed an update", worker_id)
            finally:
                self._message_queue.task_done()

    async def _input_peer(self, chat_id: int) -> object | None:
        """Resolve a chat the account already knows, without a network call.

        Only chats the session has seen are archived, so a cache miss means
        the daemon has not met this chat yet and the target should wait rather
        than spend a resolve on it.
        """
        try:
            peer: object = await self.client.get_input_entity(chat_id)
        except (ValueError, TypeError):
            return None
        return peer

    async def _named_entity(self, reference: str) -> object | None:
        """Resolve a public @username into an entity for the join queue."""
        try:
            entity: object = await self.client.get_entity(reference)
        except Exception:
            logger.warning("Cannot resolve %s", reference, exc_info=True)
            return None
        return entity

    async def reconcile_agent_watchers(self, *, apply: bool = True) -> ReconcileResult:
        """Fold assistant-authored watchers into the live set.

        Seeding the semantic index happens here, before routing is rebuilt, and
        that ordering is the whole point: a watcher that reaches routing without
        its reference vectors degrades at Level 2 on every message, which the
        default policy renders as silence. It would look enabled and never
        alert -- the failure this feature exists to remove.
        """
        stored = await self.db.active_agent_watchers()
        merged, result = merge_agent_watchers(
            config_watchers=self._config_watchers,
            stored=stored,
            current=self._agent_watchers,
        )

        if not result.changed:
            self._agent_watchers = merged
            if apply:
                # The revision counter moved even though no definition did, so
                # something else about the watcher set changed -- in practice,
                # which chats it is bound to. Routing is built from bindings, and
                # skipping this reload leaves a watcher whose routing exists in
                # the database and nowhere in the running process: enabled on
                # paper, blind in fact. Reloading is cheap; seeding, below, is
                # the expensive part and stays gated on a real definition change.
                await self.reload_observation()
            return result

        self._agent_watchers = merged
        self.watchers = [*self._config_watchers, *merged.values()]
        self.watchers_by_name = {watcher.name: watcher for watcher in self.watchers}
        self.watcher_fingerprints = {
            watcher.name: effective_policy_fingerprint(watcher) for watcher in self.watchers
        }

        if not apply:
            # Startup path: the caller is about to seed the semantic index and
            # rebuild routing anyway, and doing it twice would embed every
            # reference set a second time for nothing.
            return result

        # Seed before routing: start() is fingerprint-gated, so unchanged
        # watchers cost nothing and a new or edited one is embedded exactly once.
        await self.embedding_filter.start(self.watchers)
        await self.reload_observation()
        return result

    async def reload_observation(self) -> None:
        """Rebuild routing from the chat registry without restarting.

        Every structure is mutated in place rather than rebound: the rule
        filters were handed to :class:`MessageProcessor` by reference at
        construction time, so rebinding the attribute here would leave the
        processor holding the previous mapping and a newly promoted chat would
        raise ``KeyError`` on its first message.
        """
        # Config bindings are derived from the reviewed policies only: an agent
        # watcher declares no chats, so passing the merged list here would let a
        # future non-empty `chats` field bind itself.
        await self.db.sync_config_bindings(self._config_watchers)
        snapshot = await self.db.observation_snapshot()

        self.observed_chats.clear()
        self.observed_chats.update(snapshot)

        routing: dict[int, list[Watcher]] = {}
        for chat_id, observed in snapshot.items():
            if observed.mode is not ObservationMode.MONITOR:
                continue
            bound = [
                self.watchers_by_name[name]
                for name in observed.watcher_names
                if name in self.watchers_by_name
            ]
            if bound:
                routing[chat_id] = bound
        self.chat_watchers.clear()
        self.chat_watchers.update(routing)

        self.filters.clear()
        self.filters.update({watcher.name: RuleFilter(watcher) for watcher in self.watchers})

        logger.info(
            "Observation reloaded: %d monitored, %d under reconnaissance, %d paused",
            sum(1 for c in snapshot.values() if c.mode is ObservationMode.MONITOR),
            sum(1 for c in snapshot.values() if c.mode is ObservationMode.RECON),
            sum(1 for c in snapshot.values() if c.mode is ObservationMode.PAUSED),
        )

    async def _capture_scout_message(self, event: NewMessageEvent, chat_id: int) -> None:
        """Store a message from a chat that is being explored, not monitored.

        Capture starts the moment a chat enters the registry, which closes the
        window between joining a chat and promoting it: without this, every
        message sent in that window would be lost, since backfill can only
        reach what Telegram still serves as history.
        """
        message = event.message
        forward = getattr(message, "fwd_from", None)
        forward_peer = getattr(forward, "from_id", None)
        await self.scout.store_message(
            ScoutMessage(
                chat_id=chat_id,
                telegram_msg_id=int(message.id),
                date=str(message.date),
                text=event.text or None,
                sender_id=int(message.sender_id) if message.sender_id is not None else None,
                forward_chat_id=(
                    int(getattr(forward_peer, "channel_id", 0)) or None if forward_peer else None
                ),
                forward_message_id=(
                    int(getattr(forward, "channel_post", 0)) or None if forward else None
                ),
                reply_to_message_id=reply_to_message_id(message),
                source="live",
            )
        )

    async def _ingest_update(
        self,
        event: NewMessageEvent,
    ) -> PipelineWorkItem | None:
        """Persist a Telegram update before placing work in volatile memory."""
        chat_id = event.chat_id
        if chat_id is None:
            logger.warning("Ignoring Telegram update without a chat ID")
            return None

        observed = self.observed_chats.get(chat_id)
        if observed is None or observed.mode is ObservationMode.PAUSED:
            return None
        if observed.mode is ObservationMode.RECON:
            await self._capture_scout_message(event, chat_id)
            return None

        watchers = self.chat_watchers.get(chat_id, [])
        if not watchers:
            return None

        msg_id: int | None = None
        for attempt in range(1, settings.ingress_max_attempts + 1):
            try:
                msg_id = await ingest_message(
                    event,
                    self.db,
                    store_raw_json=settings.store_raw_telegram_json,
                    watcher_names=[watcher.name for watcher in watchers],
                    watcher_fingerprints={
                        watcher.name: self.watcher_fingerprints[watcher.name]
                        for watcher in watchers
                    },
                )
                break
            except aiosqlite.OperationalError:
                if attempt == settings.ingress_max_attempts:
                    raise
                delay = settings.ingress_retry_base_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "SQLite ingress attempt %d/%d failed; retrying in %.2fs",
                    attempt,
                    settings.ingress_max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
        if msg_id is None:
            return None  # duplicate

        # Debug echo is deliberately best-effort and happens after durable
        # ingestion so an auxiliary delivery cannot lose pipeline work.
        if settings.debug_echo and event.text:
            try:
                chat = await event.get_chat()
                sender = await event.get_sender()
                await self.dispatcher.send_echo(
                    chat_title=str(getattr(chat, "title", "DM")),
                    sender_name=(
                        str(getattr(sender, "first_name", "Unknown")) if sender else "Unknown"
                    ),
                    text=event.text,
                )
            except Exception:
                logger.exception("Debug echo failed for message_id=%d", msg_id)

        return PipelineWorkItem(
            message_id=msg_id,
            text=event.text,
            watcher_names=tuple(watcher.name for watcher in watchers),
            chat_id=chat_id,
            telegram_msg_id=int(event.message.id),
            media=media_pointer(event.message),
        )

    async def _process_work_item(self, work_item: PipelineWorkItem) -> None:
        """Run all policies attached during durable ingestion."""
        for watcher_name in work_item.watcher_names:
            watcher = self.watchers_by_name.get(watcher_name)
            if watcher is None:
                await self.db.record_pipeline_outcome(
                    PipelineOutcome(
                        message_id=work_item.message_id,
                        watcher_name=watcher_name,
                        processing_status=PipelineRunStatus.FAILED,
                        error_code="watcher_removed",
                    )
                )
                continue
            try:
                await self._process_watcher(
                    watcher=watcher,
                    work_item=work_item,
                )
            except Exception:
                logger.exception(
                    "Watcher processing failed: watcher=%s message_id=%d",
                    watcher.name,
                    work_item.message_id,
                )

    async def _process_watcher(
        self,
        *,
        watcher: Watcher,
        work_item: PipelineWorkItem,
    ) -> None:
        """Send one message to whichever judge this policy declares.

        A policy with ``dispatch: external`` never touches the relevance
        ladder. It is not a cheaper path through the same stages: the
        subsystem behind it decides what a message even is, which for housing
        means grouping several Telegram messages into one advertisement before
        anything is judged at all.
        """
        if watcher.dispatch == "external":
            await self._dispatch_external(watcher=watcher, work_item=work_item)
            return
        outcome = await self.processor.process(
            watcher=watcher,
            message_id=work_item.message_id,
            text=work_item.text,
        )
        if outcome.alert_created:
            self._outbox_wakeup.set()

    async def _dispatch_external(
        self,
        *,
        watcher: Watcher,
        work_item: PipelineWorkItem,
    ) -> None:
        """Hand a message to the subsystem registered under this policy's name."""
        if watcher.name != HOUSING_WATCHER_NAME or self.housing is None:
            logger.warning(
                "Policy %s declares external dispatch with no subsystem behind it",
                watcher.name,
            )
            return
        await self.housing.record_message(
            unit_key=unit_key_for(
                work_item.chat_id,
                grouped_id=work_item.media.grouped_id,
                telegram_msg_id=work_item.telegram_msg_id,
            ),
            chat_id=work_item.chat_id,
            grouped_id=work_item.media.grouped_id,
            message_id=work_item.message_id,
            telegram_msg_id=work_item.telegram_msg_id,
            text=work_item.text,
            has_media=work_item.media.has_media,
            telegram_photo_id=work_item.media.telegram_photo_id,
        )
        # The durable job is finished the moment the message reaches the
        # subsystem, because from here the advertisement's own state machine
        # owns it. Leaving the job pending would make recovery replay it
        # forever and, worse, would eventually refuse to make progress.
        await self.db.record_pipeline_outcome(
            PipelineOutcome(
                message_id=work_item.message_id,
                watcher_name=watcher.name,
                processing_status=PipelineRunStatus.COMPLETED,
                rule_passed=True,
            )
        )

    async def _recover_pending_pipeline_jobs(self) -> None:
        """Finish message/watcher jobs left pending by an interrupted process."""
        recovered = 0
        while True:
            jobs = await self.db.get_pending_pipeline_jobs(settings.recovery_batch_size)
            if not jobs:
                break
            pending_keys = {(job.message_id, job.watcher_name) for job in jobs}
            for job in jobs:
                watcher = self.watchers_by_name.get(job.watcher_name)
                if watcher is None:
                    await self.db.record_pipeline_outcome(
                        PipelineOutcome(
                            message_id=job.message_id,
                            watcher_name=job.watcher_name,
                            processing_status=PipelineRunStatus.FAILED,
                            error_code="watcher_removed",
                        )
                    )
                    continue
                if job.watcher_config_fingerprint != self.watcher_fingerprints[job.watcher_name]:
                    await self.db.record_pipeline_outcome(
                        PipelineOutcome(
                            message_id=job.message_id,
                            watcher_name=job.watcher_name,
                            processing_status=PipelineRunStatus.FAILED,
                            error_code="watcher_policy_changed",
                        )
                    )
                    logger.warning(
                        "Skipped pending job created under another policy: "
                        "watcher=%s message_id=%d",
                        job.watcher_name,
                        job.message_id,
                    )
                    continue
                try:
                    await self._process_watcher(
                        watcher=watcher,
                        work_item=PipelineWorkItem(
                            message_id=job.message_id,
                            text=job.text,
                            watcher_names=(job.watcher_name,),
                            chat_id=job.chat_id,
                            telegram_msg_id=job.telegram_msg_id,
                            media=job.media,
                        ),
                    )
                    recovered += 1
                except Exception:
                    logger.exception(
                        "Recovered pipeline job failed: watcher=%s message_id=%d",
                        job.watcher_name,
                        job.message_id,
                    )

            remaining = await self.db.get_pending_pipeline_jobs(settings.recovery_batch_size)
            if not remaining:
                break
            remaining_keys = {(job.message_id, job.watcher_name) for job in remaining}
            if pending_keys & remaining_keys:
                raise RuntimeError("pending pipeline recovery made no progress")

        if recovered:
            logger.info("Recovered %d interrupted pipeline jobs", recovered)

    async def _run_outbox_worker(self) -> None:
        """Continuously deliver committed alerts and recover retryable failures."""
        while not self._shutdown_event.is_set():
            self._outbox_wakeup.clear()
            try:
                delivered = await self._deliver_due_alerts()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Alert outbox cycle failed")
                delivered = 0

            if delivered >= settings.outbox_batch_size:
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    self._outbox_wakeup.wait(),
                    timeout=settings.outbox_poll_seconds,
                )

    async def _deliver_due_alerts(self) -> int:
        """Deliver a bounded number of rows, leasing each just before use."""
        delivered = 0
        for _ in range(settings.outbox_batch_size):
            alerts = await self.db.claim_due_alerts(
                limit=1,
                lease_seconds=settings.outbox_lease_seconds,
            )
            if not alerts:
                break
            alert = alerts[0]
            try:
                result = await self.dispatcher.deliver_alert(
                    watcher_name=alert.watcher_name,
                    chat_title=alert.chat_title,
                    sender_name=alert.sender_name,
                    text=alert.text,
                    matched_keyword=alert.matched_keyword,
                    filter_level=alert.filter_level,
                )
                await self.db.mark_alert_delivery_result(
                    alert.alert_id,
                    result,
                    claim_token=alert.claim_token,
                    max_attempts=settings.outbox_max_attempts,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # The persisted lease expires, making this row recoverable
                # without storing exception text or losing the alert.
                logger.exception(
                    "Outbox item failed: alert_id=%d watcher=%s",
                    alert.alert_id,
                    alert.watcher_name,
                )
            delivered += 1
        return delivered

    async def _run_summary_scheduler(self) -> None:
        """Sleep until summary_hour_utc each day, then generate and send digests."""
        while not self._shutdown_event.is_set():
            now = datetime.now(UTC)
            target = now.replace(
                hour=settings.summary_hour_utc,
                minute=0,
                second=0,
                microsecond=0,
            )
            if target <= now:
                target += timedelta(days=1)
            sleep_seconds = (target - now).total_seconds()
            logger.info(
                "Next summary in %.0f seconds (at %s UTC)", sleep_seconds, target.strftime("%H:%M")
            )

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=sleep_seconds,
                )
                return  # shutdown requested
            except TimeoutError:
                pass  # time to generate summary

            try:
                await self._generate_summaries(window_end=target)
            except Exception:
                logger.exception("Scheduled summary cycle failed")

    async def _run_retention_sweeper(self) -> None:
        """Continuously enforce the configured message-retention window."""
        interval_seconds = settings.retention_sweep_hours * 3600
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval_seconds,
                )
                return
            except TimeoutError:
                try:
                    purged = await self.db.purge_expired_data(settings.retention_days)
                except Exception:
                    logger.exception("Retention sweep failed")
                else:
                    if purged:
                        logger.info("Retention sweep purged %d messages", purged)

    async def _generate_summaries(self, *, window_end: datetime | None = None) -> None:
        """Generate one best-effort 24-hour summary window for every watcher."""
        end = window_end or datetime.now(UTC)
        start = end - timedelta(days=1)
        date_str = f"{start.strftime('%Y-%m-%d %H:%MZ')} → {end.strftime('%Y-%m-%d %H:%MZ')}"

        for watcher in self.watchers:
            try:
                messages = await self.db.get_accepted_messages_between(
                    watcher.name,
                    start.isoformat(),
                    end.isoformat(),
                )
                if not messages:
                    logger.info(
                        "No messages for [%s] on %s, skipping summary",
                        watcher.name,
                        date_str,
                    )
                    continue

                summary = await self.summarizer.summarize(
                    messages=messages,
                    watcher_name=watcher.name,
                    target_date=end.date(),
                )
                if summary and not await self.dispatcher.send_summary(
                    watcher_name=watcher.name,
                    summary=summary,
                    date_str=date_str,
                    message_count=len(messages),
                ):
                    logger.error("Best-effort summary delivery failed: watcher=%s", watcher.name)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Summary watcher failed: watcher=%s", watcher.name)

    def _setup_signals(self) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._shutdown_event.set)

    def _remove_signals(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.remove_signal_handler(sig)


def main() -> None:
    try:
        app = Eidolon()
    except (RuntimeConfigurationError, WatcherConfigError) as error:
        logger.error("Startup configuration error: %s", error)
        sys.exit(2)
    asyncio.run(app.start())


if __name__ == "__main__":
    main()
