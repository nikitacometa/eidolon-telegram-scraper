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
from datetime import UTC, datetime, timedelta

from telethon import TelegramClient, events
from telethon.sessions import StringSession

from config.settings import settings
from config.watchers import Watcher, get_chat_watchers, load_watchers
from pipeline.dispatcher import AlertDispatcher
from pipeline.embeddings import EmbeddingFilter
from pipeline.filters import RuleFilter
from pipeline.ingestion import NewMessageEvent, ingest_message
from pipeline.llm import LLMClassifier, decision_verdict
from pipeline.models import PipelineOutcome, StageStatus
from pipeline.summarizer import DailySummarizer
from storage.db import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eidolon")


class Eidolon:
    """Main application: Telethon client + event loop + pipeline."""

    def __init__(self) -> None:
        self.db = Database(settings.db_path)
        self.client = TelegramClient(
            StringSession(settings.telegram_session_string),
            settings.telegram_api_id,
            settings.telegram_api_hash,
            sequential_updates=True,
        )
        self.dispatcher = AlertDispatcher()
        self.embedding_filter = EmbeddingFilter()
        self.llm_classifier = LLMClassifier()
        self.summarizer = DailySummarizer()
        self._shutdown_event = asyncio.Event()
        self._message_queue: asyncio.Queue[NewMessageEvent] = asyncio.Queue(
            maxsize=settings.processing_queue_size
        )
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._summary_task: asyncio.Task[None] | None = None

        # Load watcher configs
        self.watchers = load_watchers(settings.watchers_path)
        self.chat_watchers = get_chat_watchers(self.watchers)
        self.filters: dict[str, RuleFilter] = {w.name: RuleFilter(w) for w in self.watchers}

    async def start(self) -> None:
        """Connect to Telegram and database, register handlers, run until shutdown."""
        self._setup_signals()
        self._register_handlers()

        try:
            async with AsyncExitStack() as stack:
                await self.db.connect()
                stack.push_async_callback(self.db.close)
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
        if self._summary_task is not None:
            self._summary_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._summary_task
            self._summary_task = None

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

    def _register_handlers(self) -> None:
        """Register Telethon event handlers."""

        @self.client.on(events.NewMessage)  # type: ignore[untyped-decorator]
        async def on_new_message(event: NewMessageEvent) -> None:
            await self._message_queue.put(event)

    async def _message_worker(self, worker_id: int) -> None:
        """Process queued Telegram updates with bounded concurrency."""
        while True:
            event = await self._message_queue.get()
            try:
                await self._process_message(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Message worker %d failed an update", worker_id)
            finally:
                self._message_queue.task_done()

    async def _process_message(self, event: NewMessageEvent) -> None:
        """Full pipeline: ingest → filter → dispatch."""
        chat_id = event.chat_id
        if chat_id is None:
            logger.warning("Ignoring Telegram update without a chat ID")
            return

        # Check if any watcher monitors this chat
        watchers = self.chat_watchers.get(chat_id, [])
        if not watchers:
            return

        # Extract common info
        chat = await event.get_chat()
        chat_title = getattr(chat, "title", "DM")
        sender = await event.get_sender()
        sender_name = getattr(sender, "first_name", "Unknown") if sender else "Unknown"

        # Debug echo: forward ALL messages from monitored chats
        if settings.debug_echo and event.text:
            await self.dispatcher.send_echo(
                chat_title=chat_title,
                sender_name=sender_name,
                text=event.text,
            )

        # Ingest message into DB
        msg_id = await ingest_message(
            event,
            self.db,
            store_raw_json=settings.store_raw_telegram_json,
        )
        if msg_id is None:
            return  # duplicate

        # Run through each watcher's filter
        for watcher in watchers:
            try:
                await self._process_watcher(
                    watcher=watcher,
                    message_id=msg_id,
                    text=event.text,
                    chat_title=chat_title,
                    sender_name=sender_name,
                )
            except Exception:
                logger.exception(
                    "Watcher processing failed: watcher=%s message_id=%d",
                    watcher.name,
                    msg_id,
                )

    async def _process_watcher(
        self,
        *,
        watcher: Watcher,
        message_id: int,
        text: str | None,
        chat_title: str,
        sender_name: str,
    ) -> None:
        """Run one watcher and persist exactly one idempotent outcome."""
        outcome = PipelineOutcome(message_id=message_id, watcher_name=watcher.name)
        filter_level = 1
        llm_response: str | None = None
        matched_keyword: str | None = None

        try:
            rule_filter = self.filters[watcher.name]
            result = rule_filter.check(text)
            if not result:
                return
            outcome.rule_passed = True
            matched_keyword = result.matched_keyword

            # Level 2: Embedding similarity (if configured)
            if watcher.llm_level >= 2 and text:
                embedding = await self.embedding_filter.check(text, watcher.name)
                outcome.embedding_status = embedding.status
                outcome.embedding_passed = embedding.passed
                outcome.embedding_score = embedding.score
                if embedding.error_code:
                    outcome.error_code = embedding.error_code
                if not embedding.passed:
                    logger.info(
                        "Embedding filtered message: watcher=%s message_id=%d score=%s",
                        watcher.name,
                        message_id,
                        embedding.score,
                    )
                    return
                if embedding.status is StageStatus.OK:
                    filter_level = 2

            # Level 3: LLM classification (if configured)
            if watcher.llm_level >= 3 and text:
                objective = "\n\n".join(
                    part for part in (watcher.description.strip(), watcher.prompt.strip()) if part
                )
                classification = await self.llm_classifier.classify(
                    text=text,
                    watcher_prompt=objective,
                )
                verdict = decision_verdict(classification)
                outcome.llm_status = classification.status
                outcome.llm_relevant = classification.result.relevant
                outcome.llm_verdict = verdict.value
                outcome.llm_confidence = classification.result.confidence
                if classification.error_code:
                    outcome.error_code = classification.error_code
                llm_response = classification.result.model_dump_json()
                if classification.status is StageStatus.OK:
                    filter_level = 3
                if not classification.result.relevant:
                    logger.info(
                        "LLM filtered message: watcher=%s message_id=%d confidence=%.3f",
                        watcher.name,
                        message_id,
                        classification.result.confidence,
                    )
                    return

            # Store alert in DB
            alert_id = await self.db.store_alert(
                watcher_name=watcher.name,
                message_id=message_id,
                filter_level=filter_level,
                score=outcome.embedding_score,
                llm_response=llm_response,
            )
            outcome.alert_created = True

            # Dispatch alert
            if watcher.alert == "immediate":
                sent = await self.dispatcher.send_alert(
                    watcher_name=watcher.name,
                    chat_title=chat_title,
                    sender_name=sender_name,
                    text=text or "",
                    matched_keyword=matched_keyword,
                    filter_level=filter_level,
                )
                if sent:
                    await self.db.mark_alert_sent(alert_id)
                    outcome.alert_sent = True
        except Exception as error:
            outcome.error_code = type(error).__name__
            raise
        finally:
            await self.db.record_pipeline_outcome(outcome)

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

            await self._generate_summaries()

    async def _generate_summaries(self) -> None:
        """Generate and send summaries for all watchers."""
        today = datetime.now(UTC).date()
        date_str = today.isoformat()

        for watcher in self.watchers:
            messages = await self.db.get_daily_messages(watcher.chats, date_str)
            if not messages:
                logger.info("No messages for [%s] on %s, skipping summary", watcher.name, date_str)
                continue

            summary = await self.summarizer.summarize(
                messages=messages,
                watcher_name=watcher.name,
                target_date=today,
            )
            if summary:
                await self.dispatcher.send_summary(
                    watcher_name=watcher.name,
                    summary=summary,
                    date_str=date_str,
                    message_count=len(messages),
                )

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
    if not settings.telegram_session_string:
        logger.error("TELEGRAM_SESSION_STRING is empty. Run 'python3 auth.py' first.")
        sys.exit(1)

    app = Eidolon()
    asyncio.run(app.start())


if __name__ == "__main__":
    main()
