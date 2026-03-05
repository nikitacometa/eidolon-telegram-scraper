"""Eidolon — AI-powered Telegram monitor.

Entry point: connects to Telegram via MTProto, listens for messages
in configured chats, runs them through the filter pipeline, and
dispatches alerts.
"""

import asyncio
import logging
import signal
import sys
from datetime import date, datetime, timedelta, timezone

from telethon import TelegramClient, events
from telethon.sessions import StringSession

from config.settings import settings
from config.watchers import get_chat_watchers, load_watchers
from pipeline.dispatcher import AlertDispatcher
from pipeline.embeddings import EmbeddingFilter
from pipeline.filters import RuleFilter
from pipeline.ingestion import ingest_message
from pipeline.llm import LLMClassifier, Verdict
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
        )
        self.dispatcher = AlertDispatcher()
        self.embedding_filter = EmbeddingFilter()
        self.llm_classifier = LLMClassifier()
        self.summarizer = DailySummarizer()
        self._shutdown_event = asyncio.Event()

        # Load watcher configs
        self.watchers = load_watchers(settings.watchers_path)
        self.chat_watchers = get_chat_watchers(self.watchers)
        self.filters: dict[str, RuleFilter] = {
            w.name: RuleFilter(w) for w in self.watchers
        }

    async def start(self) -> None:
        """Connect to Telegram and database, register handlers, run until shutdown."""
        await self.db.connect()
        await self.dispatcher.start()
        await self.embedding_filter.start(self.watchers)
        await self.llm_classifier.start()
        await self.summarizer.start()
        await self.client.start()

        me = await self.client.get_me()
        logger.info("Connected as %s (ID: %d)", me.first_name, me.id)
        logger.info(
            "Monitoring %d chats via %d watchers",
            len(self.chat_watchers),
            len(self.watchers),
        )

        self._register_handlers()
        self._setup_signals()

        if settings.debug_echo:
            logger.info("DEBUG ECHO MODE — forwarding ALL messages from monitored chats")

        # Start daily summary scheduler
        if settings.summary_enabled:
            self._summary_task = asyncio.create_task(self._run_summary_scheduler())
            logger.info("Summary scheduler enabled (daily at %02d:00 UTC)", settings.summary_hour_utc)

        logger.info("Eidolon is listening...")
        await self._shutdown_event.wait()

        logger.info("Shutting down...")
        if hasattr(self, "_summary_task"):
            self._summary_task.cancel()
        await self.client.disconnect()
        await self.summarizer.close()
        await self.llm_classifier.close()
        await self.embedding_filter.close()
        await self.dispatcher.close()
        await self.db.close()
        logger.info("Goodbye.")

    def _register_handlers(self) -> None:
        """Register Telethon event handlers."""

        @self.client.on(events.NewMessage)
        async def on_new_message(event: events.NewMessage.Event) -> None:
            await self._process_message(event)

    async def _process_message(self, event: events.NewMessage.Event) -> None:
        """Full pipeline: ingest → filter → dispatch."""
        chat_id = event.chat_id

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
        msg_id = await ingest_message(event, self.db)
        if msg_id is None:
            return  # duplicate

        # Run through each watcher's filter
        for watcher in watchers:
            rule_filter = self.filters[watcher.name]
            result = rule_filter.check(event.text)

            # Update filter stats
            await self.db.update_filter_stats(
                watcher_name=watcher.name,
                level_passed=1 if result.passed else None,
            )

            if not result:
                continue

            # Level 2: Embedding similarity (if configured)
            filter_level = 1
            llm_response = None
            if watcher.llm_level >= 2 and event.text:
                emb_passed = await self.embedding_filter.check(event.text, watcher.name)
                await self.db.update_filter_stats(
                    watcher_name=watcher.name,
                    level_passed=2 if emb_passed else None,
                )
                if not emb_passed:
                    logger.info(
                        "Embedding filtered out [%s]: %s",
                        watcher.name, (event.text or "")[:50],
                    )
                    continue
                filter_level = 2

            # Level 3: LLM classification (if configured)
            if watcher.llm_level >= 3 and event.text:
                verdict = await self.llm_classifier.classify(
                    text=event.text,
                    watcher_prompt=watcher.prompt,
                )
                llm_response = verdict.value
                filter_level = 3
                await self.db.update_filter_stats(
                    watcher_name=watcher.name,
                    level_passed=3 if verdict == Verdict.OFFER else None,
                )
                if verdict != Verdict.OFFER:
                    logger.info(
                        "LLM filtered out [%s]: %s → %s",
                        watcher.name, (event.text or "")[:50], verdict.value,
                    )
                    continue

            # Store alert in DB
            alert_id = await self.db.store_alert(
                watcher_name=watcher.name,
                message_id=msg_id,
                filter_level=filter_level,
                llm_response=llm_response,
            )

            # Dispatch alert
            if watcher.alert == "immediate":
                sent = await self.dispatcher.send_alert(
                    watcher_name=watcher.name,
                    chat_title=chat_title,
                    sender_name=sender_name,
                    text=event.text or "",
                    matched_keyword=result.matched_keyword,
                    filter_level=filter_level,
                )
                if sent:
                    await self.db.mark_alert_sent(alert_id)
                    await self.db.update_filter_stats(
                        watcher_name=watcher.name,
                        alert_sent=True,
                    )

    async def _run_summary_scheduler(self) -> None:
        """Sleep until summary_hour_utc each day, then generate and send digests."""
        while not self._shutdown_event.is_set():
            now = datetime.now(timezone.utc)
            target = now.replace(
                hour=settings.summary_hour_utc, minute=0, second=0, microsecond=0,
            )
            if target <= now:
                target += timedelta(days=1)
            sleep_seconds = (target - now).total_seconds()
            logger.info("Next summary in %.0f seconds (at %s UTC)", sleep_seconds, target.strftime("%H:%M"))

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=sleep_seconds,
                )
                return  # shutdown requested
            except TimeoutError:
                pass  # time to generate summary

            await self._generate_summaries()

    async def _generate_summaries(self) -> None:
        """Generate and send summaries for all watchers."""
        today = date.today()
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
            loop.add_signal_handler(sig, self._shutdown_event.set)


def main() -> None:
    if not settings.telegram_session_string:
        logger.error("TELEGRAM_SESSION_STRING is empty. Run 'python3 auth.py' first.")
        sys.exit(1)

    app = Eidolon()
    asyncio.run(app.start())


if __name__ == "__main__":
    main()
