"""Eidolon — AI-powered Telegram monitor.

Entry point: connects to Telegram via MTProto, listens for messages
in configured chats, runs them through the filter pipeline, and
dispatches alerts.
"""

import asyncio
import logging
import signal
import sys

from telethon import TelegramClient, events
from telethon.sessions import StringSession

from config.settings import settings
from config.watchers import get_chat_watchers, load_watchers
from pipeline.dispatcher import AlertDispatcher
from pipeline.filters import RuleFilter
from pipeline.ingestion import ingest_message
from pipeline.llm import LLMClassifier, Verdict
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
        self.llm_classifier = LLMClassifier()
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
        await self.llm_classifier.start()
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

        logger.info("Eidolon is listening...")
        await self._shutdown_event.wait()

        logger.info("Shutting down...")
        await self.client.disconnect()
        await self.llm_classifier.close()
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

            # Level 2+: LLM classification (if configured)
            filter_level = 1
            llm_response = None
            if watcher.llm_level >= 2 and event.text:
                verdict = await self.llm_classifier.classify(
                    text=event.text,
                    watcher_prompt=watcher.prompt,
                )
                llm_response = verdict.value
                filter_level = 2
                await self.db.update_filter_stats(
                    watcher_name=watcher.name,
                    level_passed=2 if verdict == Verdict.OFFER else None,
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
