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
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Connect to Telegram and database, register handlers, run until shutdown."""
        await self.db.connect()
        await self.client.start()

        me = await self.client.get_me()
        logger.info("Connected as %s (ID: %d)", me.first_name, me.id)

        self._register_handlers()
        self._setup_signals()

        logger.info("Eidolon is listening...")
        await self._shutdown_event.wait()

        logger.info("Shutting down...")
        await self.client.disconnect()
        await self.db.close()
        logger.info("Goodbye.")

    def _register_handlers(self) -> None:
        """Register Telethon event handlers."""

        @self.client.on(events.NewMessage)
        async def on_new_message(event: events.NewMessage.Event) -> None:
            # Will be wired to the full pipeline in Milestone 3
            # For now: log incoming messages
            chat = await event.get_chat()
            chat_title = getattr(chat, "title", "DM")
            sender = await event.get_sender()
            sender_name = getattr(sender, "first_name", "Unknown") if sender else "Unknown"
            text_preview = (event.text or "")[:80]
            logger.debug("[%s] %s: %s", chat_title, sender_name, text_preview)

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
