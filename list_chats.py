"""List all groups and channels the account is a member of.

Run after generating a session string:
    python3 list_chats.py

Outputs a table of chat titles and IDs for use in watchers.yml.
"""

import asyncio
import sys

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.types import Channel, Chat

from config.settings import settings


async def list_chats() -> None:
    client = TelegramClient(
        StringSession(settings.telegram_session_string),
        settings.telegram_api_id,
        settings.telegram_api_hash,
    )
    await client.start()

    print(f"{'Title':<50} {'ID':<20} {'Type':<12} {'Members':>8}")
    print("-" * 95)

    async for dialog in client.iter_dialogs():
        entity = dialog.entity
        if isinstance(entity, (Channel, Chat)):
            chat_type = "channel" if isinstance(entity, Channel) and entity.broadcast else "group"
            members = getattr(entity, "participants_count", None) or ""
            # Telegram supergroup/channel IDs need -100 prefix
            chat_id = f"-100{entity.id}" if isinstance(entity, Channel) else str(-entity.id)
            print(f"{dialog.title:<50} {chat_id:<20} {chat_type:<12} {members:>8}")

    await client.disconnect()


def main() -> None:
    if not settings.telegram_session_string:
        print("Error: TELEGRAM_SESSION_STRING is empty. Run 'python3 auth.py' first.")
        sys.exit(1)
    asyncio.run(list_chats())


if __name__ == "__main__":
    main()
