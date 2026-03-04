"""Generate a Telethon StringSession for headless authentication.

Run once interactively:
    python3 auth.py

The script prints a session string. Copy it into .env as TELEGRAM_SESSION_STRING.
"""

import asyncio
import sys

from telethon import TelegramClient
from telethon.sessions import StringSession

from config.settings import settings


async def generate_session() -> str:
    """Connect to Telegram, authenticate, and return a StringSession string."""
    client = TelegramClient(
        StringSession(),
        settings.telegram_api_id,
        settings.telegram_api_hash,
    )
    await client.start(phone=settings.telegram_phone)
    session_string = client.session.save()
    await client.disconnect()
    return session_string


def main() -> None:
    print("Generating Telegram session string...")
    print("You will be prompted for a verification code sent to your Telegram app.\n")

    session_string = asyncio.run(generate_session())

    print("\n" + "=" * 60)
    print("Session string generated successfully!")
    print("=" * 60)
    print("\nAdd this to your .env file:")
    print(f"\nTELEGRAM_SESSION_STRING={session_string}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    if not settings.telegram_api_id or not settings.telegram_api_hash:
        print("Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        sys.exit(1)
    if not settings.telegram_phone:
        print("Error: TELEGRAM_PHONE must be set in .env")
        sys.exit(1)
    main()
