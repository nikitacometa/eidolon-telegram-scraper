"""Generate a Telethon StringSession for headless authentication.

Run once interactively. By default the session is written to `.env` with mode
0600 instead of being exposed in terminal history.
"""

import argparse
import asyncio
import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

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
    try:
        await client.start(phone=settings.telegram_phone)
        return str(client.session.save())
    finally:
        await client.disconnect()


def save_session(session_string: str, env_path: Path = Path(".env")) -> None:
    """Atomically update TELEGRAM_SESSION_STRING in a private env file."""
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = existing.splitlines()
    replacement = f"TELEGRAM_SESSION_STRING={session_string}"

    for index, line in enumerate(lines):
        if line.startswith("TELEGRAM_SESSION_STRING="):
            lines[index] = replacement
            break
    else:
        lines.append(replacement)

    env_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{env_path.name}.",
        dir=env_path.parent,
        text=True,
    )
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file.write("\n".join(lines).rstrip() + "\n")
        os.replace(temporary_name, env_path)
        os.chmod(env_path, 0o600)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-session",
        action="store_true",
        help="print the credential to stdout instead of writing .env (unsafe)",
    )
    arguments = parser.parse_args()

    print("Generating Telegram session string...")
    print("You will be prompted for a verification code sent to your Telegram app.\n")

    session_string = asyncio.run(generate_session())
    if arguments.print_session:
        print("WARNING: the following value grants access to the Telegram account.")
        print(f"TELEGRAM_SESSION_STRING={session_string}")
    else:
        save_session(session_string)
        print("Session saved to .env with owner-only permissions.")


if __name__ == "__main__":
    if not settings.telegram_api_id or not settings.telegram_api_hash:
        print("Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        sys.exit(1)
    if not settings.telegram_phone:
        print("Error: TELEGRAM_PHONE must be set in .env")
        sys.exit(1)
    main()
