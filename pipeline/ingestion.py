"""Message ingestion — extracts fields from Telethon events and stores in DB."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from telethon.tl.types import (
    Channel,
    Chat,
    User,
)

from storage.db import Database

logger = logging.getLogger(__name__)


async def ingest_message(event, db: Database) -> int | None:
    """Extract message data from a Telethon NewMessage event and store it.

    Returns the database row ID, or None if the message is a duplicate.
    """
    msg = event.message

    # Extract sender info
    sender = await event.get_sender()
    sender_id = getattr(sender, "id", None)
    sender_name = _get_sender_name(sender)

    # Extract chat info
    chat = await event.get_chat()
    chat_id = msg.chat_id or event.chat_id
    chat_title = _get_chat_title(chat)
    chat_type = _get_chat_type(chat)

    # Store the message
    row_id = await db.store_message(
        telegram_msg_id=msg.id,
        chat_id=chat_id,
        chat_title=chat_title,
        sender_id=sender_id,
        sender_name=sender_name,
        text=msg.text,
        date=msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
        raw_json=_safe_to_json(msg.to_dict()) if msg.text else None,
    )

    # Update chat metadata
    await db.update_chat(chat_id=chat_id, title=chat_title, chat_type=chat_type)

    if row_id:
        logger.debug(
            "Ingested msg %d from [%s] %s: %s",
            msg.id,
            chat_title,
            sender_name,
            (msg.text or "")[:60],
        )

    return row_id


def _get_sender_name(sender) -> str:
    """Extract a display name from a Telethon sender entity."""
    if sender is None:
        return "Unknown"
    if isinstance(sender, User):
        parts = [sender.first_name or "", sender.last_name or ""]
        return " ".join(p for p in parts if p) or "Unknown"
    return getattr(sender, "title", None) or getattr(sender, "username", None) or "Unknown"


def _get_chat_title(chat) -> str:
    """Extract chat title from a Telethon chat entity."""
    if chat is None:
        return "Unknown"
    return getattr(chat, "title", None) or "DM"


def _get_chat_type(chat) -> str:
    """Determine chat type string."""
    if isinstance(chat, Channel):
        return "channel" if chat.broadcast else "supergroup"
    if isinstance(chat, Chat):
        return "group"
    return "private"


def _safe_to_json(obj: dict) -> str | None:
    """Serialize a dict to JSON, handling non-serializable values."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
