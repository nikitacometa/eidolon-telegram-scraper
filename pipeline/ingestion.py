"""Message ingestion — extracts fields from Telethon events and stores in DB."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from telethon.tl.types import (
    Channel,
    Chat,
)

from pipeline.models import MediaPointer
from storage.db import Database

logger = logging.getLogger(__name__)


class TelegramMessage(Protocol):
    """Fields consumed from a Telethon message without leaking untyped APIs."""

    id: int
    chat_id: int | None
    text: str | None
    date: object
    sender_id: int | None
    fwd_from: object | None
    reply_to: object | None

    def to_dict(self) -> dict[str, object]: ...


class NewMessageEvent(Protocol):
    """Minimal async event contract used by ingestion and the worker queue."""

    message: TelegramMessage
    chat_id: int | None
    text: str | None

    async def get_sender(self) -> object | None: ...

    async def get_chat(self) -> object | None: ...


async def ingest_message(
    event: NewMessageEvent,
    db: Database,
    *,
    store_raw_json: bool = False,
    watcher_names: Sequence[str] = (),
    watcher_fingerprints: Mapping[str, str] | None = None,
) -> int | None:
    """Extract message data from a Telethon NewMessage event and store it.

    Returns the database row ID, or None if the message is a duplicate.
    """
    msg = event.message

    # Entity lookup is network-backed and optional. Persist the update even
    # when Telegram cannot enrich sender/chat metadata.
    try:
        sender = await event.get_sender()
    except Exception as error:
        logger.warning("Sender metadata unavailable: error=%s", type(error).__name__)
        sender = None
    sender_id = getattr(sender, "id", None)
    sender_name = _get_sender_name(sender)

    try:
        chat = await event.get_chat()
    except Exception as error:
        logger.warning("Chat metadata unavailable: error=%s", type(error).__name__)
        chat = None
    chat_id = msg.chat_id if msg.chat_id is not None else event.chat_id
    if chat_id is None:
        raise ValueError("Telegram update has no chat ID")
    chat_title = _get_chat_title(chat)
    chat_type = _get_chat_type(chat)
    raw_json = None
    if store_raw_json and msg.text:
        try:
            raw_json = _safe_to_json(msg.to_dict())
        except Exception as error:
            logger.warning("Raw Telegram payload unavailable: error=%s", type(error).__name__)

    # Store the message
    row_id = await db.store_message(
        telegram_msg_id=msg.id,
        chat_id=chat_id,
        chat_title=chat_title,
        chat_type=chat_type,
        sender_id=sender_id,
        sender_name=sender_name,
        text=msg.text,
        date=msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
        raw_json=raw_json,
        reply_to_message_id=reply_to_message_id(msg),
        media=media_pointer(msg),
        watcher_names=watcher_names,
        watcher_fingerprints=watcher_fingerprints,
    )

    if row_id:
        logger.debug(
            "Ingested Telegram message %d in chat %d",
            msg.id,
            chat_id,
        )

    return row_id


def media_pointer(message: object) -> MediaPointer:
    """Read what a message carries without asking Telegram anything.

    Everything here is already in the delivered object, so this costs no
    request and no rate-limit budget. Downloading the file is a separate,
    governed decision made much later, and only for messages that turned out
    to be worth it.

    A document is counted when Telegram labels it an image: Telegram clients
    send a photograph as a document whenever the sender ticks "send without
    compression", which is exactly what someone photographing an apartment
    tends to do.
    """
    grouped = getattr(message, "grouped_id", None)
    grouped_id = (
        int(grouped) if isinstance(grouped, int) and not isinstance(grouped, bool) else None
    )

    photo = getattr(message, "photo", None)
    photo_id = getattr(photo, "id", None) if photo is not None else None
    if photo_id is None:
        document = getattr(message, "document", None)
        mime = str(getattr(document, "mime_type", "") or "") if document is not None else ""
        if document is not None and mime.startswith("image/"):
            photo_id = getattr(document, "id", None)
            photo = document

    has_media = photo is not None and photo_id is not None
    return MediaPointer(
        has_media=has_media,
        telegram_photo_id=int(photo_id) if isinstance(photo_id, int) else None,
        grouped_id=grouped_id,
    )


def reply_to_message_id(message: object) -> int | None:
    """Extract Telethon's nested reply target without treating the header as an integer."""
    reply_header = getattr(message, "reply_to", None)
    value = getattr(reply_header, "reply_to_msg_id", None)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _get_sender_name(sender: object | None) -> str:
    """Prefer a reachable Telegram handle while retaining the display name."""
    if sender is None:
        return "Unknown"
    return format_sender_identity(sender) or "Unknown"


def format_sender_identity(sender: object) -> str | None:
    """Return ``@handle (Display Name)`` when Telegram exposes both."""
    parts = [getattr(sender, "first_name", None), getattr(sender, "last_name", None)]
    display_name = " ".join(str(part).strip() for part in parts if part)
    if not display_name:
        display_name = str(getattr(sender, "title", None) or "").strip()
    username = str(getattr(sender, "username", None) or "").strip().lstrip("@")
    if username:
        handle = f"@{username}"
        return f"{handle} ({display_name})" if display_name else handle
    return display_name or None


def _get_chat_title(chat: object | None) -> str:
    """Extract chat title from a Telethon chat entity."""
    if chat is None:
        return "Unknown"
    return getattr(chat, "title", None) or "DM"


def _get_chat_type(chat: object | None) -> str:
    """Determine chat type string."""
    if isinstance(chat, Channel):
        return "channel" if chat.broadcast else "supergroup"
    if isinstance(chat, Chat):
        return "group"
    return "private"


def _safe_to_json(obj: Mapping[str, object]) -> str | None:
    """Serialize a dict to JSON, handling non-serializable values."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
