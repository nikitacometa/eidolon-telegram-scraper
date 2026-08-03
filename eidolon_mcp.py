#!/usr/bin/env python3
"""MCP bridge: gives an OpenClaw agent read access to the Telegram corpus.

Deliberately thin. It holds no Telegram session, opens the search index
read-only, and never imports Telethon. The daemon owns the session and one
process may hold it; a bridge that could touch it would be a second claimant on
the account, so it cannot.

The one write it can perform is appending to ``join_queue``, which is a request
rather than an action: the daemon's join worker picks the row up on its own
schedule, subject to its own budget and flood-wait ladder. Nothing here can make
Telegram do anything now.

Two profiles:
  --profile full      every tool, including queueing a chat for joining
  --profile readonly  search and reporting only; no writes are even registered

Run:
    python3 eidolon_mcp.py --profile full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ServerCapabilities, TextContent, Tool, ToolsCapability

from config.settings import settings
from pipeline.recon_models import normalize_username
from storage.search import SearchDatabase, build_fts_query

logger = logging.getLogger("eidolon.mcp")

# Results are read by a model with a finite context. A long answer is not a
# better answer here: it is the same answer with the top of it truncated by
# OpenClaw's context manager before the model ever sees it.
MESSAGE_TEXT_LIMIT = 400
MAX_LIMIT = 60


def _iso(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")


class EidolonTools:
    """Tool implementations over the search index."""

    def __init__(self, *, search_db: Path, scout_db: Path, writable: bool) -> None:
        self._search_db = search_db
        self._scout_db = scout_db
        self._writable = writable
        self._search = SearchDatabase(search_db)
        self._search.connect(read_only=True)
        self._embedder: Any | None = None

    async def _embed(self, text: str) -> list[float] | None:
        """Embed a query, returning None when the semantic lane is unavailable.

        A missing API key or a provider outage must degrade to lexical-only
        search rather than failing the call: a keyword answer is worth much more
        to the caller than an error, and the response says which lanes ran.
        """
        if not settings.openai_api_key:
            return None
        try:
            if self._embedder is None:
                from pipeline.indexer import EmbeddingIndexer

                self._embedder = EmbeddingIndexer(self._search)
            return await self._embedder.embed_query(text)
        except Exception as exc:
            logger.warning("Query embedding failed, falling back to lexical only: %s", exc)
            return None

    # ------------------------------------------------------------------

    async def search_messages(
        self,
        query: str,
        *,
        keywords: list[str] | None = None,
        chat_ids: list[int] | None = None,
        days: int | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 15,
        semantic: bool = True,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, MAX_LIMIT))
        if since is None and days:
            since = _iso(days)
        terms = keywords or query.split()
        match_query = build_fts_query(terms) if terms else None
        vector = await self._embed(query) if semantic else None

        rows = self._search.hybrid_search(
            match_query=match_query,
            query_vector=vector,
            limit=limit,
            chat_ids=chat_ids,
            since=since,
            until=until,
        )
        lanes = [
            n for n, on in (("lexical", bool(match_query)), ("semantic", vector is not None)) if on
        ]
        return {
            "query": query,
            "lanes_used": lanes,
            "fts_expression": match_query,
            "result_count": len(rows),
            "results": [r.as_dict(text_limit=MESSAGE_TEXT_LIMIT) for r in rows],
            "note": (
                "Russian is inflected: keywords are matched as prefixes, so pass stems "
                "(концерт, площадк) rather than full forms. Venue brand names written with "
                "stylized letters (SYNCHØUSE) will not match here — use search_places for those."
            ),
        }

    async def search_places(
        self,
        *,
        name: str | None = None,
        city: str | None = None,
        place_type: str | None = None,
        event_type: str | None = None,
        min_mentions: int = 1,
        limit: int = 25,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, MAX_LIMIT))
        rows = self._search.search_places(
            name_query=name,
            city_area=city,
            place_type=place_type,
            event_types=[event_type] if event_type else None,
            min_mentions=min_mentions,
            limit=limit,
        )
        status = self._search.status()
        return {
            "result_count": len(rows),
            "places": rows,
            "index_coverage": {
                "places_known": status["places"],
                "messages_awaiting_extraction": status["extraction_backlog"],
            },
        }

    async def list_chats(self) -> dict[str, Any]:
        chats = self._search.chats()
        return {
            "chat_count": len(chats),
            "chats": chats,
            "modes": {
                "monitor": "live pipeline and alerts",
                "recon": "stored, no alerts",
                "history-only": "backfilled history, not currently observed",
            },
        }

    async def chat_history(
        self,
        chat_id: int,
        *,
        days: int | None = 7,
        since: str | None = None,
        until: str | None = None,
        limit: int = 40,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, MAX_LIMIT))
        if since is None and days:
            since = _iso(days)
        rows = self._search.recent(chat_id=chat_id, since=since, until=until, limit=limit)
        return {
            "chat_id": chat_id,
            "window": {"since": since, "until": until},
            "result_count": len(rows),
            "messages": [r.as_dict(text_limit=MESSAGE_TEXT_LIMIT) for r in rows],
        }

    async def corpus_status(self) -> dict[str, Any]:
        status = self._search.status()
        stale = None
        newest = status.get("newest_message")
        if newest:
            try:
                delta = datetime.now(UTC) - datetime.fromisoformat(
                    newest.replace(" ", "T")
                ).astimezone(UTC)
                stale = round(delta.total_seconds() / 3600, 1)
            except ValueError:
                stale = None
        status["hours_since_newest_message"] = stale
        status["warning"] = (
            "Index is behind the live daemon; results may miss recent messages."
            if stale is not None and stale > 6
            else None
        )
        return status

    async def find_chat_candidates(
        self, *, min_mentions: int = 2, limit: int = 25, include_known: bool = False
    ) -> dict[str, Any]:
        rows = self._search.chat_references(
            only_unknown=not include_known,
            min_mentions=min_mentions,
            limit=max(1, min(limit, MAX_LIMIT)),
        )
        return {
            "result_count": len(rows),
            "candidates": rows,
            "note": (
                "These are Telegram references already present in downloaded messages. "
                "Collecting them costs no Telegram action budget. A high mention count "
                "means many people in the monitored chats linked it."
            ),
        }

    async def queue_chat_join(
        self, chat_ref: str, *, label: str | None = None, target_days: int = 730
    ) -> dict[str, Any]:
        """Ask the daemon to join a chat. It decides when, not the caller."""
        if not self._writable:
            raise PermissionError("this bridge runs read-only; joining is not available")
        # The daemon stores queue rows under normalize_username(); anything
        # else here -- a trailing slash, a ?start= suffix, different case --
        # becomes a second row for the same chat and a second join attempt,
        # which is exactly the budget this queue exists to protect.
        ref = normalize_username(chat_ref)
        if not ref:
            raise ValueError("chat_ref is empty")

        with sqlite3.connect(f"file:{self._scout_db}", uri=True, timeout=15.0) as conn:
            conn.execute("PRAGMA busy_timeout=15000")
            existing = conn.execute(
                "SELECT state, joined_chat_id FROM join_queue WHERE chat_ref = ?", (ref,)
            ).fetchone()
            if existing:
                return {
                    "chat_ref": ref,
                    "queued": False,
                    "already": existing[0],
                    "joined_chat_id": existing[1],
                }
            conn.execute(
                """
                INSERT INTO join_queue (chat_ref, label, target_days, state)
                VALUES (?, ?, ?, 'pending')
                """,
                (ref, label, target_days),
            )
            conn.commit()
        return {
            "chat_ref": ref,
            "queued": True,
            "note": (
                "Queued. The daemon joins at most a few chats a day, deliberately: join "
                "rate is what gets a Telegram account limited. History download starts "
                "automatically once the join succeeds. Check back with corpus_status."
            ),
        }


READ_TOOLS = [
    Tool(
        name="search_messages",
        description=(
            "Full-text and semantic search across every Telegram message Eidolon has "
            "collected — live monitoring plus downloaded history, back to 2024. Use for "
            "open questions about what people said, discussed, recommended or announced. "
            "Pass `keywords` as word STEMS for Russian (концерт, not концерта)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of what to find. Used for semantic search.",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Word stems for exact matching, matched as prefixes and OR-ed. "
                        "Include both Russian and English forms: the corpus is ~96% "
                        "Cyrillic but event posts often use English party vocabulary."
                    ),
                },
                "chat_ids": {"type": "array", "items": {"type": "integer"}},
                "days": {"type": "integer", "description": "Only messages from the last N days."},
                "since": {"type": "string", "description": "ISO date lower bound."},
                "until": {"type": "string", "description": "ISO date upper bound."},
                "limit": {"type": "integer", "default": 15, "maximum": 60},
                "semantic": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_places",
        description=(
            "Search venues and physical places extracted from the message corpus — bars, "
            "cafes, rooftops, studios, community spaces — with what happens at each and a "
            "verbatim quote as evidence. Use this, not search_messages, when the question "
            "is about WHERE something could happen or where events are held."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Venue name; tolerant of typos and stylized spelling.",
                },
                "city": {"type": "string", "description": "e.g. Da Nang, Hoi An"},
                "place_type": {
                    "type": "string",
                    "description": "bar, cafe, restaurant, rooftop, club, hotel, studio, yoga, gallery, coworking, community_space, beach_club, hostel, theatre",
                },
                "event_type": {
                    "type": "string",
                    "description": "concert, live_music, dj_set, open_mic, jam, party, festival, quiz, board_games, film_screening, meetup, workshop, lecture, market, yoga, meditation, sound_healing, ecstatic_dance, retreat",
                },
                "min_mentions": {"type": "integer", "default": 1},
                "limit": {"type": "integer", "default": 25, "maximum": 60},
            },
        },
    ),
    Tool(
        name="list_chats",
        description=(
            "List every Telegram chat in the corpus with message counts and date coverage, "
            "and whether it is actively monitored or only has downloaded history."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="chat_history",
        description=(
            "Read messages from one chat in a time window, newest first. Use after "
            "list_chats when the question is about what a specific chat has been discussing."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "chat_id": {"type": "integer"},
                "days": {"type": "integer", "default": 7},
                "since": {"type": "string"},
                "until": {"type": "string"},
                "limit": {"type": "integer", "default": 40, "maximum": 60},
            },
            "required": ["chat_id"],
        },
    ),
    Tool(
        name="corpus_status",
        description=(
            "Index health: how many messages and venues are indexed, the date range "
            "covered, and how far behind the live daemon the index is. Check this before "
            "reporting that something does not exist — an empty result can mean a stale "
            "index rather than an absent fact."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="find_chat_candidates",
        description=(
            "Telegram chats and channels referenced inside collected messages that the "
            "account has not joined. The cheapest source of new chats to monitor: these "
            "links were already downloaded and cost no Telegram budget to collect."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "min_mentions": {"type": "integer", "default": 2},
                "limit": {"type": "integer", "default": 25, "maximum": 60},
                "include_known": {"type": "boolean", "default": False},
            },
        },
    ),
]

WRITE_TOOLS = [
    Tool(
        name="queue_chat_join",
        description=(
            "Queue a public chat for the daemon to join, after which its history is "
            "downloaded and it is monitored. This is a request, not an immediate action: "
            "the daemon joins a few chats a day at most, because join rate is what gets a "
            "Telegram account limited. Safe to call and idempotent per chat."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "chat_ref": {
                    "type": "string",
                    "description": "Public username or t.me link, e.g. danangevents or https://t.me/danangevents",
                },
                "label": {"type": "string", "description": "Human-readable name for the queue."},
                "target_days": {
                    "type": "integer",
                    "default": 730,
                    "description": "How far back to download history.",
                },
            },
            "required": ["chat_ref"],
        },
    ),
]


def build_server(tools: EidolonTools, *, writable: bool) -> Server:
    server: Server = Server("eidolon")
    catalog = READ_TOOLS + (WRITE_TOOLS if writable else [])

    # The MCP SDK's registration decorators are untyped; the handlers below are
    # annotated, so the ignore is scoped to the decorator itself.
    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[Tool]:
        return catalog

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = getattr(tools, name, None)
        if handler is None or name not in {t.name for t in catalog}:
            raise ValueError(f"unknown tool: {name}")
        args = dict(arguments or {})
        positional: list[Any] = []
        for key in ("query", "chat_id", "chat_ref"):
            if key in args:
                positional.append(args.pop(key))
        result = await handler(*positional, **args)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=1))]

    return server


async def serve(profile: str) -> None:
    writable = profile == "full"
    tools = EidolonTools(
        search_db=settings.search_db_path,
        scout_db=settings.scout_db_path,
        writable=writable,
    )
    server = build_server(tools, writable=writable)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="eidolon",
                server_version="1.0.0",
                capabilities=ServerCapabilities(tools=ToolsCapability(listChanged=False)),
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "readonly"), default="readonly")
    args = parser.parse_args()
    # stdout is the JSON-RPC channel; anything printed there corrupts the
    # protocol, so every log line goes to stderr.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    asyncio.run(serve(args.profile))
    return 0


if __name__ == "__main__":
    sys.exit(main())
