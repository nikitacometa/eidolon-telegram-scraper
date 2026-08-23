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
from storage.search import SearchDatabase, build_fts_query, content_terms

logger = logging.getLogger("eidolon.mcp")

# Results are read by a model with a finite context. A long answer is not a
# better answer here: it is the same answer with the top of it truncated by
# OpenClaw's context manager before the model ever sees it.
MESSAGE_TEXT_LIMIT = 400
MAX_LIMIT = 60


def supplied_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Drop keys whose value is null, so a default can apply.

    Models routinely send `null` for arguments they are not using, and the tool
    schemas accept it. Passing it through would mean "the value is None", which
    reaches min(None, 60) and fails a call the model had every right to make.
    """
    return {key: value for key, value in (arguments or {}).items() if value is not None}


def _iso(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")


class EidolonTools:
    """Tool implementations over the search index."""

    def __init__(
        self,
        *,
        search_db: Path,
        scout_db: Path,
        writable: bool,
        live_db: Path | None = None,
    ) -> None:
        self._search_db = search_db
        self._scout_db = scout_db
        self._live_db = live_db if live_db is not None else settings.db_path
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
        # Explicit keywords are taken as given; a natural-language query is
        # mined for its content words rather than split, because ORing "в" and
        # "где" into an FTS query ranks the whole corpus above the answer.
        terms = keywords if keywords else content_terms(query)
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
        query: str | None = None,
        *,
        name: str | None = None,
        city: str | None = None,
        place_type: str | None = None,
        event_type: str | None = None,
        entity_kind: str | None = None,
        access_mode: str | None = None,
        min_mentions: int = 1,
        limit: int = 25,
        include_contacts: bool = True,
        semantic: bool = True,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, MAX_LIMIT))
        query = query.strip() if query and query.strip() else None
        name = name.strip() if name and name.strip() else None
        semantic_requested = bool(query and semantic and settings.place_semantic_enabled)
        vector = await self._embed(query) if query and semantic_requested else None
        rows = self._search.search_places(
            name_query=name,
            query=query,
            city_area=city,
            place_type=place_type,
            event_types=[event_type] if event_type else None,
            entity_kind=entity_kind,
            access_mode=access_mode,
            min_mentions=min_mentions,
            limit=limit,
            include_contacts=include_contacts,
            expanded_fts=settings.place_expanded_fts_enabled,
            semantic_enabled=semantic_requested,
            query_vector=vector,
            embedding_model=settings.embedding_model,
            semantic_cutoff=settings.place_semantic_cutoff,
        )
        status = self._search.status()
        lanes = sorted({lane for row in rows for lane in row["matched_via"]})
        return {
            "result_count": len(rows),
            "places": rows,
            "lanes_used": lanes,
            "semantic_available": vector is not None,
            "descriptor_embedding_backlog": status["descriptor_embedding_backlog"],
            "active_prompt_version": status["active_prompt_version"],
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

    async def resume_chat(
        self, chat_id: int, *, mode: str = "monitor", watcher_name: str | None = None
    ) -> dict[str, Any]:
        """Put an already-joined chat back under observation.

        Without ``watcher_name`` a monitored chat is bound to every policy its
        neighbours use, which is right for one city and wrong for two: a
        Phangan chat resumed that way ended up classified by the Da Nang
        events policy. Naming the policy adds that one and nothing else;
        bindings the chat already has are left as they are, because a chat
        legitimately answers to several policies and an LLM-reachable call
        should not be the thing that silently drops one.
        """
        if not self._writable:
            raise PermissionError(
                "this bridge runs read-only; changing observation is not available"
            )
        if mode not in {"monitor", "recon", "paused"}:
            raise ValueError(f"mode must be monitor, recon or paused; got {mode!r}")
        if watcher_name is not None:
            if mode != "monitor":
                raise ValueError("watcher_name only makes sense with mode='monitor'")
            known = self.known_watcher_names()
            if watcher_name not in known:
                raise ValueError(f"unknown watcher {watcher_name!r}; known: {known}")

        with sqlite3.connect(f"file:{self._live_db}", uri=True, timeout=15.0) as conn:
            conn.execute("PRAGMA busy_timeout=15000")
            title_row = conn.execute(
                "SELECT title FROM chats WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            conn.execute(
                """
                INSERT INTO observed_chats (chat_id, mode, title, source)
                VALUES (?, ?, ?, 'manual')
                ON CONFLICT(chat_id) DO UPDATE SET
                    mode = excluded.mode,
                    title = COALESCE(excluded.title, observed_chats.title),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, mode, title_row[0] if title_row else None),
            )
            watchers: list[str] = []
            if mode == "monitor":
                if watcher_name is not None:
                    watchers = [watcher_name]
                else:
                    watchers = [
                        str(r[0])
                        for r in conn.execute(
                            "SELECT DISTINCT watcher_name FROM chat_policy_bindings"
                        )
                    ]
                conn.executemany(
                    "INSERT INTO chat_policy_bindings (chat_id, watcher_name, source) "
                    "VALUES (?, ?, 'manual') ON CONFLICT(chat_id, watcher_name) DO NOTHING",
                    [(chat_id, name) for name in watchers],
                )
            # Routing lives in the daemon's memory; a row nobody reloads is a
            # change that did not happen.
            conn.execute("UPDATE agent_watchers_meta SET generation = generation + 1 WHERE id = 1")
            conn.commit()
            # Report what the chat answers to now, not only what this call added.
            if mode == "monitor":
                watchers = [
                    str(r[0])
                    for r in conn.execute(
                        "SELECT watcher_name FROM chat_policy_bindings WHERE chat_id = ? "
                        "ORDER BY watcher_name",
                        (chat_id,),
                    )
                ]
        return {
            "chat_id": chat_id,
            "mode": mode,
            "watchers": watchers,
            "note": (
                "The daemon picks this up within about half a minute, without restarting. "
                "Messages sent before now are not recovered by this — use backfill_chat for those."
            ),
        }

    async def backfill_chat(
        self, chat_id: int, *, days: int = 730, label: str | None = None
    ) -> dict[str, Any]:
        """Ask the daemon to walk a joined chat's history backwards."""
        if not self._writable:
            raise PermissionError("this bridge runs read-only; backfill is not available")
        if not 1 <= days <= 3650:
            raise ValueError("days must be between 1 and 3650")
        if not settings.backfill_enabled:
            # The row would sit pending forever: main.py only starts
            # BackfillWorker when this is on. Reporting success for work with no
            # worker is the failure this whole system keeps producing.
            raise RuntimeError(
                "history backfill is switched off on this deployment "
                "(BACKFILL_ENABLED=false), so a request would never be executed"
            )

        with sqlite3.connect(f"file:{self._scout_db}", uri=True, timeout=15.0) as conn:
            conn.execute("PRAGMA busy_timeout=15000")
            existing = conn.execute(
                "SELECT state, messages_stored, target_days FROM backfill_targets WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO backfill_targets (chat_id, label, target_days)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    label = COALESCE(excluded.label, backfill_targets.label),
                    target_days = MAX(backfill_targets.target_days, excluded.target_days),
                    state = CASE
                        WHEN excluded.target_days > backfill_targets.target_days
                            AND backfill_targets.state = 'complete'
                        THEN 'pending'
                        ELSE backfill_targets.state
                    END,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, label, days),
            )
            conn.commit()
        return {
            "chat_id": chat_id,
            "target_days": days,
            "previous": (
                {"state": existing[0], "messages_stored": existing[1], "target_days": existing[2]}
                if existing
                else None
            ),
            "note": (
                "History is walked one page of 100 messages every 8 seconds, only while the "
                "daemon is otherwise idle — roughly 45,000 messages an hour. A large chat takes "
                "tens of minutes; progress is a cursor, so nothing is lost to a restart. "
                "Asking again with the same or fewer days does nothing."
            ),
        }

    def known_watcher_names(self) -> list[str]:
        """Policies a joined chat can be bound to: the config file plus stored ones."""
        from config.watchers import WatcherConfigError, load_watchers

        names: list[str] = []
        try:
            names.extend(watcher.name for watcher in load_watchers(settings.watchers_path))
        except WatcherConfigError:
            logger.warning("watchers config unreadable; only stored watchers are known")
        try:
            with sqlite3.connect(f"file:{self._live_db}?mode=ro", uri=True, timeout=15.0) as conn:
                rows = conn.execute(
                    "SELECT name FROM agent_watchers WHERE status = 'active' ORDER BY name"
                ).fetchall()
        except sqlite3.Error:
            rows = []
        names.extend(row[0] for row in rows if row[0] not in names)
        return names

    async def queue_chat_join(
        self,
        chat_ref: str,
        *,
        label: str | None = None,
        target_days: int = 730,
        watcher_name: str | None = None,
    ) -> dict[str, Any]:
        """Ask the daemon to join a chat. It decides when, not the caller.

        ``watcher_name`` decides what the join turns into. With a policy the
        chat is monitored and alerts; without one it is only archived for
        search, which is silent by design and must be said so in the reply.
        """
        if not self._writable:
            raise PermissionError("this bridge runs read-only; joining is not available")
        if not settings.join_queue_enabled:
            raise RuntimeError(
                "the join queue is switched off on this deployment "
                "(JOIN_QUEUE_ENABLED=false), so a request would never be executed"
            )
        # The daemon stores queue rows under normalize_username(); anything
        # else here -- a trailing slash, a ?start= suffix, different case --
        # becomes a second row for the same chat and a second join attempt,
        # which is exactly the budget this queue exists to protect.
        ref = normalize_username(chat_ref)
        if not ref:
            raise ValueError("chat_ref is empty")
        if watcher_name is not None:
            known = self.known_watcher_names()
            if watcher_name not in known:
                raise ValueError(
                    f"unknown watcher {watcher_name!r}; a chat bound to a policy nobody "
                    f"runs would look monitored and never alert. Known: {known}"
                )

        with sqlite3.connect(f"file:{self._scout_db}", uri=True, timeout=15.0) as conn:
            conn.execute("PRAGMA busy_timeout=15000")
            existing = conn.execute(
                "SELECT state, joined_chat_id, watcher_name FROM join_queue WHERE chat_ref = ?",
                (ref,),
            ).fetchone()
            if existing:
                state, joined_chat_id, bound = existing
                reply: dict[str, Any] = {
                    "chat_ref": ref,
                    "queued": False,
                    "already": state,
                    "joined_chat_id": joined_chat_id,
                    "watcher_name": bound,
                }
                # A repeat with a policy is a request to upgrade, not a
                # duplicate: a pending row takes the policy for its join, a
                # joined chat is bound to it now.
                if watcher_name is not None and watcher_name != bound:
                    if state == "pending":
                        conn.execute(
                            "UPDATE join_queue SET watcher_name = ?, updated_at = CURRENT_TIMESTAMP "
                            "WHERE chat_ref = ?",
                            (watcher_name, ref),
                        )
                        conn.commit()
                        reply["watcher_name"] = watcher_name
                        reply["monitoring"] = f"will alert through {watcher_name!r} once joined"
                    elif state == "joined" and joined_chat_id is not None:
                        bound_now = await self.resume_chat(
                            int(joined_chat_id), mode="monitor", watcher_name=watcher_name
                        )
                        conn.execute(
                            "UPDATE join_queue SET watcher_name = ?, updated_at = CURRENT_TIMESTAMP "
                            "WHERE chat_ref = ?",
                            (watcher_name, ref),
                        )
                        conn.commit()
                        reply["watcher_name"] = watcher_name
                        reply["monitoring"] = (
                            f"already joined; now bound to {bound_now['watchers']}, "
                            "earlier bindings kept"
                        )
                return reply
            conn.execute(
                """
                INSERT INTO join_queue (chat_ref, label, target_days, watcher_name, state)
                VALUES (?, ?, ?, ?, 'pending')
                """,
                (ref, label, target_days, watcher_name),
            )
            conn.commit()
        return {
            "chat_ref": ref,
            "queued": True,
            "watcher_name": watcher_name,
            "monitoring": (
                f"will alert through {watcher_name!r}"
                if watcher_name
                else "archive only: searchable, but no alerts until resume_chat(mode='monitor')"
            ),
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
                # anyOf, not type: ["array", "null"]. OpenClaw compiles tool
                # schemas with TypeBox 1.3.3, whose code for a typed union with
                # `items` runs `.every` on null; the model passes null for unused
                # arrays, and the call died with "Cannot read properties of null".
                "keywords": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "description": (
                        "Word stems for exact matching, matched as prefixes and OR-ed. "
                        "Include both Russian and English forms: the corpus is ~96% "
                        "Cyrillic but event posts often use English party vocabulary."
                    ),
                },
                "chat_ids": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "integer"}},
                        {"type": "null"},
                    ],
                },
                "days": {
                    "type": ["integer", "null"],
                    "description": "Only messages from the last N days.",
                },
                "since": {"type": ["string", "null"], "description": "ISO date lower bound."},
                "until": {"type": ["string", "null"], "description": "ISO date upper bound."},
                "limit": {"type": ["integer", "null"], "default": 15, "maximum": 60},
                "semantic": {"type": ["boolean", "null"], "default": True},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_places",
        description=(
            "Search named places, people, organizations, and local service providers with "
            "verbatim source evidence. `name` constrains a brand/person identity; `query` "
            "is an open category, offering, service, or activity. Each entity "
            "carries `contacts` (handles, phones, Instagram, map links published in the "
            "messages that name it) and `posted_by` (who announces it, ranked by how often)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": ["string", "null"],
                    "description": "Open category, offering, service, or activity query.",
                },
                "name": {
                    "type": ["string", "null"],
                    "description": "Brand or person name; tolerant of stylized spelling.",
                },
                "city": {"type": ["string", "null"], "description": "e.g. Da Nang, Hoi An"},
                "place_type": {
                    "type": ["string", "null"],
                    "description": "bar, cafe, restaurant, rooftop, club, hotel, studio, yoga, gallery, coworking, community_space, beach_club, hostel, theatre",
                },
                "event_type": {
                    "type": ["string", "null"],
                    "description": "concert, live_music, dj_set, open_mic, jam, party, festival, quiz, board_games, film_screening, meetup, workshop, lecture, market, yoga, meditation, sound_healing, ecstatic_dance, retreat",
                },
                "entity_kind": {
                    "type": ["string", "null"],
                    "enum": ["place", "person", "organization", None],
                },
                "access_mode": {
                    "type": ["string", "null"],
                    "enum": ["visit", "house_call", "delivery", "remote", "unknown", None],
                },
                "min_mentions": {"type": ["integer", "null"], "default": 1},
                "limit": {"type": ["integer", "null"], "default": 25, "maximum": 60},
                "include_contacts": {
                    "type": ["boolean", "null"],
                    "default": True,
                    "description": "Set false only when scanning many places and contacts are not needed.",
                },
                "semantic": {
                    "type": ["boolean", "null"],
                    "default": True,
                    "description": "Allow the descriptor semantic lane when deployment enables it.",
                },
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
                "days": {"type": ["integer", "null"], "default": 7},
                "since": {"type": ["string", "null"]},
                "until": {"type": ["string", "null"]},
                "limit": {"type": ["integer", "null"], "default": 40, "maximum": 60},
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
                "min_mentions": {"type": ["integer", "null"], "default": 2},
                "limit": {"type": ["integer", "null"], "default": 25, "maximum": 60},
                "include_known": {"type": ["boolean", "null"], "default": False},
            },
        },
    ),
]

WRITE_TOOLS = [
    Tool(
        name="resume_chat",
        description=(
            "Put a chat the account is ALREADY in back under live monitoring, or pause it. "
            "Use when list_chats shows a chat as 'history-only' but you want new messages "
            "watched again. Takes effect within about half a minute. Does not fetch old "
            "messages — pair it with backfill_chat for those."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "chat_id": {"type": "integer", "description": "From list_chats."},
                "mode": {
                    "type": ["string", "null"],
                    "enum": ["monitor", "recon", "paused", None],
                    "description": (
                        "monitor runs the full pipeline and can alert; recon stores messages "
                        "with no alerts and no LLM cost; paused stops both. Defaults to monitor."
                    ),
                },
                "watcher_name": {
                    "type": ["string", "null"],
                    "description": (
                        "Policy to bind when monitoring, e.g. danang-signal. Omitted, the chat "
                        "is bound to every policy its neighbours use, which mixes cities."
                    ),
                },
            },
            "required": ["chat_id"],
        },
    ),
    Tool(
        name="backfill_chat",
        description=(
            "Walk an already-joined chat's history backwards and store it. Use to fill a gap "
            "for a chat that has been joined longer than it has been monitored. Runs in the "
            "background at roughly 45,000 messages an hour, only while the daemon is idle, so "
            "a big chat takes tens of minutes. Safe to call twice; asking for the same or "
            "fewer days does nothing."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "chat_id": {"type": "integer", "description": "From list_chats."},
                "days": {
                    "type": ["integer", "null"],
                    "description": "How far back to reach, in days. Defaults to 730.",
                },
                "label": {"type": ["string", "null"]},
            },
            "required": ["chat_id"],
        },
    ),
    Tool(
        name="queue_chat_join",
        description=(
            "Queue a chat for the daemon to join, after which its history is downloaded "
            "(730 days by default) and it becomes searchable. Pass `watcher_name` to also "
            "monitor it for alerts; without one the chat is archived silently. This is a "
            "request, not an immediate action: the daemon joins a few chats a day at most, "
            "because join rate is what gets a Telegram account limited. Safe to call and "
            "idempotent per chat."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "chat_ref": {
                    "type": "string",
                    "description": (
                        "Public username, t.me link or invite link, e.g. danangevents, "
                        "https://t.me/danangevents or https://t.me/+AbCdEf"
                    ),
                },
                "label": {
                    "type": ["string", "null"],
                    "description": "Human-readable name for the queue.",
                },
                "target_days": {
                    "type": ["integer", "null"],
                    "default": 730,
                    "description": "How far back to download history.",
                },
                "watcher_name": {
                    "type": ["string", "null"],
                    "description": (
                        "Monitoring policy to bind the chat to once joined, e.g. "
                        "danang-signal. Omit to archive for search only, with no alerts. "
                        "An unknown name is refused."
                    ),
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
        args = supplied_arguments(arguments)
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
        live_db=settings.db_path,
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
