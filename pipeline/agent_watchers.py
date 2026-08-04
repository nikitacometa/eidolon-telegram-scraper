"""Watchers the assistant creates, reaching a daemon that must not restart.

The daemon owns the one Telegram session, so "add a watcher" cannot mean
"restart". It also cannot mean "write a row and hope": until this module
existed, an external write to the observation tables took effect only after the
next join or restart, which made any such tool a confident no-op.

The mechanism is a counter and a poll, the same shape as the join and backfill
workers already in this codebase. It was chosen over a socket the daemon serves
and over signal handling for one reason: an exception escaping the ingest path
terminates the process, so the safest control surface is the one that adds no
listener, no signal handler, and no new way to be woken — only a periodic read
that fails into a log line and retries.

Seeding the semantic index is part of applying a watcher, not a follow-up. A
watcher that reaches routing without its reference vectors is live and blind:
every message degrades at Level 2, and the default policy turns that into
silence — the exact failure this whole feature exists to remove.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from config.watchers import Watcher

logger = logging.getLogger(__name__)

# Agent-authored names live in their own space so a merge with the git-tracked
# policy file can never be a field-by-field reconciliation of two sources.
AGENT_NAME_PATTERN = re.compile(r"^agent-[a-z0-9][a-z0-9-]{0,54}$")

DEFAULT_POLL_SECONDS = 30.0


class AgentWatcherError(ValueError):
    """A proposed watcher was refused. The message is shown to the requester."""


def validate_agent_name(name: str) -> str:
    """Return the name if an agent is allowed to author it."""
    if not AGENT_NAME_PATTERN.match(name):
        raise AgentWatcherError(
            f"agent watcher names must match {AGENT_NAME_PATTERN.pattern!r}; got {name!r}"
        )
    return name


def parse_agent_watcher(name: str, definition_json: str) -> Watcher:
    """Parse a stored definition, refusing anything an agent may not set.

    Validation happens on read as well as on write. A row can be written by an
    older version of the tool, edited by hand, or restored from a backup, and
    the daemon is the last place that can refuse it before it starts judging
    live traffic.
    """
    try:
        payload = json.loads(definition_json)
    except json.JSONDecodeError as exc:
        raise AgentWatcherError(f"{name}: definition is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AgentWatcherError(f"{name}: definition is not an object")

    payload["name"] = validate_agent_name(name)
    # Which chats a policy watches is runtime state in observed_chats, not part
    # of a policy an agent hands over. Forcing it empty here means a malformed
    # or hostile definition cannot bind itself to chats.
    payload["chats"] = []

    try:
        watcher = Watcher.model_validate(payload)
    except ValidationError as exc:
        raise AgentWatcherError(f"{name}: {exc}") from exc

    if watcher.llm_level < 2:
        # The point of these watchers is to match on meaning. A level-1 policy
        # is a keyword list, which is the failure being replaced: measured on
        # this corpus, a curated keyword list missed 56% of real announcements.
        raise AgentWatcherError(
            f"{name}: llm_level must be at least 2; a level-1 watcher matches "
            "keywords only, which is what semantic watchers exist to replace"
        )
    return watcher


@dataclass(slots=True)
class ReconcileResult:
    """What changed in one reconciliation."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


def merge_agent_watchers(
    *,
    config_watchers: list[Watcher],
    stored: list[tuple[str, str, str]],
    current: dict[str, Watcher],
) -> tuple[dict[str, Watcher], ReconcileResult]:
    """Fold stored agent definitions into the live set.

    Config always wins a name collision, and says so in the log rather than
    quietly shadowing one policy with another.
    """
    config_names = {watcher.name for watcher in config_watchers}
    result = ReconcileResult()
    merged: dict[str, Watcher] = {}

    for name, definition_json, _created_by in stored:
        if name in config_names:
            logger.warning(
                "Agent watcher %s collides with a config watcher; the config one wins", name
            )
            result.rejected.append(name)
            continue
        try:
            watcher = parse_agent_watcher(name, definition_json)
        except AgentWatcherError as exc:
            # One unusable row must not stop the others from loading, and must
            # not be retried noisily on every tick.
            logger.warning("Skipping agent watcher %s: %s", name, exc)
            result.rejected.append(name)
            continue
        merged[name] = watcher
        if name not in current:
            result.added.append(name)
        elif current[name] != watcher:
            result.updated.append(name)

    result.removed = sorted(set(current) - set(merged))
    return merged, result


class AgentWatcherSync:
    """Polls the revision counter and applies changes to the running daemon.

    Modelled on JoinWorker.run_forever deliberately: the loop swallows and logs
    every non-cancellation exception, so a bug here costs a stale watcher set
    until the next tick and never the process.
    """

    def __init__(
        self,
        *,
        read_generation: Callable[[], Awaitable[int]],
        reconcile: Callable[[], Awaitable[ReconcileResult]],
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self._read_generation = read_generation
        self._reconcile = reconcile
        self._poll_seconds = poll_seconds
        self._seen_generation: int | None = None

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        logger.info("Agent watcher sync started (every %.0fs)", self._poll_seconds)
        while not shutdown.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Agent watcher sync tick failed; retrying next tick")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                continue

    async def run_once(self) -> ReconcileResult | None:
        """Reconcile if the counter moved, and report what changed."""
        generation = await self._read_generation()
        if self._seen_generation == generation:
            return None
        result = await self._reconcile()
        # Only advance after a successful apply: a failed reconciliation must be
        # retried, not marked as seen.
        self._seen_generation = generation
        if result.changed:
            logger.info(
                "Agent watchers reconciled: +%s ~%s -%s (generation %d)",
                result.added,
                result.updated,
                result.removed,
                generation,
            )
        return result
