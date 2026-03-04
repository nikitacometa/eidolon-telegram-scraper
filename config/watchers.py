"""Watcher configuration loader — parses watchers.yml into typed dataclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class WatcherRules:
    keywords: list[str] = field(default_factory=list)
    keywords_negative: list[str] = field(default_factory=list)
    min_length: int = 0
    languages: list[str] = field(default_factory=list)


@dataclass
class Watcher:
    name: str
    chats: list[int]
    rules: WatcherRules
    description: str = ""
    alert: str = "immediate"  # "immediate" or "digest"
    llm_level: int = 1
    prompt: str = ""


def load_watchers(config_path: Path) -> list[Watcher]:
    """Load watcher configs from a YAML file."""
    if not config_path.exists():
        logger.warning("Watchers config not found: %s", config_path)
        return []

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data or "watchers" not in data:
        return []

    watchers = []
    for entry in data["watchers"]:
        rules_data = entry.get("rules", {})
        rules = WatcherRules(
            keywords=[kw.lower() for kw in rules_data.get("keywords", [])],
            keywords_negative=[kw.lower() for kw in rules_data.get("keywords_negative", [])],
            min_length=rules_data.get("min_length", 0),
            languages=rules_data.get("languages", []),
        )
        watcher = Watcher(
            name=entry["name"],
            description=entry.get("description", ""),
            chats=entry.get("chats", []),
            rules=rules,
            alert=entry.get("alert", "immediate"),
            llm_level=entry.get("llm_level", 1),
            prompt=entry.get("prompt", ""),
        )
        watchers.append(watcher)

    logger.info("Loaded %d watchers: %s", len(watchers), [w.name for w in watchers])
    return watchers


def get_chat_watchers(watchers: list[Watcher]) -> dict[int, list[Watcher]]:
    """Build a mapping of chat_id → list of watchers monitoring that chat."""
    chat_map: dict[int, list[Watcher]] = {}
    for watcher in watchers:
        for chat_id in watcher.chats:
            chat_map.setdefault(chat_id, []).append(watcher)
    return chat_map
