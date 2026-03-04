"""Tests for config/watchers.py — watcher configuration loader."""

from pathlib import Path

import pytest
import yaml

from config.watchers import Watcher, WatcherRules, get_chat_watchers, load_watchers


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a temporary watchers config file."""
    config = {
        "watchers": [
            {
                "name": "test-watcher",
                "description": "Test watcher",
                "chats": [-100111, -100222],
                "rules": {
                    "keywords": ["hello", "WORLD"],
                    "keywords_negative": ["spam"],
                    "min_length": 10,
                    "languages": ["en"],
                },
                "alert": "immediate",
                "llm_level": 2,
                "prompt": "Test prompt",
            },
            {
                "name": "second-watcher",
                "chats": [-100222, -100333],
                "rules": {"keywords": ["test"]},
            },
        ]
    }
    path = tmp_path / "watchers.yml"
    path.write_text(yaml.dump(config))
    return path


def test_load_watchers(config_file: Path) -> None:
    """Should parse YAML into Watcher dataclasses."""
    watchers = load_watchers(config_file)
    assert len(watchers) == 2
    assert watchers[0].name == "test-watcher"
    assert watchers[0].llm_level == 2
    assert watchers[0].alert == "immediate"


def test_keywords_lowercased(config_file: Path) -> None:
    """Keywords should be normalized to lowercase."""
    watchers = load_watchers(config_file)
    assert "world" in watchers[0].rules.keywords
    assert "WORLD" not in watchers[0].rules.keywords


def test_load_nonexistent_file() -> None:
    """Missing config file should return empty list, not crash."""
    watchers = load_watchers(Path("/nonexistent/watchers.yml"))
    assert watchers == []


def test_load_empty_file(tmp_path: Path) -> None:
    """Empty YAML file should return empty list."""
    path = tmp_path / "empty.yml"
    path.write_text("")
    assert load_watchers(path) == []


def test_get_chat_watchers(config_file: Path) -> None:
    """get_chat_watchers should map chat IDs to their watchers."""
    watchers = load_watchers(config_file)
    chat_map = get_chat_watchers(watchers)

    # -100111 is only in test-watcher
    assert len(chat_map[-100111]) == 1
    assert chat_map[-100111][0].name == "test-watcher"

    # -100222 is in both watchers
    assert len(chat_map[-100222]) == 2
    names = {w.name for w in chat_map[-100222]}
    assert names == {"test-watcher", "second-watcher"}

    # -100333 is only in second-watcher
    assert len(chat_map[-100333]) == 1


def test_watcher_defaults() -> None:
    """Watcher with minimal config should have sensible defaults."""
    w = Watcher(name="minimal", chats=[], rules=WatcherRules())
    assert w.alert == "immediate"
    assert w.llm_level == 1
    assert w.prompt == ""
    assert w.rules.min_length == 0
    assert w.rules.keywords == []
