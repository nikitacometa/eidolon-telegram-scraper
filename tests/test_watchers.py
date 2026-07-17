"""Tests for config/watchers.py — watcher configuration loader."""

from pathlib import Path

import pytest
import yaml

from config.watchers import (
    Watcher,
    WatcherConfigError,
    WatcherRules,
    get_chat_watchers,
    load_watchers,
)


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
                "examples": {
                    "positive": ["Villa available near Srithanu"],
                    "negative": ["Looking for a villa"],
                },
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
    """Missing monitoring policy should fail startup instead of silently doing nothing."""
    with pytest.raises(WatcherConfigError, match="watchers config not found"):
        load_watchers(Path("/nonexistent/watchers.yml"))


def test_load_empty_file(tmp_path: Path) -> None:
    """An empty YAML document should fail validation."""
    path = tmp_path / "empty.yml"
    path.write_text("")
    with pytest.raises(WatcherConfigError, match="invalid watchers config"):
        load_watchers(path)


@pytest.mark.parametrize(
    "watchers",
    [
        [],
        [
            {
                "name": "invalid-alert",
                "chats": [-100111],
                "rules": {},
                "alert": "eventually",
            }
        ],
        [
            {
                "name": "invalid-chat",
                "chats": [100111],
                "rules": {},
            }
        ],
        [
            {
                "name": "invalid-level",
                "chats": [-100111],
                "rules": {},
                "llm_level": 4,
            }
        ],
    ],
)
def test_invalid_watcher_config_fails_fast(
    tmp_path: Path,
    watchers: list[dict[str, object]],
) -> None:
    """Invalid policy values must not produce a partially configured daemon."""
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump({"watchers": watchers}))

    with pytest.raises(WatcherConfigError, match="invalid watchers config"):
        load_watchers(path)


def test_duplicate_watcher_names_fail_fast(tmp_path: Path) -> None:
    """Watcher names are stable persistence keys and therefore must be unique."""
    watcher = {
        "name": "duplicate-watcher",
        "chats": [-100111],
        "rules": {"keywords": ["villa"]},
    }
    path = tmp_path / "duplicates.yml"
    path.write_text(yaml.safe_dump({"watchers": [watcher, watcher]}))

    with pytest.raises(WatcherConfigError, match="watcher names must be unique"):
        load_watchers(path)


def test_positive_and_negative_examples_are_parsed(config_file: Path) -> None:
    """Semantic examples should remain separated for contrastive retrieval."""
    watcher = load_watchers(config_file)[0]

    assert watcher.examples.positive == ["Villa available near Srithanu"]
    assert watcher.examples.negative == ["Looking for a villa"]


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
    w = Watcher(name="minimal", chats=[-100111], rules=WatcherRules())
    assert w.alert == "immediate"
    assert w.llm_level == 1
    assert w.prompt == ""
    assert w.rules.min_length == 0
    assert w.rules.keywords == []
    assert w.examples.positive == []
    assert w.examples.negative == []
