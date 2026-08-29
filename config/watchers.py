"""Validated watcher configuration loaded from YAML."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)
TargetIntent = Literal["offer", "seek", "other"]
DegradedPolicy = Literal["reject", "accept"]


def _default_target_intents() -> list[TargetIntent]:
    return ["offer"]


class WatcherConfigError(ValueError):
    """Raised when watcher configuration is missing or invalid."""


class WatcherRules(BaseModel):
    """Deterministic first-stage filtering rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    keywords: list[str] = Field(default_factory=list)
    keywords_negative: list[str] = Field(default_factory=list)
    min_length: int = Field(default=0, ge=0, le=100_000)
    languages: list[str] = Field(default_factory=list)

    @field_validator("keywords", "keywords_negative")
    @classmethod
    def normalize_keywords(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().lower() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))


class WatcherExamples(BaseModel):
    """Human-authored semantic examples used by the embedding stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    positive: list[str] = Field(default_factory=list, max_length=100)
    negative: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("positive", "negative")
    @classmethod
    def normalize_examples(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))


class Watcher(BaseModel):
    """A single monitoring policy.

    ``chats`` is a seed, not the source of truth. Which chats a policy actually
    watches lives in the ``observed_chats`` registry, because reconnaissance
    adds chats at runtime and this file is git-tracked and read-only to the
    daemon. Chats listed here are reconciled into the registry on startup.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    chats: list[int] = Field(default_factory=list)
    rules: WatcherRules
    description: str = ""
    alert: Literal["immediate", "digest"] = "immediate"
    # Where a matching message is sent for judgement. "pipeline" is the
    # rule/embedding/LLM relevance ladder every policy has always used;
    # "external" hands the message to a subsystem registered under the same
    # name and skips the ladder entirely. Housing is external because its
    # question is not "is this relevant" but "what does this advertisement
    # say", answered over several messages and two arrival timelines.
    dispatch: Literal["pipeline", "external"] = "pipeline"
    llm_level: int = Field(default=1, ge=1, le=3)
    target_intents: list[TargetIntent] = Field(
        default_factory=_default_target_intents,
        min_length=1,
    )
    degraded_policy: DegradedPolicy = "reject"
    prompt: str = ""
    examples: WatcherExamples = Field(default_factory=WatcherExamples)
    embedding_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("chats")
    @classmethod
    def validate_chat_ids(cls, values: list[int]) -> list[int]:
        if any(chat_id >= 0 for chat_id in values):
            raise ValueError("Telegram group/channel chat IDs must be negative")
        return list(dict.fromkeys(values))

    @field_validator("target_intents")
    @classmethod
    def unique_target_intents(
        cls,
        values: list[TargetIntent],
    ) -> list[TargetIntent]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def require_semantic_reference_source(self) -> Watcher:
        if (
            self.llm_level >= 2
            and not self.examples.positive
            and not self.rules.keywords
            and not self.prompt.strip()
        ):
            raise ValueError("llm_level >= 2 requires positive examples, keywords, or a prompt")
        return self


class WatcherFile(BaseModel):
    """Top-level watcher document."""

    model_config = ConfigDict(extra="forbid")

    watchers: list[Watcher] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_names(self) -> WatcherFile:
        names = [watcher.name for watcher in self.watchers]
        if len(names) != len(set(names)):
            raise ValueError("watcher names must be unique")
        return self


def load_watchers(config_path: Path) -> list[Watcher]:
    """Load and validate watcher configuration.

    Invalid or empty configuration is a startup error: silently running with no
    monitoring policies would create a healthy-looking but useless daemon.
    """
    if not config_path.exists():
        raise WatcherConfigError(
            f"watchers config not found: {config_path}; copy config/watchers.example.yml first"
        )

    try:
        with config_path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
        config = WatcherFile.model_validate(data)
    except ValidationError as error:
        # Pydantic's rendered message includes input values, which may contain
        # private chat IDs or monitoring objectives. Keep only schema locations
        # and machine-readable error types.
        issues = ", ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}:{issue['type']}"
            for issue in error.errors(include_input=False)
        )
        raise WatcherConfigError(f"invalid watchers config {config_path}: {issues}") from error
    except (OSError, yaml.YAMLError) as error:
        raise WatcherConfigError(
            f"invalid watchers config {config_path}: {type(error).__name__}"
        ) from error

    logger.info("Loaded %d watcher policies", len(config.watchers))
    logger.debug("Loaded watcher names: %s", [watcher.name for watcher in config.watchers])
    return config.watchers


def get_chat_watchers(watchers: list[Watcher]) -> dict[int, list[Watcher]]:
    """Build a mapping of chat_id → list of watchers monitoring that chat."""
    chat_map: dict[int, list[Watcher]] = {}
    for watcher in watchers:
        for chat_id in watcher.chats:
            chat_map.setdefault(chat_id, []).append(watcher)
    return chat_map
