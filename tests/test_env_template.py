"""The committed .env.example must not silently contradict the code defaults.

A deployment created from the template inherits every key it sets, so a stale
value there quietly reverts a considered decision. Measured 2026-08-03: the
classifier model was changed on evidence, deployed, and the template still
pinned the retired model plus a timeout budget belonging to the previous model
family — a fresh install would have run the old configuration while the repo
claimed otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings

# Keys whose template value is deliberately a placeholder rather than the
# default: secrets, and paths a real deployment overrides.
PLACEHOLDER_KEYS = frozenset(
    {
        "telegram_api_id",
        "telegram_api_hash",
        "telegram_session_string",
        "telegram_phone",
        "openai_api_key",
        "pantheon_bot_token",
        "pantheon_chat_id",
        "eidolon_bot_token",
        "debug_echo",
    }
)


def _template() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in Path(".env.example").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()
    return values


def test_every_template_key_is_a_real_setting() -> None:
    # A key the model does not define is dead text that reads as configuration.
    unknown = set(_template()) - set(Settings.model_fields)
    assert not unknown, f".env.example sets keys Settings does not define: {sorted(unknown)}"


@pytest.mark.parametrize(
    "field",
    ["llm_model", "extraction_model", "summary_model", "llm_engine", "embedding_model"],
)
def test_model_choices_match_the_code(field: str) -> None:
    # These are the values chosen from measurements; a template that disagrees
    # sends a fresh deployment back to a model the holdout rejected.
    template = _template()
    if field not in template:
        pytest.skip(f"{field} is not pinned in the template")
    assert template[field] == str(getattr(Settings(), field))


def test_no_template_value_contradicts_a_default() -> None:
    # Compared after the model has coerced both sides: `true` and `True` are the
    # same setting, and flagging that would train the reader to ignore this test.
    defaults = Settings()
    configured = Settings(
        **{
            key: value
            for key, value in _template().items()
            if key not in PLACEHOLDER_KEYS and key in Settings.model_fields
        }
    )
    mismatched = {
        key: (getattr(configured, key), getattr(defaults, key))
        for key in _template()
        if key not in PLACEHOLDER_KEYS
        and key in Settings.model_fields
        and getattr(configured, key) != getattr(defaults, key)
    }
    assert not mismatched, (
        "these template values differ from the code defaults (template, code); "
        f"change one or the other deliberately: {mismatched}"
    )
