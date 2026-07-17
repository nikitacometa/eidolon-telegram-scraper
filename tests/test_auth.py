"""Tests for secure Telethon session persistence."""

import stat
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import auth
from config.settings import Settings


def test_save_session_replaces_value_and_sets_private_mode(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=keep-me\nTELEGRAM_SESSION_STRING=old-session\nDEBUG_ECHO=false\n",
        encoding="utf-8",
    )
    env_path.chmod(0o644)

    auth.save_session("new-session", env_path)

    content = env_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=keep-me" in content
    assert "DEBUG_ECHO=false" in content
    assert "old-session" not in content
    assert content.count("TELEGRAM_SESSION_STRING=new-session") == 1
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_main_saves_session_without_exposing_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_string = "sensitive-session-value"

    async def generate_session() -> str:
        return session_string

    save_session = Mock()
    monkeypatch.setattr(auth, "generate_session", generate_session)
    monkeypatch.setattr(auth, "save_session", save_session)
    monkeypatch.setattr(sys, "argv", ["auth.py"])
    monkeypatch.setattr(auth.settings, "telegram_api_id", 123)
    monkeypatch.setattr(auth.settings, "telegram_api_hash", "hash")
    monkeypatch.setattr(auth.settings, "telegram_phone", "+10000000000")

    auth.main()

    output = capsys.readouterr().out
    assert session_string not in output
    save_session.assert_called_once_with(session_string)


def test_console_entrypoint_validates_credentials(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["auth.py"])
    monkeypatch.setattr(auth.settings, "telegram_api_id", 0)
    monkeypatch.setattr(auth.settings, "telegram_api_hash", "")
    monkeypatch.setattr(auth.settings, "telegram_phone", "")

    with pytest.raises(SystemExit) as exit_info:
        auth.main()

    assert exit_info.value.code == 1
    assert "TELEGRAM_API_ID" in capsys.readouterr().out


def test_blank_template_values_use_typed_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("TELEGRAM_API_ID", "PANTHEON_CHAT_ID", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TELEGRAM_API_ID=\nPANTHEON_CHAT_ID=\nOPENAI_API_KEY=\n",
        encoding="utf-8",
    )

    loaded = Settings(_env_file=env_path)

    assert loaded.telegram_api_id == 0
    assert loaded.pantheon_chat_id == 0
    assert loaded.openai_api_key == ""
