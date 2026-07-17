"""Tests for secure Telethon session persistence."""

import stat
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import auth


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

    auth.main()

    output = capsys.readouterr().out
    assert session_string not in output
    save_session.assert_called_once_with(session_string)
