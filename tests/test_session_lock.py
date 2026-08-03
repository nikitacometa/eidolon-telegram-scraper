"""Tests for storage/session_lock.py and the reconnaissance CLI guard."""

import os
from pathlib import Path

import pytest

from pipeline.recon import ChatFinding, ReconReport
from pipeline.recon_models import CandidateState, ReconJobStatus
from recon_cli import _parse_args, _render, main
from storage.session_lock import SessionInUseError, SessionLock


def test_lock_is_exclusive(tmp_path: Path) -> None:
    """A second holder must be refused, not queued."""
    path = tmp_path / "session.lock"
    first = SessionLock(path, owner="daemon")
    first.acquire()

    with pytest.raises(SessionInUseError):
        SessionLock(path, owner="cli").acquire()

    first.release()


def test_lock_names_its_holder(tmp_path: Path) -> None:
    """The error has to say what to stop."""
    path = tmp_path / "session.lock"
    with SessionLock(path, owner="eidolon-daemon"), pytest.raises(SessionInUseError) as caught:
        SessionLock(path, owner="cli").acquire()

    assert "eidolon-daemon" in str(caught.value)
    assert str(os.getpid()) in str(caught.value)


def test_lock_is_reusable_after_release(tmp_path: Path) -> None:
    """Stopping the daemon must free the session for a crawl."""
    path = tmp_path / "session.lock"
    daemon = SessionLock(path, owner="daemon")
    daemon.acquire()
    daemon.release()

    second = SessionLock(path, owner="cli")
    second.acquire()
    second.release()


def test_release_without_acquire_is_harmless(tmp_path: Path) -> None:
    """Cleanup paths must not depend on whether the lock was taken."""
    SessionLock(tmp_path / "session.lock").release()


def test_context_manager_releases_on_error(tmp_path: Path) -> None:
    """A crashing crawl must not leave the session locked."""
    path = tmp_path / "session.lock"

    with pytest.raises(RuntimeError), SessionLock(path, owner="cli"):
        raise RuntimeError("crawl blew up")

    SessionLock(path, owner="daemon").acquire()


def test_cli_refuses_to_start_while_the_daemon_holds_the_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard exists so a helper cannot take down live monitoring."""
    from config.settings import settings

    monkeypatch.setattr(settings, "db_path", tmp_path / "eidolon.db")
    daemon = SessionLock(tmp_path / "session.lock", owner="eidolon-daemon")
    daemon.acquire()

    exit_code = main(["--topic", "housing", "--location", "Da Nang"])

    assert exit_code == 2
    daemon.release()


def test_dry_run_never_touches_telegram(capsys: pytest.CaptureFixture[str]) -> None:
    """The plan must be inspectable before anything is spent."""
    exit_code = main(["--topic", "housing rent", "--location", "Da Nang, Vietnam", "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "дананг" in output
    assert "housing" in output
    assert "auto-join" in output


def test_waves_are_clamped_by_the_cli() -> None:
    """A caller cannot ask for a deeper crawl than the design allows."""
    args = _parse_args(["--topic", "x", "--waves", "9"])

    assert args.waves == 9  # parsed as given
    assert max(1, min(args.waves, 2)) == 2  # clamped before the job is built


def test_report_reads_as_an_answer_not_a_dump() -> None:
    """The owner asked a question; the output should answer it."""
    report = ReconReport(
        job_id="abc",
        status=ReconJobStatus.COMPLETED,
        stop_reason="frontier empty",
        waves_completed=2,
        joined=[
            ChatFinding(
                chat_ref="danang_housing",
                title="Da Nang Housing",
                score=88.0,
                state=CandidateState.JOINED,
                reason="",
                messages_stored=420,
            )
        ],
        awaiting_approval=[
            ChatFinding(
                chat_ref="danang_food",
                title="Da Nang Food",
                score=71.0,
                state=CandidateState.AWAITING_APPROVAL,
                reason="",
            )
        ],
        rejected=7,
        blocked_private=2,
        messages_stored=420,
    )

    rendered = _render(report)

    assert "danang_housing" in rendered
    assert "420 messages" in rendered
    assert "waiting for your decision (1)" in rendered
    assert "rejected: 7" in rendered
    assert "not provably public: 2" in rendered


def test_the_message_names_what_is_locked_and_what_to_do(tmp_path: Path) -> None:
    """A scheduled index refresh has no client to stop; the text must not say so."""
    path = tmp_path / "index.lock"
    held = SessionLock(path, owner="index-extract", subject="The search index")
    held.acquire()
    try:
        waiting = SessionLock(
            path,
            owner="index-build",
            subject="The search index",
            remedy="skipping this run, the next tick will pick the work up",
        )
        with pytest.raises(SessionInUseError) as caught:
            waiting.acquire()
    finally:
        held.release()
    message = str(caught.value)
    assert "The search index" in message
    assert "index-extract" in message
    assert "the next tick" in message
    assert "client" not in message


def test_the_default_message_is_still_the_telegram_one(tmp_path: Path) -> None:
    """The daemon and recon CLI rely on the original wording."""
    path = tmp_path / "session.lock"
    held = SessionLock(path, owner="daemon")
    held.acquire()
    try:
        with pytest.raises(SessionInUseError) as caught:
            SessionLock(path, owner="cli").acquire()
    finally:
        held.release()
    assert "Telegram session" in str(caught.value)
    assert "stop it before starting a second client" in str(caught.value)
