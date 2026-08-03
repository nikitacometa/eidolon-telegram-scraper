"""One process per Telegram session, enforced rather than documented.

Two clients on one session do not merely conflict: Telegram invalidates the
authorisation, which takes down live monitoring as well as whatever the second
process was doing. The rule has always been written down; this makes it a
mechanism, so a helper script cannot break the daemon by accident.
"""

from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)


class SessionInUseError(RuntimeError):
    """Raised when another process already holds the Telegram session."""


class SessionLock:
    """An advisory exclusive lock on a session's lock file."""

    def __init__(
        self,
        path: Path,
        *,
        owner: str = "eidolon",
        subject: str = "Telegram session",
    ) -> None:
        self.path = path
        self._owner = owner
        # What the lock protects, named for the error message. The mechanism is
        # generic; the consequence of losing the race is not, and a caller
        # guarding an index rebuild should not tell the operator its Telegram
        # session is in use.
        self._subject = subject
        self._handle: int | None = None

    def acquire(self) -> None:
        """Take the lock or explain who has it.

        The lock is advisory and process-scoped: the kernel releases it if the
        holder dies, so a crashed daemon never leaves the session unusable.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            holder = _read_holder(handle)
            os.close(handle)
            raise SessionInUseError(
                f"{self._subject} is already held by {holder or 'another process'}; "
                "stop it before starting a second client"
            ) from error

        os.ftruncate(handle, 0)
        os.write(handle, f"{self._owner} pid={os.getpid()}\n".encode())
        os.fsync(handle)
        self._handle = handle
        logger.info("%s lock acquired: %s", self._subject, self.path)

    def release(self) -> None:
        """Release the lock if this process holds it."""
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
        finally:
            os.close(self._handle)
            self._handle = None

    def __enter__(self) -> SessionLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


def _read_holder(handle: int) -> str:
    try:
        os.lseek(handle, 0, os.SEEK_SET)
        return os.read(handle, 256).decode(errors="replace").strip()
    except OSError:
        return ""
