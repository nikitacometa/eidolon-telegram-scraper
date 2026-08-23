"""Running discovery jobs from inside the daemon.

``recon_cli.py`` can only run while the daemon is stopped, because both need
the one Telegram session. That made "find chats about X" an operator task with
downtime attached. This worker lets the daemon itself work ``recon_jobs`` rows
that an agent queued, at the pace the governor allows.

It never joins. A discovery job here is forced to zero join attempts whatever
the row says: joining has one path, the join queue, where a chat gets a
policy and a human-visible pace, and a second path would bypass both.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from pipeline.recon import ReconRunner
from pipeline.recon_models import ReconJob, ReconJobStatus
from storage.scout import ScoutDatabase

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 60.0


class DiscoveryWorker:
    """Works queued discovery jobs one at a time."""

    def __init__(
        self,
        *,
        scout: ScoutDatabase,
        runner: ReconRunner,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self._scout = scout
        self._runner = runner
        self._poll_seconds = poll_seconds

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        """Work the queue until asked to stop."""
        logger.info("Discovery worker started")
        while not shutdown.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A discovery problem must never take the monitor down with it.
                logger.exception("Discovery cycle failed")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                continue
        logger.info("Discovery worker stopped")

    async def run_once(self) -> bool:
        """Run at most one job. Returns whether a job was picked up."""
        job = await self._scout.next_queued_job()
        if job is None:
            return False
        if job.max_join_attempts:
            logger.warning(
                "Discovery job %s asked for %d joins; running it without any — joins go "
                "through the join queue",
                job.id,
                job.max_join_attempts,
            )
        discover_only = replace(job, max_join_attempts=0)
        logger.info("Discovery job %s: %s / %s", job.id, job.topic, job.location or "-")
        try:
            report = await self._runner.run(discover_only)
        except Exception as error:
            await self._scout.update_job_status(
                job.id,
                ReconJobStatus.FAILED,
                stop_reason="worker error",
                error_code=type(error).__name__,
            )
            raise
        logger.info(
            "Discovery job %s: %s (%s); candidates=%d recommended=%d rejected=%d private=%d",
            job.id,
            report.status.value,
            report.stop_reason,
            report.total_candidates,
            len(report.recommended),
            report.rejected,
            report.blocked_private,
        )
        return True


__all__ = ["DEFAULT_POLL_SECONDS", "DiscoveryWorker", "ReconJob"]
