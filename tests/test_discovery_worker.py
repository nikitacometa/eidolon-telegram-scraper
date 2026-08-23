"""Tests for discovery jobs worked from inside the daemon."""

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.discovery_worker import DiscoveryWorker
from pipeline.recon import ReconReport, _search_terms
from pipeline.recon_models import JobRequest, ReconJob, ReconJobStatus
from pipeline.scoring import build_policy
from storage.scout import ScoutDatabase


@pytest.fixture
async def scout(tmp_path: Path) -> AsyncIterator[ScoutDatabase]:
    database = ScoutDatabase(tmp_path / "scout.db")
    await database.connect()
    yield database
    await database.close()


class FakeRunner:
    """Records what it was asked to run and settles the job like the real one."""

    def __init__(self, scout: ScoutDatabase, *, error: Exception | None = None) -> None:
        self._scout = scout
        self.error = error
        self.jobs: list[ReconJob] = []

    async def run(self, job: ReconJob) -> ReconReport:
        self.jobs.append(job)
        if self.error is not None:
            raise self.error
        await self._scout.update_job_status(
            job.id, ReconJobStatus.COMPLETED, stop_reason="frontier empty"
        )
        return ReconReport(
            job_id=job.id, status=ReconJobStatus.COMPLETED, stop_reason="frontier empty"
        )


def _worker(scout: ScoutDatabase, runner: FakeRunner) -> DiscoveryWorker:
    return DiscoveryWorker(scout=scout, runner=runner, poll_seconds=0.01)  # type: ignore[arg-type]


async def test_a_queued_job_is_run_and_completed(scout: ScoutDatabase) -> None:
    runner = FakeRunner(scout)
    job = await scout.create_job(
        JobRequest(idempotency_key="k1", topic="expats", location="Da Lat", max_join_attempts=0)
    )

    assert await _worker(scout, runner).run_once()

    assert [j.id for j in runner.jobs] == [job.id]
    stored = await scout.get_job(job.id)
    assert stored is not None and stored.status is ReconJobStatus.COMPLETED
    assert not await _worker(scout, runner).run_once()


async def test_joins_are_stripped_before_the_runner_sees_the_job(scout: ScoutDatabase) -> None:
    """Discovery never joins: the join queue is the one path with a policy and a pace."""
    runner = FakeRunner(scout)
    await scout.create_job(
        JobRequest(idempotency_key="k2", topic="expats", location="Da Lat", max_join_attempts=6)
    )

    await _worker(scout, runner).run_once()

    assert runner.jobs[0].max_join_attempts == 0


async def test_a_job_that_is_already_running_is_left_alone(scout: ScoutDatabase) -> None:
    runner = FakeRunner(scout)
    job = await scout.create_job(JobRequest(idempotency_key="k3", topic="expats"))
    await scout.update_job_status(job.id, ReconJobStatus.DISCOVERING)

    assert not await _worker(scout, runner).run_once()
    assert runner.jobs == []


async def test_a_crashing_job_is_marked_failed_and_not_retried(scout: ScoutDatabase) -> None:
    runner = FakeRunner(scout, error=RuntimeError("telegram went away"))
    job = await scout.create_job(JobRequest(idempotency_key="k4", topic="expats"))
    worker = _worker(scout, runner)

    with pytest.raises(RuntimeError):
        await worker.run_once()

    stored = await scout.get_job(job.id)
    assert stored is not None
    assert stored.status is ReconJobStatus.FAILED
    assert stored.error_code == "RuntimeError"
    assert not await worker.run_once()


async def test_the_oldest_queued_job_goes_first(scout: ScoutDatabase) -> None:
    runner = FakeRunner(scout)
    first = await scout.create_job(JobRequest(idempotency_key="a", topic="first"))
    await scout.create_job(JobRequest(idempotency_key="b", topic="second"))

    await _worker(scout, runner).run_once()

    assert runner.jobs[0].id == first.id


def test_search_terms_cover_every_spelling_and_keep_the_topic() -> None:
    """Cyrillic and Vietnamese spellings must be searched, and the topic must survive the cap."""
    hashtags, titles = _search_terms(build_policy(topic="expats community", location="Da Lat"))

    assert hashtags == ["dalat", "dalatexpats", "далат", "đàlạt"]
    assert titles == ["da lat", "da lat expats", "dalat", "далат", "да лат", "đà lạt"]


def test_search_terms_without_a_place_fall_back_to_the_topic() -> None:
    assert _search_terms(build_policy(topic="handpan", location=None)) == (["handpan"], ["handpan"])
    # A hashtag is one token: the policy splits the topic on whitespace before
    # it gets here, so a multi-word topic never reaches the hashtag surface whole.
    assert _search_terms(build_policy(topic="expats community", location=None)) == (
        ["expats"],
        ["expats"],
    )


def test_replace_keeps_the_rest_of_the_job() -> None:
    job = ReconJob(
        id="x",
        idempotency_key="x",
        account_id="a",
        topic="t",
        status=ReconJobStatus.QUEUED,
        max_join_attempts=3,
    )
    assert replace(job, max_join_attempts=0).topic == "t"
