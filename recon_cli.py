"""Run one reconnaissance job from the command line.

Intended for operating a crawl by hand: the same runner also drives jobs
submitted from an agent, and both take the session lock, so this cannot start
while the daemon is listening.

    python3 recon_cli.py --topic "housing rent" --location "Da Nang, Vietnam"
    python3 recon_cli.py --topic "housing" --location "Da Nang" --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from contextlib import AsyncExitStack

from telethon import TelegramClient
from telethon.sessions import StringSession

from config.settings import settings
from pipeline.crawler import TelegramCrawler
from pipeline.discovery import TelegramDiscovery
from pipeline.governor import TelegramActionGovernor
from pipeline.recon import ReconReport, ReconRunner
from pipeline.recon_models import ActionKind, JobRequest
from pipeline.scoring import build_policy
from storage.db import Database
from storage.scout import ScoutDatabase
from storage.session_lock import SessionInUseError, SessionLock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("recon")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Telegram reconnaissance job")
    parser.add_argument("--topic", required=True, help="what the chats should be about")
    parser.add_argument("--location", default=None, help="city or region, e.g. 'Da Nang, Vietnam'")
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        dest="seeds",
        help="public @username to start from; may be repeated",
    )
    parser.add_argument("--key", default=None, help="idempotency key; repeats resume the same job")
    parser.add_argument("--waves", type=int, default=2, help="how far to follow links (max 2)")
    parser.add_argument("--pages", type=int, default=3, help="history pages per joined chat")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what the job would search for and stop before touching Telegram",
    )
    return parser.parse_args(argv)


def _render(report: ReconReport) -> str:
    lines = [
        f"job {report.job_id}: {report.status.value} ({report.stop_reason})",
        f"waves completed: {report.waves_completed}",
        f"candidates considered: {report.total_candidates}",
        "",
    ]
    if report.joined:
        lines.append(f"joined and read ({len(report.joined)}):")
        for finding in report.joined:
            lines.append(
                f"  @{finding.chat_ref} — {finding.title or 'untitled'} "
                f"[score {finding.score:.0f}, {finding.messages_stored} messages]"
            )
        lines.append("")
    if report.awaiting_approval:
        lines.append(f"waiting for your decision ({len(report.awaiting_approval)}):")
        for finding in report.awaiting_approval:
            lines.append(
                f"  @{finding.chat_ref} — {finding.title or 'untitled'} [score {finding.score:.0f}]"
            )
        lines.append("")
    lines.append(
        f"rejected: {report.rejected} · not provably public: {report.blocked_private} · "
        f"messages stored: {report.messages_stored}"
    )
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    policy = build_policy(topic=args.topic, location=args.location)
    if args.dry_run:
        print("dry run — nothing will touch Telegram\n")
        print(f"location keywords: {', '.join(policy.location_keywords) or '(none)'}")
        print(f"topic keywords:    {', '.join(policy.topic_keywords) or '(none)'}")
        print(f"seeds:             {', '.join(args.seeds) or '(none)'}")
        print(
            f"thresholds:        auto-join {policy.auto_join_threshold:.0f}, "
            f"ask {policy.approval_threshold:.0f}"
        )
        return 0

    request = JobRequest(
        idempotency_key=args.key or f"cli:{args.topic}:{args.location}",
        topic=args.topic,
        location=args.location,
        seeds=tuple(args.seeds),
        max_waves=max(1, min(args.waves, 2)),
    )

    lock = SessionLock(settings.db_path.parent / "session.lock", owner="recon-cli")
    try:
        lock.acquire()
    except SessionInUseError as error:
        logger.error("%s", error)
        return 2

    try:
        async with AsyncExitStack() as stack:
            db = Database(settings.db_path)
            await db.connect()
            stack.push_async_callback(db.close)
            scout = ScoutDatabase(settings.scout_db_path)
            await scout.connect()
            stack.push_async_callback(scout.close)

            client = TelegramClient(
                StringSession(settings.telegram_session_string),
                settings.telegram_api_id,
                settings.telegram_api_hash,
            )
            await client.start()
            stack.push_async_callback(client.disconnect)

            governor = TelegramActionGovernor(scout=scout)
            runner = ReconRunner(
                scout=scout,
                db=db,
                client=client,
                governor=governor,
                discovery=TelegramDiscovery(client=client, governor=governor),
                crawler=TelegramCrawler(client=client, governor=governor),
                backfill_pages_per_chat=args.pages,
            )

            job = await scout.create_job(request)
            logger.info("Running job %s: %s", job.id, job.topic)
            report = await runner.run(job)
            print()
            print(_render(report))

            used_hour, used_day = await scout.budget_usage(
                account_id=governor.account_id,
                kind=ActionKind.JOIN,
            )
            print(f"\njoins used: {used_hour} this hour, {used_day} today")
    finally:
        lock.release()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the reconnaissance CLI."""
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
