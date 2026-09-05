"""Report reconnaissance progress to Telegram — only when there is news.

Meant for an hourly timer. The join queue and the history archive advance on
their own for hours, so the owner should learn where they got to without
asking; but a report that says "44 joined, 0 queued" twice a day for a month
is noise. The decision of whether to send lives in pipeline/recon_status.py as
a pure state machine over two snapshots: the current one, and the one stored in
`data/recon_status_state.json` when the last report went out. Reads state only;
it never touches Telegram's MTProto API.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from pipeline.recon_models import BackfillState, JoinQueueState  # noqa: E402
from pipeline.recon_status import ReconSnapshot, ReconStatusState, decide  # noqa: E402
from storage.scout import ScoutDatabase  # noqa: E402

TELEGRAM_API = "https://api.telegram.org"
STATE_PATH = settings.scout_db_path.parent / "recon_status_state.json"


def _send(text: str) -> bool:
    token = settings.eidolon_bot_token or settings.pantheon_bot_token
    chat_id = settings.pantheon_chat_id
    if not token or not chat_id:
        print(text)
        return True
    payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    # The URL is a fixed https constant plus the bot token; no user input
    # reaches the scheme.
    request = urllib.request.Request(  # noqa: S310
        f"{TELEGRAM_API}/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            response.read()
    except urllib.error.URLError as error:
        # A failed status report must not look like a failed crawl — and must
        # not advance the stored state, or the report is lost for good.
        print(f"status delivery failed: {error}", file=sys.stderr)
        return False
    return True


async def snapshot() -> ReconSnapshot:
    """Read the queue and the archive as they stand."""
    scout = ScoutDatabase(settings.scout_db_path)
    await scout.connect()
    try:
        queue = await scout.join_queue()
        targets = await scout.backfill_targets()
        cursor = await scout.conn.execute("SELECT COUNT(*) FROM scout_messages")
        row = await cursor.fetchone()
        stored = int(row[0]) if row is not None else 0
        cursor = await scout.conn.execute(
            "SELECT MIN(date), MAX(date) FROM scout_messages WHERE date IS NOT NULL"
        )
        span = await cursor.fetchone()
    finally:
        await scout.close()

    return ReconSnapshot(
        joined=tuple(e.chat_ref for e in queue if e.state is JoinQueueState.JOINED),
        pending=tuple(e.chat_ref for e in queue if e.state is JoinQueueState.PENDING),
        requested=tuple(e.chat_ref for e in queue if e.state is JoinQueueState.REQUESTED),
        failed=tuple(
            (e.chat_ref, e.last_error or "") for e in queue if e.state is JoinQueueState.FAILED
        ),
        backfill_pending=sum(1 for t in targets if t.state is BackfillState.PENDING),
        backfill_done=sum(1 for t in targets if t.state is not BackfillState.PENDING),
        messages_stored=stored,
        oldest=str(span[0]) if span and span[0] else None,
        newest=str(span[1]) if span and span[1] else None,
    )


def load_state() -> ReconStatusState:
    """The state stored when the last report went out; idle when none."""
    try:
        return ReconStatusState.from_dict(json.loads(STATE_PATH.read_text()))
    except (OSError, ValueError):
        return ReconStatusState()


def save_state(state: ReconStatusState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state.as_dict(), ensure_ascii=False, indent=2))


def main() -> int:
    """Decide, send if there is news, store the state the report was made on."""
    now = datetime.now(UTC)
    current = asyncio.run(snapshot())
    decision = decide(current, load_state(), now=now)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    if decision.message is None:
        print(f"{stamp} silent: {decision.reason} (active_work={current.active_work})")
        # The idle state carries nothing worth persisting except the phase,
        # which did not change; writing it anyway keeps the file present.
        save_state(decision.state)
        return 0
    if _send(decision.message):
        save_state(decision.state)
        print(f"{stamp} sent: {decision.reason}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
