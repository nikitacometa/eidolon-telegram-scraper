"""Report reconnaissance progress to Telegram.

Meant for a timer: the join queue and the history archive both advance on
their own for hours, so the owner should learn where they got to without
asking. Reads state only; it never touches Telegram's MTProto API.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from pipeline.recon_models import BackfillState, JoinQueueState  # noqa: E402
from storage.scout import ScoutDatabase  # noqa: E402

TELEGRAM_API = "https://api.telegram.org"


def _send(text: str) -> None:
    token = settings.eidolon_bot_token or settings.pantheon_bot_token
    chat_id = settings.pantheon_chat_id
    if not token or not chat_id:
        print(text)
        return
    payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    request = urllib.request.Request(
        f"{TELEGRAM_API}/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    except urllib.error.URLError as error:
        # A failed status report must not look like a failed crawl.
        print(f"status delivery failed: {error}", file=sys.stderr)


def _plural(count: int, one: str, few: str, many: str) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return one
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


async def build_report() -> str:
    """Render the current state of the queue and the archive."""
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

    joined = [entry for entry in queue if entry.state is JoinQueueState.JOINED]
    pending = [entry for entry in queue if entry.state is JoinQueueState.PENDING]
    requested = [entry for entry in queue if entry.state is JoinQueueState.REQUESTED]
    failed = [entry for entry in queue if entry.state is JoinQueueState.FAILED]
    working = [t for t in targets if t.state is BackfillState.PENDING]
    done = [t for t in targets if t.state is not BackfillState.PENDING]

    lines = [
        "🛰 <b>Разведка Дананга — статус</b>",
        "",
        f"<b>Чаты:</b> вступил в {len(joined)}, "
        f"в очереди {len(pending)}, ждёт админа {len(requested)}, "
        f"не вышло {len(failed)}",
        f"<b>История:</b> {stored} "
        f"{_plural(stored, 'сообщение', 'сообщения', 'сообщений')} "
        f"из {len(targets)} {_plural(len(targets), 'чата', 'чатов', 'чатов')}",
    ]
    if span and span[0]:
        lines.append(f"<b>Глубина:</b> {str(span[0])[:10]} .. {str(span[1])[:10]}")
    lines.append(f"<b>Докачивается:</b> {len(working)}, завершено {len(done)}")

    if joined:
        lines += ["", "<b>Уже читаю:</b>"]
        lines += [f"• @{entry.chat_ref}" for entry in joined[:12]]
    if pending:
        lines += ["", f"<b>Следующие в очереди ({len(pending)}):</b>"]
        lines += [f"• @{entry.chat_ref}" for entry in pending[:6]]
    if failed:
        lines += ["", "<b>Не получилось:</b>"]
        lines += [f"• @{entry.chat_ref} — {entry.last_error or '?'}" for entry in failed[:6]]

    lines += ["", "#дананг #разведка #статус"]
    return "\n".join(lines)


def main() -> int:
    """Build and deliver one status report."""
    _send(asyncio.run(build_report()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
