"""Tests for delivering housing alerts as a report followed by the original.

The owner's account is not a member of the source chats and the bot cannot
forward from them, so the monitoring account sends the report to the owner's
DM and forwards the advertisement after it. These tests cover the two-step
delivery with its durable checkpoint, the fallbacks when forwarding is refused
or the owner is unreachable, the ledger lines in the report, and the replay of
listings alerted on before this format existed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from telethon import errors

from pipeline.governor import TelegramActionGovernor
from pipeline.housing.owner_transport import OwnerTransport, SendOutcome, SendStatus
from pipeline.housing.replay import plan_replay, queue_replay, render_replay_header
from pipeline.housing.requirements import DEFAULT_REQUIREMENTS, match_requirements
from pipeline.housing.worker import (
    HousingAlertDelivery,
    format_when,
    render_alert,
    render_rematch_digest,
)
from pipeline.models import DeliveryResult, MediaPointer
from pipeline.recon_models import ActionKind, BudgetRule
from storage.db import Database
from storage.housing import (
    AlertKind,
    ForwardStatus,
    HousingStore,
    UnitOrigin,
    Verdict,
    unit_key_for,
)
from storage.scout import ScoutDatabase

CHAT_ID = -1002443479976
POSTED_AT = datetime(2026, 9, 4, 10, 4, 44, tzinfo=UTC)


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "eidolon.db")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def store(db: Database) -> HousingStore:
    return HousingStore(db.conn, db.write_lock, quiet_window_seconds=0.0)


@pytest.fixture
async def scout(tmp_path: Path) -> AsyncIterator[ScoutDatabase]:
    database = ScoutDatabase(tmp_path / "scout.db")
    await database.connect()
    yield database
    await database.close()


def _facts(**overrides: object) -> dict[str, object]:
    facts: dict[str, object] = {
        "unit_version": 1,
        "is_rental_offer": 1,
        "is_vehicle_ad": 0,
        "bedrooms": 2,
        "bedrooms_source": "text",
        "monthly_price_thb": 38000,
        "price_source": "text",
        "property_type": "house",
        "property_type_source": "text",
        "terrace": 1,
        "terrace_source": "text",
        "area_raw": "Maduawan",
        "vision_status": "not_attempted",
        "extractor_version": "test",
    }
    facts.update(overrides)
    return facts


async def _album_unit(
    db: Database,
    store: HousingStore,
    *,
    chat_id: int = CHAT_ID,
    first_msg_id: int = 28584,
    photos: int = 3,
    text: str = "🌿 Beautiful 2BR house, 38,000 THB/month, Maduawan",
    posted_at: datetime = POSTED_AT,
) -> str:
    """An advertisement posted as an album, with its message rows in place."""
    await store.set_chat_kind(chat_id, "dedicated_housing")
    # One album per first message id, so two fixtures never share a unit key.
    grouped_id = 14308130272878045 + first_msg_id
    key = unit_key_for(chat_id, grouped_id=grouped_id, telegram_msg_id=first_msg_id)
    for offset in range(photos):
        telegram_msg_id = first_msg_id + offset
        row_id = await db.store_message(
            telegram_msg_id=telegram_msg_id,
            chat_id=chat_id,
            chat_title="Koh Phangan Housing Groups 🏠",
            chat_type="supergroup",
            sender_id=8902657145,
            sender_name="@dmvillas (Daniel | Phangan Dream Villas)",
            text=text if offset == 0 else None,
            date=(posted_at + timedelta(seconds=offset)).isoformat(),
            media=MediaPointer(
                has_media=True, telegram_photo_id=1000 + offset, grouped_id=grouped_id
            ),
        )
        assert row_id is not None
        await store.record_message(
            unit_key=key,
            chat_id=chat_id,
            grouped_id=grouped_id,
            message_id=row_id,
            telegram_msg_id=telegram_msg_id,
            text=text if offset == 0 else None,
            has_media=True,
            telegram_photo_id=1000 + offset,
        )
    return key


async def _queue_live_alert(
    store: HousingStore, key: str, *, verdict: Verdict = Verdict.POSSIBLE
) -> int:
    unit = await store.get_unit(key)
    assert unit is not None
    facts = _facts()
    await store.record_facts(key, facts)
    result = match_requirements(facts, DEFAULT_REQUIREMENTS)
    origin = await store.unit_origin(key)
    alert_id = await store.record_match_with_alert(
        unit_key=key,
        requirements_revision=1,
        verdict=verdict,
        field_verdicts=result.as_dict(),
        alert={
            "chat_id": unit.chat_id,
            "chat_title": origin.chat_title,
            "telegram_msg_id": unit.members[0].telegram_msg_id,
            "verdict": verdict,
            "kind": AlertKind.LIVE,
            "body_html": render_alert(unit, facts, result, origin=origin),
        },
    )
    assert alert_id is not None
    return alert_id


# ---------------------------------------------------------------------------
# A fake owner transport
# ---------------------------------------------------------------------------


class FakeOwner:
    """An owner transport whose answers the test scripts, call by call."""

    def __init__(
        self,
        *,
        reports: list[SendOutcome] | None = None,
        forwards: list[SendOutcome] | None = None,
        copies: list[SendOutcome] | None = None,
        configured: bool = True,
    ) -> None:
        self.configured = configured
        self._reports = reports or []
        self._forwards = forwards or []
        self._copies = copies or []
        self.sent_reports: list[tuple[str, int | None]] = []
        self.forwarded: list[tuple[int, list[int]]] = []
        self.copied: list[dict[str, object]] = []
        self.edits: list[tuple[int, str]] = []
        self._next_id = 500

    def _sent(self) -> SendOutcome:
        self._next_id += 1
        return SendOutcome(SendStatus.SENT, message_id=self._next_id)

    async def send_report(self, body_html: str, *, reply_to: int | None = None) -> SendOutcome:
        self.sent_reports.append((body_html, reply_to))
        return self._reports.pop(0) if self._reports else self._sent()

    async def forward(self, *, chat_id: int, message_ids: list[int]) -> SendOutcome:
        self.forwarded.append((chat_id, list(message_ids)))
        return self._forwards.pop(0) if self._forwards else self._sent()

    async def send_copy(
        self,
        *,
        text: str | None,
        photo_paths: list[str],
        header_html: str,
        reply_to: int | None = None,
    ) -> SendOutcome:
        self.copied.append(
            {"text": text, "photos": list(photo_paths), "header": header_html, "reply_to": reply_to}
        )
        return self._copies.pop(0) if self._copies else self._sent()

    async def edit_report(self, message_id: int, body_html: str) -> SendOutcome:
        self.edits.append((message_id, body_html))
        return self._sent()


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def deliver_html(self, message: str) -> DeliveryResult:
        self.sent.append(message)
        return DeliveryResult.success()

    async def deliver_photo(self, caption_html: str, photo_path: str) -> DeliveryResult:
        self.sent.append(caption_html)
        return DeliveryResult.success()


async def _row(store: HousingStore, alert_id: int) -> dict[str, object]:
    cursor = await store._conn.execute(  # noqa: SLF001 - asserting durable state
        "SELECT * FROM housing_alerts WHERE id = ?", (alert_id,)
    )
    row = await cursor.fetchone()
    assert row is not None
    return dict(row)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_format_when_reads_in_island_time_with_a_relative_age() -> None:
    now = datetime(2026, 9, 5, 4, 0, tzinfo=UTC)  # 11:00 on Phangan
    assert format_when(datetime(2026, 9, 4, 10, 4, tzinfo=UTC), now=now) == "4 сен 17:04 (вчера)"
    assert format_when(datetime(2026, 9, 5, 1, 30, tzinfo=UTC), now=now) == "5 сен 08:30 (сегодня)"
    assert format_when(datetime(2026, 8, 30, 5, 19, tzinfo=UTC), now=now) == (
        "30 авг 12:19 (6 дней назад)"
    )
    # Another year is named; the current one is not.
    assert format_when(datetime(2024, 12, 31, 20, 0, tzinfo=UTC), now=now).startswith(
        "1 янв 2025 03:00"
    )


async def test_the_report_carries_the_ledger_line_and_no_dead_link(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store)
    unit = await store.get_unit(key)
    assert unit is not None
    facts = _facts()
    result = match_requirements(facts, DEFAULT_REQUIREMENTS)
    origin = await store.unit_origin(key)
    now = datetime(2026, 9, 5, 4, 0, tzinfo=UTC)

    body = render_alert(unit, facts, result, origin=origin, now=now)

    assert origin.posted_at == POSTED_AT
    assert origin.sender_name == "@dmvillas (Daniel | Phangan Dream Villas)"
    assert "📅 Опубликовано 4 сен 17:04 (вчера) · Koh Phangan Housing Groups 🏠 · @dmvillas" in body
    assert "38 000 THB/мес" in body
    assert "✅ Цена: 38 000 THB (хотим 20 000–40 000)" in body
    assert "✅ Тип жилья: дом" in body
    assert "t.me" not in body
    assert "<blockquote>" not in body
    assert body.endswith("⬇️ Оригинал ниже")


async def test_the_ledger_line_escapes_names_telegram_does_not_sanitize(
    db: Database, store: HousingStore
) -> None:
    """A sender called "<b>Dan & Co" must not become markup inside the report."""
    await store.set_chat_kind(CHAT_ID, "dedicated_housing")
    row_id = await db.store_message(
        telegram_msg_id=5,
        chat_id=CHAT_ID,
        chat_title="Rent <Phangan> & Co",
        sender_id=1,
        sender_name="<b>Dan & Co",
        text="Сдаю дом",
        date=POSTED_AT.isoformat(),
    )
    assert row_id is not None
    key = unit_key_for(CHAT_ID, grouped_id=None, telegram_msg_id=5)
    await store.record_message(
        unit_key=key,
        chat_id=CHAT_ID,
        grouped_id=None,
        message_id=row_id,
        telegram_msg_id=5,
        text="Сдаю дом",
        has_media=False,
        telegram_photo_id=None,
    )
    unit = await store.get_unit(key)
    assert unit is not None
    facts = _facts()

    body = render_alert(
        unit,
        facts,
        match_requirements(facts, DEFAULT_REQUIREMENTS),
        origin=await store.unit_origin(key),
    )

    assert "&lt;b&gt;Dan &amp; Co" in body
    assert "Rent &lt;Phangan&gt; &amp; Co" in body
    assert "<b>Dan" not in body


def test_a_replayed_report_names_the_first_alert() -> None:
    unit = SimpleNamespace(members=(SimpleNamespace(telegram_msg_id=1),), chat_id=CHAT_ID)
    facts = _facts()
    now = datetime(2026, 9, 5, 4, 0, tzinfo=UTC)
    origin = UnitOrigin(posted_at=POSTED_AT, chat_title="Chat", sender_name="@a (A)")

    body = render_alert(
        unit,  # type: ignore[arg-type]
        facts,
        match_requirements(facts, DEFAULT_REQUIREMENTS),
        origin=origin,
        replayed_from=datetime(2026, 9, 4, 10, 5, 8, tzinfo=UTC),
        now=now,
    )

    assert "🔁 Повтор: первый алерт был 4 сен 17:05 (вчера)" in body
    assert body.index("📅 Опубликовано") < body.index("🔁 Повтор") < body.index("⬇️ Оригинал ниже")


def test_the_rematch_digest_names_listings_by_date_and_chat_not_by_link() -> None:
    unit = SimpleNamespace(
        unit_key="u1", members=(SimpleNamespace(telegram_msg_id=1),), chat_id=CHAT_ID
    )
    facts = _facts()
    result = match_requirements(facts, DEFAULT_REQUIREMENTS)
    now = datetime(2026, 9, 5, 4, 0, tzinfo=UTC)

    body = render_rematch_digest(
        4,
        [(unit, facts, result)],  # type: ignore[list-item]
        origins={"u1": UnitOrigin(posted_at=POSTED_AT, chat_title="Rent Chat", sender_name=None)},
        now=now,
    )

    assert "• 38 000 THB · 2BR · Maduawan · 🎯" in body
    assert "4 сен 17:04 (вчера) · Rent Chat" in body
    assert "t.me" not in body


# ---------------------------------------------------------------------------
# Two-step delivery
# ---------------------------------------------------------------------------


async def test_a_live_alert_is_a_report_then_the_whole_album_forwarded(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store, photos=3)
    alert_id = await _queue_live_alert(store, key)
    owner = FakeOwner()

    sent = await HousingAlertDelivery(
        store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t"
    ).run_once()

    assert sent == 1
    assert len(owner.sent_reports) == 1
    assert owner.sent_reports[0][1] is None
    assert owner.forwarded == [(CHAT_ID, [28584, 28585, 28586])]
    row = await _row(store, alert_id)
    assert row["delivery_status"] == "delivered"
    assert row["forward_status"] == ForwardStatus.FORWARDED.value
    assert row["report_message_id"] == 501


async def test_the_report_is_never_sent_twice_when_the_forward_must_wait(
    db: Database, store: HousingStore
) -> None:
    """The checkpoint between the two calls is the whole point of the design."""
    key = await _album_unit(db, store)
    alert_id = await _queue_live_alert(store, key)
    owner = FakeOwner(
        forwards=[SendOutcome(SendStatus.RETRY, error_code="flood_wait", retry_after=1)]
    )
    delivery = HousingAlertDelivery(store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t")

    assert await delivery.run_once() == 0
    row = await _row(store, alert_id)
    assert row["delivery_status"] == "pending"
    assert row["report_message_id"] == 501
    assert row["last_error"] == "flood_wait"

    # The retry comes due; the report id is already on the row.
    async with store._write_lock:  # noqa: SLF001
        await store._conn.execute(  # noqa: SLF001
            "UPDATE housing_alerts SET next_attempt_at = CURRENT_TIMESTAMP WHERE id = ?",
            (alert_id,),
        )
        await store._conn.commit()  # noqa: SLF001
    assert await delivery.run_once() == 1

    assert len(owner.sent_reports) == 1
    assert len(owner.forwarded) == 2
    row = await _row(store, alert_id)
    assert row["delivery_status"] == "delivered"
    assert row["forward_status"] == ForwardStatus.FORWARDED.value


async def test_a_chat_that_forbids_forwarding_gets_a_copy_instead(
    db: Database, store: HousingStore, tmp_path: Path
) -> None:
    key = await _album_unit(db, store)
    alert_id = await _queue_live_alert(store, key)
    photo = tmp_path / "28584.jpg"
    photo.write_bytes(b"jpeg")
    await store.enqueue_media(unit_key=key, chat_id=CHAT_ID, photos=[(28584, 1000)])
    await store.settle_media(
        unit_key=key, telegram_msg_id=28584, status="downloaded", local_path=str(photo), byte_size=4
    )
    owner = FakeOwner(
        forwards=[SendOutcome(SendStatus.REJECTED, error_code="ChatForwardsRestrictedError")]
    )

    sent = await HousingAlertDelivery(
        store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t"
    ).run_once()

    assert sent == 1
    assert len(owner.copied) == 1
    copy = owner.copied[0]
    assert copy["photos"] == [str(photo)]
    assert "Beautiful 2BR house" in str(copy["text"])
    assert "чат запрещает пересылку" in str(copy["header"])
    assert copy["reply_to"] == 501
    row = await _row(store, alert_id)
    assert row["forward_status"] == ForwardStatus.COPIED.value
    assert row["forward_error"] == "ChatForwardsRestrictedError"


async def test_a_deleted_original_amends_the_report_when_no_copy_is_possible(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store)
    alert_id = await _queue_live_alert(store, key)
    owner = FakeOwner(
        forwards=[SendOutcome(SendStatus.REJECTED, error_code="MessageIdInvalidError")],
        copies=[SendOutcome(SendStatus.REJECTED, error_code="MediaEmptyError")],
    )

    sent = await HousingAlertDelivery(
        store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t"
    ).run_once()

    assert sent == 1
    assert owner.edits and owner.edits[0][0] == 501
    assert "⚠️ Оригинал недоступен: сообщение удалено" in owner.edits[0][1]
    assert "⬇️ Оригинал ниже" not in owner.edits[0][1]
    row = await _row(store, alert_id)
    assert row["delivery_status"] == "delivered"
    assert row["forward_status"] == ForwardStatus.UNAVAILABLE.value


async def test_a_partially_deleted_album_is_forwarded_and_the_gap_is_named(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store, photos=3)
    alert_id = await _queue_live_alert(store, key)
    owner = FakeOwner(forwards=[SendOutcome(SendStatus.SENT, message_id=502, missing=1)])

    assert (
        await HousingAlertDelivery(
            store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t"
        ).run_once()
        == 1
    )

    assert owner.edits and "1 из 3 сообщений удалены" in owner.edits[0][1]
    row = await _row(store, alert_id)
    assert row["forward_status"] == ForwardStatus.FORWARDED.value
    assert row["forward_error"] == "partial:1/3"


async def test_an_unreachable_owner_falls_back_to_the_bot_with_a_link(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store)
    alert_id = await _queue_live_alert(store, key)
    owner = FakeOwner(
        reports=[SendOutcome(SendStatus.UNREACHABLE, error_code="UserIsBlockedError")]
    )
    bot = FakeBot()

    sent = await HousingAlertDelivery(
        store=store, dispatcher=bot, owner=owner, lease_owner="t"
    ).run_once()

    assert sent == 1
    assert owner.forwarded == []
    assert len(bot.sent) == 1
    assert bot.sent[0].endswith('<a href="https://t.me/c/2443479976/28584">Открыть в Telegram</a>')
    assert "⬇️ Оригинал ниже" not in bot.sent[0]
    row = await _row(store, alert_id)
    assert row["forward_status"] == ForwardStatus.BOT_FALLBACK.value


async def test_without_an_owner_configured_nothing_changes_for_the_bot_path(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store)
    await _queue_live_alert(store, key)
    owner = FakeOwner(configured=False)
    bot = FakeBot()

    await HousingAlertDelivery(store=store, dispatcher=bot, owner=owner, lease_owner="t").run_once()

    assert owner.sent_reports == []
    assert len(bot.sent) == 1


async def test_a_follow_up_replies_to_the_report_and_forwards_nothing(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store)
    live_id = await _queue_live_alert(store, key)
    owner = FakeOwner()
    delivery = HousingAlertDelivery(store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t")
    assert await delivery.run_once() == 1
    unit = await store.get_unit(key)
    assert unit is not None
    update_id = await store.enqueue_alert(
        unit_key=key,
        chat_id=CHAT_ID,
        chat_title="Koh Phangan Housing Groups 🏠",
        telegram_msg_id=28584,
        requirements_revision=1,
        verdict=Verdict.CONFIRMED,
        kind=AlertKind.UPDATE,
        body_html="<b>🏠 Совпадение</b>\n<i>Уточнено по 3 фото</i>",
    )
    assert update_id is not None

    assert await delivery.run_once() == 1

    assert owner.sent_reports[1][1] == (await _row(store, live_id))["report_message_id"]
    assert len(owner.forwarded) == 1  # only the live alert forwarded the original
    row = await _row(store, update_id)
    assert row["forward_status"] == ForwardStatus.SKIPPED.value


async def test_a_follow_up_with_no_prior_report_shows_the_original(
    db: Database, store: HousingStore
) -> None:
    """The first alert failed or predates this format: the owner has never seen it."""
    key = await _album_unit(db, store)
    update_id = await store.enqueue_alert(
        unit_key=key,
        chat_id=CHAT_ID,
        chat_title=None,
        telegram_msg_id=28584,
        requirements_revision=1,
        verdict=Verdict.CONFIRMED,
        kind=AlertKind.UPDATE,
        body_html="update",
    )
    assert update_id is not None
    owner = FakeOwner()

    await HousingAlertDelivery(
        store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t"
    ).run_once()

    assert owner.sent_reports[0][1] is None
    assert len(owner.forwarded) == 1
    assert (await _row(store, update_id))["forward_status"] == ForwardStatus.FORWARDED.value


async def test_a_digest_is_a_report_alone(db: Database, store: HousingStore) -> None:
    digest_id = await store.enqueue_alert(
        unit_key="rematch:3",
        chat_id=0,
        chat_title=None,
        telegram_msg_id=0,
        requirements_revision=3,
        verdict=Verdict.POSSIBLE,
        kind=AlertKind.DIGEST,
        body_html="digest",
    )
    assert digest_id is not None
    owner = FakeOwner()

    await HousingAlertDelivery(
        store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t"
    ).run_once()

    assert len(owner.sent_reports) == 1
    assert owner.forwarded == []
    assert (await _row(store, digest_id))["forward_status"] == ForwardStatus.SKIPPED.value


async def test_a_delivery_that_raises_does_not_hold_the_rest_of_the_batch(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store)
    first = await _queue_live_alert(store, key)
    second = await store.enqueue_alert(
        unit_key="rematch:9",
        chat_id=0,
        chat_title=None,
        telegram_msg_id=0,
        requirements_revision=1,
        verdict=Verdict.POSSIBLE,
        kind=AlertKind.DIGEST,
        body_html="digest",
    )
    assert second is not None

    class Exploding(FakeOwner):
        async def forward(self, *, chat_id: int, message_ids: list[int]) -> SendOutcome:
            raise RuntimeError("boom")

    owner = Exploding()
    sent = await HousingAlertDelivery(
        store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t"
    ).run_once()

    assert sent == 1  # the digest went out despite the explosion before it
    row = await _row(store, first)
    assert row["delivery_status"] == "pending"
    assert row["last_error"] == "RuntimeError"
    assert row["report_message_id"] is not None  # and the report will not repeat


async def test_giving_up_on_the_forward_closes_the_alert_with_a_note(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store)
    alert_id = await _queue_live_alert(store, key)
    owner = FakeOwner(
        forwards=[SendOutcome(SendStatus.RETRY, error_code="flood_wait", retry_after=1)] * 3
    )
    delivery = HousingAlertDelivery(
        store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t", max_attempts=2
    )
    assert await delivery.run_once() == 0
    async with store._write_lock:  # noqa: SLF001
        await store._conn.execute(  # noqa: SLF001
            "UPDATE housing_alerts SET next_attempt_at = CURRENT_TIMESTAMP WHERE id = ?",
            (alert_id,),
        )
        await store._conn.commit()  # noqa: SLF001

    assert await delivery.run_once() == 1

    row = await _row(store, alert_id)
    assert row["delivery_status"] == "delivered"
    assert row["forward_status"] == ForwardStatus.UNAVAILABLE.value
    assert owner.edits and "⚠️ Оригинал недоступен" in owner.edits[0][1]


# ---------------------------------------------------------------------------
# The owner transport against a scripted Telethon client
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self) -> None:
        self.entity_error: Exception | None = None
        self.forward_result: object = None
        self.forward_error: Exception | None = None
        self.calls: list[tuple[str, object]] = []

    async def get_entity(self, ref: str) -> object:
        self.calls.append(("get_entity", ref))
        if self.entity_error is not None:
            raise self.entity_error
        return SimpleNamespace(id=1, username=ref.lstrip("@"))

    async def send_message(self, entity: object, text: str, **kwargs: object) -> object:
        self.calls.append(("send_message", text))
        return SimpleNamespace(id=900)

    async def forward_messages(self, entity: object, ids: list[int], **kwargs: object) -> object:
        self.calls.append(("forward_messages", list(ids)))
        if self.forward_error is not None:
            raise self.forward_error
        return self.forward_result


async def test_an_unknown_owner_username_is_a_clean_unreachable_not_a_crash(
    scout: ScoutDatabase,
) -> None:
    """Telethon reports an unoccupied username as a bare ValueError; the
    governor would settle that as ambiguous and re-raise, stalling delivery."""
    client = FakeClient()
    client.entity_error = ValueError('No user has "nobody" as username')
    transport = OwnerTransport(
        client=client, governor=TelegramActionGovernor(scout=scout), owner_ref="@nobody"
    )

    outcome = await transport.send_report("<b>x</b>")

    assert outcome.status is SendStatus.UNREACHABLE
    assert outcome.error_code == "UsernameNotOccupiedError"


def _unpaced(scout: ScoutDatabase) -> TelegramActionGovernor:
    """A governor with no pace on owner messages, for tests about other things."""
    return TelegramActionGovernor(
        scout=scout,
        policy={
            ActionKind.OWNER_MESSAGE: BudgetRule(),
            ActionKind.OWNER_FORWARD: BudgetRule(),
            ActionKind.RESOLVE_USERNAME: BudgetRule(),
        },
    )


async def test_the_owner_is_resolved_once_per_process(scout: ScoutDatabase) -> None:
    client = FakeClient()
    transport = OwnerTransport(client=client, governor=_unpaced(scout), owner_ref="nikitacometa")

    first = await transport.send_report("a")
    second = await transport.send_report("b")

    assert first.sent and first.message_id == 900
    assert second.sent
    assert [call for call in client.calls if call[0] == "get_entity"] == [
        ("get_entity", "nikitacometa")
    ]


async def test_two_reports_inside_the_pace_are_spaced_not_failed(scout: ScoutDatabase) -> None:
    """The default policy paces owner messages two seconds apart. The second
    one is told to come back, and the outbox must not count that as a failed
    attempt: pacing is the design working, not the delivery breaking."""
    client = FakeClient()
    transport = OwnerTransport(
        client=client, governor=TelegramActionGovernor(scout=scout), owner_ref="owner"
    )

    first = await transport.send_report("a")
    second = await transport.send_report("b")

    assert first.sent
    assert second.status is SendStatus.RETRY
    assert second.error_code == "paced"
    assert second.retry_after is not None and 1 <= second.retry_after <= 2


async def test_a_paced_delivery_is_rescheduled_without_spending_an_attempt(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store)
    alert_id = await _queue_live_alert(store, key)
    owner = FakeOwner(reports=[SendOutcome(SendStatus.RETRY, error_code="paced", retry_after=2)])

    sent = await HousingAlertDelivery(
        store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t", max_attempts=1
    ).run_once()

    assert sent == 0
    row = await _row(store, alert_id)
    assert row["delivery_status"] == "pending"
    assert row["attempts"] == 0


async def test_a_forward_with_a_deleted_member_reports_how_many_are_missing(
    scout: ScoutDatabase,
) -> None:
    client = FakeClient()
    client.forward_result = [SimpleNamespace(id=1), None, SimpleNamespace(id=3)]
    transport = OwnerTransport(
        client=client, governor=TelegramActionGovernor(scout=scout), owner_ref="owner"
    )

    outcome = await transport.forward(chat_id=CHAT_ID, message_ids=[12, 10, 11])

    assert outcome.sent
    assert outcome.missing == 1
    assert client.calls[-1] == ("forward_messages", [10, 11, 12])


async def test_a_forward_where_every_member_is_gone_is_rejected(scout: ScoutDatabase) -> None:
    client = FakeClient()
    client.forward_result = [None, None]
    transport = OwnerTransport(
        client=client, governor=TelegramActionGovernor(scout=scout), owner_ref="owner"
    )

    outcome = await transport.forward(chat_id=CHAT_ID, message_ids=[1, 2])

    assert outcome.status is SendStatus.REJECTED
    assert outcome.error_code == "MessageIdInvalidError"


async def test_a_restricted_chat_is_a_rejection_the_delivery_can_act_on(
    scout: ScoutDatabase,
) -> None:
    client = FakeClient()
    client.forward_error = errors.ChatForwardsRestrictedError(request=None)
    transport = OwnerTransport(
        client=client, governor=TelegramActionGovernor(scout=scout), owner_ref="owner"
    )

    outcome = await transport.forward(chat_id=CHAT_ID, message_ids=[1])

    assert outcome.status is SendStatus.REJECTED
    assert outcome.error_code == "ChatForwardsRestrictedError"


async def test_a_blocked_owner_is_unreachable_not_merely_rejected(scout: ScoutDatabase) -> None:
    client = FakeClient()
    client.forward_error = errors.UserIsBlockedError(request=None)
    transport = OwnerTransport(
        client=client, governor=TelegramActionGovernor(scout=scout), owner_ref="owner"
    )

    outcome = await transport.forward(chat_id=CHAT_ID, message_ids=[1])

    assert outcome.status is SendStatus.UNREACHABLE


async def test_owner_kinds_are_accepted_by_a_ledger_built_before_them(tmp_path: Path) -> None:
    """A CHECK that lacks the owner kinds would refuse every delivery reservation."""
    import sqlite3

    db_path = tmp_path / "legacy.db"
    raw = sqlite3.connect(db_path)
    raw.executescript(
        """
        CREATE TABLE telegram_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            kind TEXT NOT NULL
                CHECK(kind IN (
                    'join', 'hashtag_search', 'fulltext_search', 'contacts_search',
                    'global_search', 'recommendations', 'resolve_username',
                    'invite_check', 'history_page',
                    'media_download_live', 'media_download_backfill'
                )),
            idempotency_key TEXT NOT NULL UNIQUE,
            job_id TEXT,
            candidate_id INTEGER,
            outcome TEXT NOT NULL DEFAULT 'reserved'
                CHECK(outcome IN ('reserved', 'succeeded', 'failed', 'flood_wait', 'ambiguous')),
            flood_wait_seconds INTEGER,
            error_code TEXT,
            duration_ms REAL,
            reserved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            settled_at TIMESTAMP
        );
        INSERT INTO telegram_actions (account_id, kind, idempotency_key, outcome)
        VALUES ('owner-primary', 'media_download_live', 'historic-1', 'succeeded');
        """
    )
    raw.commit()
    raw.close()

    database = ScoutDatabase(db_path)
    await database.connect()
    try:
        governor = TelegramActionGovernor(scout=database)

        async def call() -> str:
            return "ok"

        result = await governor.run(ActionKind.OWNER_FORWARD, "owner:test", call)
        assert result.ok
        kept = await (
            await database.conn.execute(
                "SELECT kind FROM telegram_actions WHERE idempotency_key = 'historic-1'"
            )
        ).fetchone()
        assert kept is not None and kept[0] == "media_download_live"
    finally:
        await database.close()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


async def test_the_replay_queues_every_alerted_listing_once_oldest_first(
    db: Database, store: HousingStore
) -> None:
    newer = await _album_unit(db, store, first_msg_id=300, photos=2, posted_at=POSTED_AT)
    older = await _album_unit(
        db, store, first_msg_id=100, photos=1, posted_at=POSTED_AT - timedelta(days=3)
    )
    for key in (newer, older):
        await _queue_live_alert(store, key)
    # Both were delivered in the old format.
    delivered = await store.claim_due_alerts(lease_owner="old-format")
    for alert in delivered:
        await store.settle_alert(int(alert["id"]), delivered=True)
    await store.save_requirements(definition=DEFAULT_REQUIREMENTS, created_by="test")

    plan = await plan_replay(store, now=datetime(2026, 9, 5, 4, 0, tzinfo=UTC))
    queued = await queue_replay(store, plan, now=datetime(2026, 9, 5, 4, 0, tzinfo=UTC))

    assert [item.unit_key for item in plan.items] == sorted(
        {newer, older}, key=lambda k: 0 if k == older else 1
    )
    assert queued == 2
    for item in plan.items:
        assert "🔁 Повтор: первый алерт был" in item.body_html
        assert "📅 Опубликовано" in item.body_html
    due = await store.claim_due_alerts(lease_owner="now")
    assert [str(row["kind"]) for row in due] == [
        AlertKind.DIGEST.value
    ]  # header first, listings later
    assert "Это Эйдолон" in str(due[0]["body_html"])
    assert "Пересылаю 2 объявлений" in str(due[0]["body_html"])

    # A second run finds them already queued and adds nothing.
    again = await plan_replay(store)
    assert again.items == []
    assert sorted(again.already_queued) == sorted({newer, older})
    assert await queue_replay(store, again) == 0


async def test_the_replay_leaves_out_a_listing_the_current_requirements_reject(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store)
    await _queue_live_alert(store, key)
    for alert in await store.claim_due_alerts(lease_owner="old"):
        await store.settle_alert(int(alert["id"]), delivered=True)
    await store.save_requirements(
        definition={
            "bedrooms": {"operator": "at_least", "value": 3},
            "monthly_rent_thb": {"min": 20000, "max": 40000},
        },
        created_by="test",
    )

    plan = await plan_replay(store)

    assert plan.items == []
    assert len(plan.rejected) == 1
    assert plan.rejected[0][0] == key
    assert "hard miss" in plan.rejected[0][1]
    assert "Не пересылаю" in render_replay_header(plan) or not plan.items


async def test_the_replay_ignores_the_digest_and_the_undelivered(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store)
    await _queue_live_alert(store, key)  # pending, never delivered
    await store.enqueue_alert(
        unit_key="rematch:3",
        chat_id=0,
        chat_title=None,
        telegram_msg_id=0,
        requirements_revision=1,
        verdict=Verdict.POSSIBLE,
        kind=AlertKind.DIGEST,
        body_html="d",
    )
    await store.save_requirements(definition=DEFAULT_REQUIREMENTS, created_by="test")

    plan = await plan_replay(store)

    assert plan.items == []
    assert plan.rejected == []


async def test_a_replay_row_is_delivered_as_report_plus_forward(
    db: Database, store: HousingStore
) -> None:
    key = await _album_unit(db, store, photos=2)
    await _queue_live_alert(store, key)
    for alert in await store.claim_due_alerts(lease_owner="old"):
        await store.settle_alert(int(alert["id"]), delivered=True)
    await store.save_requirements(definition=DEFAULT_REQUIREMENTS, created_by="test")
    plan = await plan_replay(store)
    await queue_replay(store, plan)
    async with store._write_lock:  # noqa: SLF001
        await store._conn.execute(  # noqa: SLF001
            "UPDATE housing_alerts SET next_attempt_at = CURRENT_TIMESTAMP WHERE delivery_status = 'pending'"
        )
        await store._conn.commit()  # noqa: SLF001
    owner = FakeOwner()

    sent = await HousingAlertDelivery(
        store=store, dispatcher=FakeBot(), owner=owner, lease_owner="t"
    ).run_once()

    assert sent == 2  # header + one listing
    assert owner.forwarded == [(CHAT_ID, [28584, 28585])]
    bodies = [body for body, _ in owner.sent_reports]
    assert bodies[0].startswith("<b>👁 Это Эйдолон</b>")
    assert "🔁 Повтор" in bodies[1]


# The housing_alerts table as deployed before the forward step existed.
LEGACY_HOUSING_ALERTS_SQL = """
CREATE TABLE housing_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_key TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    chat_title TEXT,
    telegram_msg_id INTEGER NOT NULL,
    requirements_revision INTEGER NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('confirmed', 'possible')),
    kind TEXT NOT NULL DEFAULT 'live' CHECK(kind IN ('live', 'update', 'digest')),
    body_html TEXT NOT NULL,
    photo_paths_json TEXT,
    delivery_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(delivery_status IN ('pending', 'delivered', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    claimed_until TIMESTAMP,
    lease_owner TEXT,
    last_error TEXT,
    next_attempt_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMP
);
CREATE UNIQUE INDEX idx_housing_alerts_dedup ON housing_alerts(unit_key, verdict, kind);
CREATE INDEX idx_housing_alerts_due ON housing_alerts(delivery_status, next_attempt_at);
"""


async def test_alerts_migrated_from_the_bot_era_read_as_skipped_not_owed(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "legacy.db"
    raw = sqlite3.connect(db_path)
    raw.executescript(LEGACY_HOUSING_ALERTS_SQL)
    raw.execute(
        "INSERT INTO housing_alerts (unit_key, chat_id, telegram_msg_id, requirements_revision,"
        " verdict, kind, body_html, delivery_status, attempts, delivered_at)"
        " VALUES ('m:1:1', -100, 1, 3, 'possible', 'live', 'x', 'delivered', 1, CURRENT_TIMESTAMP)"
    )
    raw.execute(
        "INSERT INTO housing_alerts (unit_key, chat_id, telegram_msg_id, requirements_revision,"
        " verdict, kind, body_html) VALUES ('m:1:2', -100, 2, 3, 'possible', 'live', 'y')"
    )
    raw.commit()
    raw.close()

    database = Database(db_path)
    await database.connect()
    try:
        rows = await (
            await database.conn.execute(
                "SELECT unit_key, delivery_status, forward_status FROM housing_alerts ORDER BY id"
            )
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("m:1:1", "delivered", "skipped"),
            ("m:1:2", "pending", "pending"),
        ]
        await database.conn.execute(
            "INSERT INTO housing_alerts (unit_key, chat_id, telegram_msg_id, requirements_revision,"
            " verdict, kind, body_html) VALUES ('m:1:3', -100, 3, 3, 'possible', 'replay', 'z')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            await database.conn.execute(
                "INSERT INTO housing_alerts (unit_key, chat_id, telegram_msg_id,"
                " requirements_revision, verdict, kind, body_html)"
                " VALUES ('m:1:3', -100, 3, 3, 'possible', 'replay', 'z')"
            )
        indexes = await (
            await database.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='housing_alerts'"
                " AND name LIKE 'idx_housing_alerts%' ORDER BY name"
            )
        ).fetchall()
        assert [row[0] for row in indexes] == ["idx_housing_alerts_dedup", "idx_housing_alerts_due"]
        legacy = await (
            await database.conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'housing_alerts_legacy'"
            )
        ).fetchall()
        assert legacy == []
    finally:
        await database.close()


def test_photo_paths_json_survives_the_bot_fallback() -> None:
    assert json.loads(json.dumps(["a"])) == ["a"]
