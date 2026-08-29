"""Tests for the housing subsystem: units, requirements, verdicts, alerts."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.housing.extractor import HousingFacts, HousingTextExtractor
from pipeline.housing.requirements import (
    DEFAULT_REQUIREMENTS,
    FieldState,
    RequirementsError,
    match_requirements,
    validate_requirements,
)
from pipeline.housing.vision import VisionReading
from pipeline.housing.worker import (
    HousingAlertDelivery,
    HousingVisionWorker,
    HousingWorker,
    render_alert,
)
from pipeline.ingestion import media_pointer
from pipeline.models import DeliveryResult
from storage.db import Database
from storage.housing import AlertKind, HousingStore, UnitState, Verdict, unit_key_for


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "eidolon.db")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def store(db: Database) -> HousingStore:
    # Zero quiet window: these tests are about what the assembler decides, not
    # about waiting for it, and a real delay would only make them slow.
    return HousingStore(db.conn, db.write_lock, quiet_window_seconds=0.0)


def _text_facts(**overrides: object) -> dict[str, object]:
    """A fully-known listing, before a test knocks one field out."""
    facts: dict[str, object] = {
        "is_rental_offer": 1,
        "is_vehicle_ad": 0,
        "bedrooms": 2,
        "bedrooms_source": "text",
        "bathrooms": 2,
        "bathrooms_source": "text",
        "monthly_price_thb": 30000,
        "price_source": "text",
        "tv_present": 1,
        "tv_size_class": "large",
        "tv_source": "text",
    }
    facts.update(overrides)
    return facts


# ---------------------------------------------------------------------------
# Content units
# ---------------------------------------------------------------------------


async def test_album_members_become_one_advertisement(store: HousingStore) -> None:
    """Three messages, one grouped_id, one listing — with the caption kept.

    Telegram delivers an album as separate updates and only one of them
    carries the text. A unit that took the last writer's text would report a
    listing with no description at all.
    """
    key = unit_key_for(-100777, grouped_id=555, telegram_msg_id=10)
    for index, (message_id, text) in enumerate(
        [(1, "Сдаю дом, 2 спальни, 30000 бат"), (2, None), (3, None)], start=10
    ):
        await store.record_message(
            unit_key=key,
            chat_id=-100777,
            grouped_id=555,
            message_id=message_id,
            telegram_msg_id=index,
            text=text,
            has_media=True,
            telegram_photo_id=900 + message_id,
        )

    units = await store.claim_settled_units()

    assert [unit.unit_key for unit in units] == [key]
    unit = units[0]
    assert unit.assembled_text == "Сдаю дом, 2 спальни, 30000 бат"
    assert unit.media_count == 3
    assert len(unit.members) == 3
    assert unit.photo_ids == (901, 902, 903)


async def test_a_lone_album_message_still_becomes_a_unit(store: HousingStore) -> None:
    """A grouped_id whose siblings never arrive must not vanish.

    This is why Telethon's own album event is not used: its filter refuses to
    dispatch a group of one, so a single-photo post carrying a grouped_id
    would be dropped in silence.
    """
    key = unit_key_for(-100777, grouped_id=999, telegram_msg_id=42)
    await store.record_message(
        unit_key=key,
        chat_id=-100777,
        grouped_id=999,
        message_id=1,
        telegram_msg_id=42,
        text="Вилла в аренду",
        has_media=True,
        telegram_photo_id=1,
    )

    units = await store.claim_settled_units()

    assert [unit.unit_key for unit in units] == [key]
    assert units[0].assembled_text == "Вилла в аренду"


async def test_a_claimed_unit_is_not_claimed_twice(store: HousingStore) -> None:
    """Claiming moves the unit out of assembling in the same transaction."""
    key = unit_key_for(-100777, grouped_id=None, telegram_msg_id=5)
    await store.record_message(
        unit_key=key,
        chat_id=-100777,
        grouped_id=None,
        message_id=1,
        telegram_msg_id=5,
        text="Сдаю",
        has_media=False,
        telegram_photo_id=None,
    )

    first = await store.claim_settled_units()
    second = await store.claim_settled_units()

    assert [unit.unit_key for unit in first] == [key]
    assert second == []


async def test_a_late_duplicate_does_not_reopen_a_finalized_unit(store: HousingStore) -> None:
    """A replayed message must not push the deadline of a unit already in flight."""
    key = unit_key_for(-100777, grouped_id=None, telegram_msg_id=8)
    await store.record_message(
        unit_key=key,
        chat_id=-100777,
        grouped_id=None,
        message_id=1,
        telegram_msg_id=8,
        text="Сдаю дом",
        has_media=False,
        telegram_photo_id=None,
    )
    await store.claim_settled_units()
    await store.set_unit_state(key, UnitState.EXTRACTING)

    await store.record_message(
        unit_key=key,
        chat_id=-100777,
        grouped_id=None,
        message_id=1,
        telegram_msg_id=8,
        text="Сдаю дом",
        has_media=False,
        telegram_photo_id=None,
    )

    unit = await store.get_unit(key)
    assert unit is not None
    assert unit.state is UnitState.EXTRACTING


async def test_replaying_a_message_does_not_inflate_the_photo_count(
    store: HousingStore,
) -> None:
    """Recovery replays an interrupted job; the unit must land on the same numbers.

    A count that grows on every retry would later request photographs that do
    not exist and report an advertisement as carrying more images than it does.
    """
    key = unit_key_for(-100777, grouped_id=321, telegram_msg_id=1)
    for _ in range(3):
        await store.record_message(
            unit_key=key,
            chat_id=-100777,
            grouped_id=321,
            message_id=1,
            telegram_msg_id=1,
            text="Сдаю виллу",
            has_media=True,
            telegram_photo_id=555,
        )

    unit = await store.get_unit(key)
    assert unit is not None
    assert unit.media_count == 1
    assert len(unit.members) == 1


def test_media_pointer_reads_a_photo_without_asking_telegram() -> None:
    """The pointer comes off the delivered object: no request, no budget."""
    message = SimpleNamespace(
        id=1,
        grouped_id=77,
        photo=SimpleNamespace(id=12345),
        document=None,
    )

    pointer = media_pointer(message)

    assert pointer.has_media is True
    assert pointer.telegram_photo_id == 12345
    assert pointer.grouped_id == 77
    assert pointer.scanned is True


def test_media_pointer_counts_an_uncompressed_photo_sent_as_a_document() -> None:
    """ "Send without compression" is how someone photographing a house sends."""
    message = SimpleNamespace(
        id=1,
        grouped_id=None,
        photo=None,
        document=SimpleNamespace(id=999, mime_type="image/jpeg"),
    )

    pointer = media_pointer(message)

    assert pointer.has_media is True
    assert pointer.telegram_photo_id == 999


def test_media_pointer_ignores_a_non_image_document() -> None:
    """A PDF is not a photograph of a bathroom."""
    message = SimpleNamespace(
        id=1,
        grouped_id=None,
        photo=None,
        document=SimpleNamespace(id=999, mime_type="application/pdf"),
    )

    assert media_pointer(message).has_media is False


# ---------------------------------------------------------------------------
# Matching: the three-valued part
# ---------------------------------------------------------------------------


def test_everything_known_and_satisfied_is_confirmed() -> None:
    result = match_requirements(_text_facts(), DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.CONFIRMED
    assert result.unknown_fields == ()


def test_an_unstated_bathroom_count_is_possible_not_a_rejection() -> None:
    """The failure this whole design exists to prevent.

    Phangan listings state bathrooms 2.3% of the time. Reading silence as
    "does not have two" would reject essentially every real advertisement and
    the owner would receive nothing at all.
    """
    facts = _text_facts(bathrooms=None, bathrooms_source="unknown")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.POSSIBLE
    assert "bathrooms" in result.unknown_fields


def test_a_listing_with_no_photos_and_no_details_still_reaches_the_owner() -> None:
    """ "Фото скину в личку" is a normal listing, not a non-match."""
    facts = _text_facts(
        bathrooms=None,
        bathrooms_source="unknown",
        tv_present=None,
        tv_size_class=None,
        tv_source="unknown",
    )

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.POSSIBLE
    assert set(result.unknown_fields) == {"bathrooms", "tv"}


def test_a_stated_bedroom_count_below_the_requirement_is_a_hard_miss() -> None:
    """A one-bedroom house is a fact about the property, so it does reject."""
    facts = _text_facts(bedrooms=1)

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.HARD_MISS


def test_a_photograph_showing_fewer_bathrooms_is_a_lower_bound() -> None:
    """Vision counts what is in frame; a bathroom off-camera is not absent.

    Rejecting on a visual count would throw away listings for the crime of
    photographing the kitchen.
    """
    facts = _text_facts(bathrooms=1, bathrooms_source="vision")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.POSSIBLE
    assert "bathrooms" in result.unknown_fields


def test_a_price_above_the_budget_is_a_hard_miss() -> None:
    facts = _text_facts(monthly_price_thb=65000)

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.HARD_MISS
    assert [f.state for f in result.fields if f.field == "monthly_rent_thb"] == [
        FieldState.VIOLATED
    ]


def test_a_missing_price_is_unknown_not_out_of_budget() -> None:
    facts = _text_facts(monthly_price_thb=None, price_source="unknown")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.POSSIBLE


def test_a_television_nobody_mentioned_is_unknown() -> None:
    facts = _text_facts(tv_present=None, tv_size_class=None, tv_source="unknown")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.POSSIBLE
    assert "tv" in result.unknown_fields


def test_a_television_reported_absent_is_a_violation() -> None:
    """Someone saying there is no TV is evidence, unlike a photo that missed it."""
    facts = _text_facts(tv_present=0, tv_size_class="none", tv_source="text")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.HARD_MISS


def test_a_television_of_unclear_size_is_unknown() -> None:
    facts = _text_facts(tv_present=1, tv_size_class="unclear", tv_source="vision")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.POSSIBLE


def test_a_small_television_misses_a_large_requirement() -> None:
    facts = _text_facts(tv_size_class="small", tv_source="text")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.HARD_MISS


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


def test_the_owners_stated_criteria_validate() -> None:
    assert validate_requirements(DEFAULT_REQUIREMENTS) == DEFAULT_REQUIREMENTS


def test_an_unknown_criterion_is_refused_rather_than_ignored() -> None:
    """A silently dropped criterion is a filter that stopped filtering."""
    with pytest.raises(RequirementsError, match="unknown requirement"):
        validate_requirements({**DEFAULT_REQUIREMENTS, "pool": {"required": True}})


def test_an_inverted_budget_is_refused() -> None:
    with pytest.raises(RequirementsError, match="must not exceed"):
        validate_requirements({"monthly_rent_thb": {"min": 50000, "max": 10000}})


def test_an_unknown_tv_class_is_refused() -> None:
    with pytest.raises(RequirementsError, match="minimum_class"):
        validate_requirements({"tv": {"minimum_class": "enormous"}})


async def test_saving_requirements_appends_a_revision_and_bumps_the_generation(
    store: HousingStore,
) -> None:
    first = await store.save_requirements(definition=DEFAULT_REQUIREMENTS, created_by="test")
    generation_after_first = await store.requirements_generation()

    second = await store.save_requirements(
        definition={**DEFAULT_REQUIREMENTS, "monthly_rent_thb": {"min": 10000, "max": 20000}},
        created_by="test",
    )

    assert (first, second) == (1, 2)
    assert await store.requirements_generation() > generation_after_first
    active = await store.active_requirements()
    assert active is not None
    assert active[0] == 2
    assert active[1]["monthly_rent_thb"] == {"min": 10000, "max": 20000}


async def test_a_stale_edit_is_refused_instead_of_overwriting(store: HousingStore) -> None:
    """Two editors from the same starting point: the loser must be told."""
    await store.save_requirements(definition=DEFAULT_REQUIREMENTS, created_by="test")
    await store.save_requirements(
        definition={"bedrooms": {"operator": "at_least", "value": 3}},
        created_by="first",
        expected_revision=1,
    )

    with pytest.raises(ValueError, match="requirements moved on"):
        await store.save_requirements(
            definition={"bedrooms": {"operator": "at_least", "value": 1}},
            created_by="second",
            expected_revision=1,
        )


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------


class FakeExtractor:
    """A text extractor whose answer the test chooses."""

    def __init__(self, facts: HousingFacts) -> None:
        self.facts = facts
        self.calls: list[str] = []

    async def extract(self, text: str) -> HousingFacts:
        self.calls.append(text)
        return self.facts


async def _queue_unit(store: HousingStore, text: str, *, chat_id: int = -1001199262612) -> str:
    key = unit_key_for(chat_id, grouped_id=None, telegram_msg_id=100)
    await store.record_message(
        unit_key=key,
        chat_id=chat_id,
        grouped_id=None,
        message_id=1,
        telegram_msg_id=100,
        text=text,
        has_media=False,
        telegram_photo_id=None,
    )
    return key


async def test_a_matching_listing_produces_one_alert(store: HousingStore) -> None:
    key = await _queue_unit(store, "Сдаю дом 2 спальни 2 санузла, большой телевизор, 30000 бат")
    worker = HousingWorker(
        store=store,
        extractor=FakeExtractor(
            HousingFacts(
                is_rental_offer=True,
                is_vehicle_ad=False,
                bedrooms=2,
                bathrooms=2,
                monthly_price_thb=30000,
                tv_present=True,
                tv_size_class="large",
                evidence_quote="Сдаю дом 2 спальни",
            )
        ),
    )

    processed = await worker.run_once()

    assert processed == 1
    alerts = await store.claim_due_alerts(lease_owner="test")
    assert len(alerts) == 1
    assert alerts[0]["verdict"] == Verdict.CONFIRMED.value
    assert alerts[0]["kind"] == AlertKind.LIVE.value
    unit = await store.get_unit(key)
    assert unit is not None
    assert unit.state is UnitState.DONE


async def test_a_scooter_advertisement_produces_no_alert(store: HousingStore) -> None:
    """Vehicles are rented with the same verbs and are the dominant noise."""
    await _queue_unit(store, "Сдам в аренду Yamaha Filano 2024, 3200 бат в месяц")
    worker = HousingWorker(
        store=store,
        extractor=FakeExtractor(HousingFacts(is_rental_offer=True, is_vehicle_ad=True)),
    )

    await worker.run_once()

    assert await store.claim_due_alerts(lease_owner="test") == []


async def test_someone_looking_for_a_house_produces_no_alert(store: HousingStore) -> None:
    await _queue_unit(store, "Ищу дом на Пангане с 1 апреля, 2 спальни, бюджет 30000")
    worker = HousingWorker(
        store=store,
        extractor=FakeExtractor(HousingFacts(is_rental_offer=False, is_vehicle_ad=False)),
    )

    await worker.run_once()

    assert await store.claim_due_alerts(lease_owner="test") == []


async def test_an_extraction_failure_records_unknowns_and_does_not_crash(
    store: HousingStore,
) -> None:
    """A provider outage must cost certainty, never the unit."""
    key = await _queue_unit(store, "Сдаю дом")
    worker = HousingWorker(
        store=store,
        extractor=FakeExtractor(HousingFacts.unreadable("APITimeoutError")),
    )

    await worker.run_once()

    facts = await store.get_facts(key)
    assert facts is not None
    assert facts["bedrooms"] is None
    assert facts["bathrooms_source"] == "unknown"
    unit = await store.get_unit(key)
    assert unit is not None
    assert unit.state is UnitState.DONE


async def test_a_listing_missing_two_criteria_is_alerted_as_possible(
    store: HousingStore,
) -> None:
    """The common Phangan case: bedrooms and price stated, nothing else."""
    await _queue_unit(store, "Сдаю дом, 2 спальни, 25000 бат, фото скину в личку")
    worker = HousingWorker(
        store=store,
        extractor=FakeExtractor(
            HousingFacts(
                is_rental_offer=True,
                is_vehicle_ad=False,
                bedrooms=2,
                monthly_price_thb=25000,
            )
        ),
    )

    await worker.run_once()

    alerts = await store.claim_due_alerts(lease_owner="test")
    assert len(alerts) == 1
    assert alerts[0]["verdict"] == Verdict.POSSIBLE.value
    body = str(alerts[0]["body_html"])
    assert "Ванные" in body
    assert "Телевизор" in body


async def test_the_same_verdict_is_never_queued_twice(store: HousingStore) -> None:
    key = await _queue_unit(store, "Сдаю дом 2 спальни 30000 бат")
    first = await store.enqueue_alert(
        unit_key=key,
        chat_id=-100777,
        chat_title=None,
        telegram_msg_id=100,
        requirements_revision=1,
        verdict=Verdict.POSSIBLE,
        kind=AlertKind.LIVE,
        body_html="one",
    )
    second = await store.enqueue_alert(
        unit_key=key,
        chat_id=-100777,
        chat_title=None,
        telegram_msg_id=100,
        requirements_revision=1,
        verdict=Verdict.POSSIBLE,
        kind=AlertKind.LIVE,
        body_html="two",
    )

    assert first is not None
    assert second is None


async def test_an_upgraded_verdict_is_a_new_alert(store: HousingStore) -> None:
    """Photographs resolving the unknowns is news, not a duplicate."""
    key = await _queue_unit(store, "Сдаю дом")
    await store.enqueue_alert(
        unit_key=key,
        chat_id=-100777,
        chat_title=None,
        telegram_msg_id=100,
        requirements_revision=1,
        verdict=Verdict.POSSIBLE,
        kind=AlertKind.LIVE,
        body_html="possible",
    )

    upgraded = await store.enqueue_alert(
        unit_key=key,
        chat_id=-100777,
        chat_title=None,
        telegram_msg_id=100,
        requirements_revision=1,
        verdict=Verdict.CONFIRMED,
        kind=AlertKind.UPDATE,
        body_html="confirmed",
    )

    assert upgraded is not None


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


class FakeDispatcher:
    """A dispatcher that answers however the test needs."""

    def __init__(self, results: list[DeliveryResult]) -> None:
        self.results = results
        self.sent: list[str] = []

    async def deliver_html(self, message: str) -> DeliveryResult:
        self.sent.append(message)
        return self.results.pop(0) if self.results else DeliveryResult.success()


async def _one_pending_alert(store: HousingStore) -> None:
    key = await _queue_unit(store, "Сдаю дом")
    await store.enqueue_alert(
        unit_key=key,
        chat_id=-100777,
        chat_title="Аренда Панган",
        telegram_msg_id=100,
        requirements_revision=1,
        verdict=Verdict.POSSIBLE,
        kind=AlertKind.LIVE,
        body_html="<b>listing</b>",
    )


async def test_a_delivered_alert_is_settled(store: HousingStore) -> None:
    await _one_pending_alert(store)
    dispatcher = FakeDispatcher([DeliveryResult.success()])

    sent = await HousingAlertDelivery(
        store=store, dispatcher=dispatcher, lease_owner="test"
    ).run_once()

    assert sent == 1
    assert dispatcher.sent == ["<b>listing</b>"]
    assert await store.claim_due_alerts(lease_owner="test") == []


async def test_a_retryable_failure_comes_back_later(store: HousingStore) -> None:
    await _one_pending_alert(store)
    dispatcher = FakeDispatcher(
        [DeliveryResult(sent=False, retryable=True, error_code="timeout", retry_after=1)]
    )

    sent = await HousingAlertDelivery(
        store=store, dispatcher=dispatcher, lease_owner="test"
    ).run_once()

    assert sent == 0
    cursor = await store._conn.execute(  # noqa: SLF001 - asserting durable state
        "SELECT delivery_status, last_error FROM housing_alerts"
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["delivery_status"] == "pending"
    assert row["last_error"] == "timeout"


async def test_a_permanent_failure_is_kept_visible_not_dropped(store: HousingStore) -> None:
    """A listing the owner never saw has to stay findable."""
    await _one_pending_alert(store)
    dispatcher = FakeDispatcher(
        [DeliveryResult(sent=False, retryable=False, error_code="dispatcher_disabled")]
    )

    await HousingAlertDelivery(store=store, dispatcher=dispatcher, lease_owner="test").run_once()

    cursor = await store._conn.execute(  # noqa: SLF001 - asserting durable state
        "SELECT delivery_status, last_error FROM housing_alerts"
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["delivery_status"] == "failed"
    assert row["last_error"] == "dispatcher_disabled"


# ---------------------------------------------------------------------------
# The message the owner reads
# ---------------------------------------------------------------------------


async def test_the_alert_names_every_criterion_including_the_unknown_ones(
    store: HousingStore,
) -> None:
    key = await _queue_unit(store, "Сдаю дом 2 спальни 25000 бат")
    unit = (await store.claim_settled_units())[0]
    facts = _text_facts(
        bathrooms=None,
        bathrooms_source="unknown",
        tv_present=None,
        tv_size_class=None,
        tv_source="unknown",
        monthly_price_thb=25000,
        area_raw="Шритану",
        evidence_quote="Сдаю дом 2 спальни",
    )
    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    body = render_alert(unit, facts, result)

    assert key.startswith("m:")
    assert "Возможно подходит" in body
    assert "✅ Спальни" in body
    assert "❔ Ванные" in body
    assert "❔ Телевизор" in body
    assert "25 000 THB/мес" in body
    assert "Шритану" in body
    assert "https://t.me/c/1199262612/100" in body


def test_extraction_ignores_a_nonsense_size_class() -> None:
    """The model is schema-bound, but a bad value must not reach the matcher."""
    from pipeline.housing.extractor import _facts_from_payload

    facts = _facts_from_payload(
        {"is_rental_offer": True, "tv_size_class": "gigantic", "bedrooms": -3}
    )

    assert facts.tv_size_class == "unclear"
    assert facts.bedrooms is None


async def test_empty_text_is_reported_as_unreadable_without_a_call() -> None:
    """No text means nothing to read; spending a model call on it is waste."""
    extractor = HousingTextExtractor(client=object())

    facts = await extractor.extract("   ")

    assert facts.error == "empty_text"


# ---------------------------------------------------------------------------
# Photographs
# ---------------------------------------------------------------------------


async def test_photos_are_requested_only_when_they_could_answer_something(
    store: HousingStore,
) -> None:
    """A listing that already states everything must not spend a download."""
    key = unit_key_for(-100777, grouped_id=1, telegram_msg_id=1)
    await store.record_message(
        unit_key=key,
        chat_id=-100777,
        grouped_id=1,
        message_id=1,
        telegram_msg_id=1,
        text="Сдаю дом 2 спальни 2 санузла большой телевизор 30000 бат",
        has_media=True,
        telegram_photo_id=11,
    )
    worker = HousingWorker(
        store=store,
        extractor=FakeExtractor(
            HousingFacts(
                is_rental_offer=True,
                is_vehicle_ad=False,
                bedrooms=2,
                bathrooms=2,
                monthly_price_thb=30000,
                tv_present=True,
                tv_size_class="large",
            )
        ),
    )

    await worker.run_once()

    assert await store.next_media_download(priority="live") is None


async def test_photos_are_requested_when_a_criterion_is_unanswered(
    store: HousingStore,
) -> None:
    key = unit_key_for(-100777, grouped_id=2, telegram_msg_id=2)
    await store.record_message(
        unit_key=key,
        chat_id=-100777,
        grouped_id=2,
        message_id=2,
        telegram_msg_id=2,
        text="Сдаю дом 2 спальни 30000 бат, фото ниже",
        has_media=True,
        telegram_photo_id=22,
    )
    worker = HousingWorker(
        store=store,
        extractor=FakeExtractor(
            HousingFacts(
                is_rental_offer=True,
                is_vehicle_ad=False,
                bedrooms=2,
                monthly_price_thb=30000,
            )
        ),
    )

    await worker.run_once()

    pending = await store.next_media_download(priority="live")
    assert pending is not None
    assert pending["telegram_photo_id"] == 22


async def test_a_photograph_already_on_disk_is_not_fetched_again(
    store: HousingStore, tmp_path: Path
) -> None:
    """The same advertisement crossposted into a second chat reuses the file."""
    existing = tmp_path / "photo.jpg"
    existing.write_bytes(b"jpeg")
    first = unit_key_for(-100777, grouped_id=None, telegram_msg_id=1)
    await store.record_message(
        unit_key=first,
        chat_id=-100777,
        grouped_id=None,
        message_id=1,
        telegram_msg_id=1,
        text="Сдаю",
        has_media=True,
        telegram_photo_id=4242,
    )
    await store.enqueue_media(unit_key=first, chat_id=-100777, photos=[(1, 4242)])
    await store.settle_media(
        unit_key=first,
        telegram_msg_id=1,
        status="downloaded",
        local_path=str(existing),
        byte_size=4,
    )

    second = unit_key_for(-100888, grouped_id=None, telegram_msg_id=9)
    await store.record_message(
        unit_key=second,
        chat_id=-100888,
        grouped_id=None,
        message_id=2,
        telegram_msg_id=9,
        text="Сдаю",
        has_media=True,
        telegram_photo_id=4242,
    )
    queued = await store.enqueue_media(unit_key=second, chat_id=-100888, photos=[(9, 4242)])

    assert queued == 0
    assert await store.downloaded_media(second) == [str(existing)]


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


def test_a_visual_bathroom_count_fills_an_unknown_and_is_marked_as_visual() -> None:
    reading = VisionReading(bathrooms_visible_min=2, confidence=0.8)
    facts = _text_facts(bathrooms=None, bathrooms_source="unknown")

    merged = reading.merged_into(facts)

    assert merged["bathrooms"] == 2
    assert merged["bathrooms_source"] == "vision"
    assert merged["vision_status"] == "done"


def test_a_stated_bathroom_count_is_never_overwritten_by_a_photograph() -> None:
    """Text describes the property; a photograph describes its frame.

    The direction that matters is a HIGHER visual count against a lower stated
    one: someone wrote "1 санузел", the album shows what looks like two, and
    believing the album would turn a listing the owner ruled out into a false
    match. A lower visual count is refused by the lower-bound rule anyway, so
    testing that direction proves nothing about this guard.
    """
    reading = VisionReading(bathrooms_visible_min=2, confidence=0.9)
    facts = _text_facts(bathrooms=1, bathrooms_source="text")

    merged = reading.merged_into(facts)

    assert merged["bathrooms"] == 1
    assert merged["bathrooms_source"] == "text"
    assert match_requirements(merged, DEFAULT_REQUIREMENTS).verdict is Verdict.HARD_MISS


def test_a_low_confidence_reading_changes_nothing() -> None:
    """Stored and shown, but not allowed to decide."""
    reading = VisionReading(bathrooms_visible_min=3, tv_size_class="large", confidence=0.2)
    facts = _text_facts(bathrooms=None, bathrooms_source="unknown", tv_source="unknown")

    merged = reading.merged_into(facts)

    assert merged["bathrooms"] is None
    assert merged["vision_status"] == "done"


def test_photographs_of_something_else_change_nothing() -> None:
    """A price card or a logo is not evidence about the property."""
    reading = VisionReading(bathrooms_visible_min=2, confidence=0.9, photos_show_this_listing=False)
    facts = _text_facts(bathrooms=None, bathrooms_source="unknown")

    assert reading.merged_into(facts)["bathrooms"] is None


def test_a_television_the_photographs_missed_stays_unknown() -> None:
    """Absence from a frame is not absence from the house."""
    reading = VisionReading(tv_present=None, tv_size_class=None, confidence=0.9)
    facts = _text_facts(tv_present=None, tv_size_class=None, tv_source="unknown")

    merged = reading.merged_into(facts)

    assert merged["tv_source"] == "unknown"
    assert match_requirements(merged, DEFAULT_REQUIREMENTS).verdict is Verdict.POSSIBLE


def test_a_failed_vision_read_leaves_every_unknown_untouched() -> None:
    reading = VisionReading.unavailable("APITimeoutError")
    facts = _text_facts(bathrooms=None, bathrooms_source="unknown")

    merged = reading.merged_into(facts)

    assert merged["bathrooms"] is None
    assert merged["vision_status"] == "error"


class FakeVision:
    """A vision extractor whose reading the test chooses."""

    def __init__(self, reading: VisionReading) -> None:
        self.reading = reading
        self.calls: list[list[str]] = []

    async def read(self, paths: list[str], *, listing_text: str | None = None) -> VisionReading:
        self.calls.append(paths)
        return self.reading


async def _unit_with_photo(store: HousingStore, tmp_path: Path, facts: HousingFacts) -> str:
    key = unit_key_for(-1001199262612, grouped_id=None, telegram_msg_id=77)
    await store.record_message(
        unit_key=key,
        chat_id=-1001199262612,
        grouped_id=None,
        message_id=1,
        telegram_msg_id=77,
        text="Сдаю дом 2 спальни 30000 бат",
        has_media=True,
        telegram_photo_id=77,
    )
    await HousingWorker(store=store, extractor=FakeExtractor(facts)).run_once()
    photo = tmp_path / "listing.jpg"
    photo.write_bytes(b"jpeg-bytes")
    await store.settle_media(
        unit_key=key,
        telegram_msg_id=77,
        status="downloaded",
        local_path=str(photo),
        byte_size=10,
    )
    return key


async def test_photographs_that_complete_the_picture_upgrade_the_verdict(
    store: HousingStore, tmp_path: Path
) -> None:
    """Possible becomes confirmed, and the owner is told again — with the photo."""
    key = await _unit_with_photo(
        store,
        tmp_path,
        HousingFacts(
            is_rental_offer=True,
            is_vehicle_ad=False,
            bedrooms=2,
            monthly_price_thb=30000,
        ),
    )
    await store.claim_due_alerts(lease_owner="drain")  # the first, text-only alert

    vision = FakeVision(
        VisionReading(
            bathrooms_visible_min=2,
            tv_size_class="large",
            tv_present=True,
            confidence=0.9,
        )
    )
    read = await HousingVisionWorker(store=store, extractor=vision).run_once()

    assert read == 1
    facts = await store.get_facts(key)
    assert facts is not None
    assert facts["bathrooms"] == 2
    assert facts["bathrooms_source"] == "vision"
    alerts = await store.claim_due_alerts(lease_owner="test")
    assert len(alerts) == 1
    assert alerts[0]["kind"] == AlertKind.UPDATE.value
    assert alerts[0]["verdict"] == Verdict.CONFIRMED.value
    assert json.loads(str(alerts[0]["photo_paths_json"]))


async def test_photographs_that_change_nothing_send_no_second_alert(
    store: HousingStore, tmp_path: Path
) -> None:
    """Repeating a verdict trains the owner to ignore these messages."""
    await _unit_with_photo(
        store,
        tmp_path,
        HousingFacts(
            is_rental_offer=True,
            is_vehicle_ad=False,
            bedrooms=2,
            monthly_price_thb=30000,
        ),
    )
    await store.claim_due_alerts(lease_owner="drain")

    vision = FakeVision(VisionReading(bathrooms_visible_min=None, confidence=0.9))
    await HousingVisionWorker(store=store, extractor=vision).run_once()

    assert await store.claim_due_alerts(lease_owner="test") == []


async def test_a_unit_with_no_downloaded_photographs_is_not_read(
    store: HousingStore,
) -> None:
    """Vision waits for the files rather than reading an empty list."""
    await _queue_unit(store, "Сдаю дом 2 спальни 30000 бат")
    await HousingWorker(
        store=store,
        extractor=FakeExtractor(
            HousingFacts(
                is_rental_offer=True,
                is_vehicle_ad=False,
                bedrooms=2,
                monthly_price_thb=30000,
            )
        ),
    ).run_once()

    vision = FakeVision(VisionReading(bathrooms_visible_min=2, confidence=0.9))
    read = await HousingVisionWorker(store=store, extractor=vision).run_once()

    assert read == 0
    assert vision.calls == []
