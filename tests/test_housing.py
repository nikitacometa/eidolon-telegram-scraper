"""Tests for the housing subsystem: units, requirements, verdicts, alerts."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.governor import ActionStatus
from pipeline.housing.extractor import HousingFacts, HousingTextExtractor
from pipeline.housing.gate import could_be_housing
from pipeline.housing.media import MAX_BYTES, MediaDownloadWorker
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
        "property_type": "house",
        "property_type_source": "text",
        "terrace": 1,
        "terrace_source": "text",
        "private_setting": 1,
        "nature_setting": 1,
    }
    facts.update(overrides)
    return facts


# The pre-preferences shape: bathrooms hard, tv at the top level. Kept as a
# named fixture because deployed revisions of exactly this shape exist and
# must keep working.
LEGACY_REQUIREMENTS: dict[str, object] = {
    "bedrooms": {"operator": "at_least", "value": 2},
    "bathrooms": {"operator": "at_least", "value": 2},
    "tv": {"minimum_class": "large"},
    "monthly_rent_thb": {"min": 20000, "max": 40000},
}


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
    assert result.preference_score == 100


def test_an_unstated_bathroom_count_is_possible_not_a_rejection() -> None:
    """The failure this whole design exists to prevent.

    Phangan listings state bathrooms 2.3% of the time. Reading silence as
    "does not have two" would reject essentially every real advertisement and
    the owner would receive nothing at all.
    """
    facts = _text_facts(bathrooms=None, bathrooms_source="unknown")

    result = match_requirements(facts, LEGACY_REQUIREMENTS)

    assert result.verdict is Verdict.POSSIBLE
    assert "bathrooms" in result.unknown_fields


def test_a_listing_with_no_photos_and_no_details_still_reaches_the_owner() -> None:
    """ "Фото скину в личку" is a normal listing, not a non-match."""
    facts = _text_facts(
        property_type=None,
        property_type_source="unknown",
        tv_present=None,
        tv_size_class=None,
        tv_source="unknown",
    )

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.POSSIBLE
    assert set(result.unknown_fields) == {"property_type"}
    assert "tv" in result.unknown_preferences


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

    result = match_requirements(facts, LEGACY_REQUIREMENTS)

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


def test_a_television_nobody_mentioned_is_an_unknown_preference() -> None:
    """The TV moved to the soft tier: silence neither rejects nor scores."""
    facts = _text_facts(tv_present=None, tv_size_class=None, tv_source="unknown")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.CONFIRMED
    assert "tv" in result.unknown_preferences
    assert result.preference_score == 70  # terrace 25 + privacy 25 + nature 20


def test_a_television_reported_absent_costs_score_not_the_listing() -> None:
    """Under the old rules a stated "no TV" rejected outright; the owner's
    television is a wish, not a dealbreaker — it now only loses its points."""
    facts = _text_facts(tv_present=0, tv_size_class="none", tv_source="text")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.CONFIRMED
    tv = [p for p in result.preferences if p.field == "tv"]
    assert [p.state for p in tv] == [FieldState.VIOLATED]
    assert result.preference_score == 70


def test_a_small_television_scores_nothing_against_a_large_wish() -> None:
    facts = _text_facts(tv_size_class="small", tv_source="text")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.CONFIRMED
    assert result.preference_score == 70


def test_a_stated_apartment_rejects_under_a_house_requirement() -> None:
    facts = _text_facts(property_type="apartment")

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    assert result.verdict is Verdict.HARD_MISS
    verdicts = {f.field: f.state for f in result.fields}
    assert verdicts["property_type"] is FieldState.VIOLATED


def test_a_photograph_can_confirm_a_house_but_never_reject_one() -> None:
    """A frame of a building says nothing about which unit is offered."""
    confirmed = match_requirements(
        _text_facts(property_type="house", property_type_source="vision"),
        DEFAULT_REQUIREMENTS,
    )
    mismatched = match_requirements(
        _text_facts(property_type="apartment", property_type_source="vision"),
        DEFAULT_REQUIREMENTS,
    )

    assert {f.field: f.state for f in confirmed.fields}["property_type"] is (FieldState.SATISFIED)
    assert {f.field: f.state for f in mismatched.fields}["property_type"] is (FieldState.UNKNOWN)


def test_an_old_revision_still_matches_with_its_tv_read_as_a_preference() -> None:
    """Deployed revisions carry tv at the top level; they must keep working,
    with the television judged as a wish rather than a dealbreaker."""
    facts = _text_facts(tv_present=0, tv_size_class="none", tv_source="text")
    legacy = {
        "bedrooms": {"operator": "at_least", "value": 2},
        "tv": {"minimum_class": "large"},
        "monthly_rent_thb": {"min": 20000, "max": 40000},
    }

    result = match_requirements(facts, legacy)

    assert result.verdict is Verdict.CONFIRMED
    assert [p.field for p in result.preferences] == ["tv"]
    assert result.preference_score == 0


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
        validate_requirements(
            {
                "bedrooms": {"operator": "at_least", "value": 2},
                "preferences": {"tv": {"minimum_class": "enormous"}},
            }
        )


def test_a_legacy_revision_validates_into_the_new_shape() -> None:
    """The deployed revision 2 predates preferences and must stay usable."""
    cleaned = validate_requirements(
        {
            "bedrooms": {"operator": "at_least", "value": 2},
            "tv": {"minimum_class": "large"},
            "monthly_rent_thb": {"min": 20000, "max": 40000},
        }
    )

    assert "tv" not in cleaned
    assert cleaned["preferences"]["tv"] == {"minimum_class": "large", "weight": 30}


def test_a_preferences_only_document_is_refused() -> None:
    """With no hard tier every listing is confirmed; that is not a filter."""
    with pytest.raises(RequirementsError, match="hard criterion"):
        validate_requirements({"preferences": {"terrace": {"weight": 50}}})


def test_an_unknown_property_type_is_refused() -> None:
    with pytest.raises(RequirementsError, match="property_type"):
        validate_requirements({"property_type": {"require": "yurt"}})


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
    # These fixtures are about what the subsystem does with an advertisement,
    # so the chat is a rentals board and the lexical gate does not apply. The
    # gate has its own tests below.
    await store.set_chat_kind(chat_id, "dedicated_housing")
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
                property_type="house",
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
    # "дом" in the text is extracted as the property type by the real
    # extractor; the fake here reports nothing, so the type stays unknown
    # and the verdict is possible rather than confirmed.
    assert alerts[0]["verdict"] == Verdict.POSSIBLE.value
    body = str(alerts[0]["body_html"])
    assert "Тип жилья" in body
    assert "Телевизор" in body.lower() or "телевизор" in body.lower()
    assert "Хотелки" in body


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
    # With no owner configured the bot path remains, and it appends the link
    # at delivery time so the stored body stays format-neutral.
    assert dispatcher.sent == [
        '<b>listing</b>\n<a href="https://t.me/c/777/100">Открыть в Telegram</a>'
    ]
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
        property_type=None,
        property_type_source="unknown",
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
    assert "❔ Тип жилья" in body
    assert "Хотелки 70%" in body
    assert "❔ телевизор" in body
    assert "✅ терраса" in body
    assert "25 000 THB/мес" in body
    assert "Шритану" in body
    # The original is forwarded right after the report, so the report carries
    # neither a quotation nor a link into a chat the owner cannot open.
    assert "t.me" not in body
    assert "<blockquote>" not in body
    assert body.endswith("⬇️ Оригинал ниже")


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
    await store.set_chat_kind(-100777, "dedicated_housing")
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
                property_type="house",
                terrace=True,
            )
        ),
    )

    await worker.run_once()

    assert await store.next_media_download(priority="live") is None


async def test_photos_are_requested_when_a_criterion_is_unanswered(
    store: HousingStore,
) -> None:
    await store.set_chat_kind(-100777, "dedicated_housing")
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
    assert match_requirements(merged, LEGACY_REQUIREMENTS).verdict is Verdict.HARD_MISS


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
    result = match_requirements(merged, DEFAULT_REQUIREMENTS)
    assert "tv" in result.unknown_preferences


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
    await store.set_chat_kind(-1001199262612, "dedicated_housing")
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
            property_type_visible="house",
            confidence=0.9,
        )
    )
    read = await HousingVisionWorker(store=store, extractor=vision).run_once()

    assert read == 1
    facts = await store.get_facts(key)
    assert facts is not None
    assert facts["bathrooms"] == 2
    assert facts["bathrooms_source"] == "vision"
    assert facts["property_type"] == "house"
    assert facts["property_type_source"] == "vision"
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


class FakeGovernor:
    """A governor that runs the call and reports success."""

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, kind: object, key: str, call: object, **_: object) -> object:
        self.calls += 1
        value = await call()  # type: ignore[operator]
        return SimpleNamespace(
            status=ActionStatus.OK,
            value=value,
            ok=True,
            error_code=None,
            retry_after_seconds=None,
        )


class FakeTelegramClient:
    """A Telegram that hands back a message and writes a file of a chosen size."""

    def __init__(self, *, size: int) -> None:
        self.size = size
        self.downloads = 0

    async def get_messages(self, chat_id: int, ids: int) -> object:
        return SimpleNamespace(id=ids, media=object())

    async def download_media(self, message: object, file: str) -> str:
        self.downloads += 1
        await asyncio.to_thread(Path(file).write_bytes, b"x" * self.size)
        return file


async def test_an_oversized_photograph_is_settled_not_retried_forever(
    store: HousingStore, tmp_path: Path
) -> None:
    """A file too large to keep must stop being fetched.

    Raising on the size check escapes the governor, so the row stays pending
    and the same enormous file is downloaded again on every single cycle —
    spending both the budget and the bandwidth with nothing to show.
    """
    key = unit_key_for(-100777, grouped_id=None, telegram_msg_id=3)
    await store.record_message(
        unit_key=key,
        chat_id=-100777,
        grouped_id=None,
        message_id=1,
        telegram_msg_id=3,
        text="Сдаю",
        has_media=True,
        telegram_photo_id=3,
    )
    await store.enqueue_media(unit_key=key, chat_id=-100777, photos=[(3, 3)])
    client = FakeTelegramClient(size=MAX_BYTES + 1)
    worker = MediaDownloadWorker(
        store=store,
        client=client,
        governor=FakeGovernor(),
        media_root=tmp_path,
    )

    assert await worker.run_once() is True
    assert await worker.run_once() is False, "the oversized row must not come back"
    assert client.downloads == 1
    assert await store.downloaded_media(key) == []


async def test_a_normal_photograph_is_stored_with_its_path(
    store: HousingStore, tmp_path: Path
) -> None:
    key = unit_key_for(-100777, grouped_id=None, telegram_msg_id=4)
    await store.record_message(
        unit_key=key,
        chat_id=-100777,
        grouped_id=None,
        message_id=1,
        telegram_msg_id=4,
        text="Сдаю",
        has_media=True,
        telegram_photo_id=4,
    )
    await store.enqueue_media(unit_key=key, chat_id=-100777, photos=[(4, 4)])
    worker = MediaDownloadWorker(
        store=store,
        client=FakeTelegramClient(size=2048),
        governor=FakeGovernor(),
        media_root=tmp_path,
    )

    assert await worker.run_once() is True

    stored = await store.downloaded_media(key)
    assert len(stored) == 1
    assert await asyncio.to_thread(Path(stored[0]).read_bytes) == b"x" * 2048


# ---------------------------------------------------------------------------
# The pre-gate
# ---------------------------------------------------------------------------


def test_the_gate_passes_a_listing_in_any_of_its_usual_forms() -> None:
    for text in [
        "Сдаю дом на длительный срок, 2 спальни, есть кондей и телевизор",
        "Пересдаю свою виллу с 1 сентября, 25000 бат в месяц, депозит месяц",
        "House for rent in Sri Thanu, 2 bedrooms, long term, 30k",
        "Освобождается студия рядом с закатным пляжем, недорого, пишите в лс",
        "Свободна квартира с 15 числа, всё есть, 18 000",
    ]:
        assert could_be_housing(text), text


def test_the_gate_drops_ordinary_island_conversation() -> None:
    """Including the words that merely CONTAIN a housing stem.

    "рядом" ends in "дом" and "комнате" is fine but "укомплектован" is not a
    room; without word boundaries the gate passed a question about a dentist
    and would have sent half the island's conversation to the model. This is
    the case that actually occurred, not an invented one.
    """
    for text in [
        "У меня эти птицы как-то напали на котенка, вот сели в кустах и мяукали",
        "Приглашаю на проф.процедуру по Шугарингу (для женщин и мужчин). Велком!",
        "Кто знает, где починить байк недалеко от Тонг Салы? Спасибо заранее",
        "Кто подскажет хорошего стоматолога рядом с пирсом? Заранее спасибо",
        "Гуляли рядом с водопадом, вода тёплая, всем советую сходить туда",
    ]:
        assert not could_be_housing(text), text


def test_the_gate_ignores_a_message_too_short_to_be_a_listing() -> None:
    assert not could_be_housing("сдаю")
    assert not could_be_housing("")
    assert not could_be_housing(None)


def test_the_gate_reads_a_spaced_price() -> None:
    """Prices in this corpus are written "15 000" as often as "15000"."""
    assert could_be_housing("Освобождается с первого числа, всё включено, 15 000 в месяц")


async def test_a_dedicated_rentals_chat_is_never_gated(store: HousingStore) -> None:
    """On a board where everything is a listing, the gate must not apply.

    A short or oddly-worded advertisement there is still an advertisement, and
    the whole board is small enough to read in full.
    """
    await store.set_chat_kind(-100999, "dedicated_housing")
    key = unit_key_for(-100999, grouped_id=None, telegram_msg_id=1)
    await store.record_message(
        unit_key=key,
        chat_id=-100999,
        grouped_id=None,
        message_id=1,
        telegram_msg_id=1,
        text="Освобождается с первого, тихо, вид на джунгли, пишите",
        has_media=False,
        telegram_photo_id=None,
    )
    extractor = FakeExtractor(HousingFacts(is_rental_offer=True, is_vehicle_ad=False, bedrooms=2))

    await HousingWorker(store=store, extractor=extractor).run_once()

    assert len(extractor.calls) == 1


async def test_a_general_chat_message_that_is_only_chatter_never_reaches_the_model(
    store: HousingStore,
) -> None:
    key = unit_key_for(-100777, grouped_id=None, telegram_msg_id=2)
    await store.record_message(
        unit_key=key,
        chat_id=-100777,
        grouped_id=None,
        message_id=1,
        telegram_msg_id=2,
        text="Кто знает, где починить байк недалеко от Тонг Салы? Спасибо заранее",
        has_media=False,
        telegram_photo_id=None,
    )
    extractor = FakeExtractor(HousingFacts(is_rental_offer=True, is_vehicle_ad=False))

    await HousingWorker(store=store, extractor=extractor).run_once()

    assert extractor.calls == []
    unit = await store.get_unit(key)
    assert unit is not None
    assert unit.state is UnitState.DONE


# ---------------------------------------------------------------------------
# The television the model invented
# ---------------------------------------------------------------------------


def test_a_claimed_absent_television_is_ignored_when_the_text_never_mentions_one() -> None:
    """Measured on 305 real listings: the model called 194 of them TV-less.

    162 of those never mention a television in any form — it was reading its
    own silence as the property's. Believing it would have rejected all 162
    outright, since a stated absence is a violation.
    """
    facts = HousingFacts(
        is_rental_offer=True,
        is_vehicle_ad=False,
        bedrooms=2,
        tv_present=False,
        tv_size_class="none",
    )

    row = facts.as_row(unit_version=1, source_text="Сдаю дом, 2 спальни, кондиционер, кухня")

    assert row["tv_present"] is None
    assert row["tv_size_class"] is None
    assert row["tv_source"] == "unknown"
    assert match_requirements(row, DEFAULT_REQUIREMENTS).verdict is Verdict.POSSIBLE


def test_an_absent_television_the_text_actually_states_is_believed() -> None:
    """A seller who writes that there is no TV is evidence: the claim
    survives the fabrication guard and marks the preference violated —
    under EVERY requirements shape, legacy top-level tv included, since a
    television is a wish now and never rejects."""
    facts = HousingFacts(
        is_rental_offer=True,
        is_vehicle_ad=False,
        bedrooms=2,
        bathrooms=2,
        monthly_price_thb=30000,
        tv_present=False,
        tv_size_class="none",
    )

    row = facts.as_row(
        unit_version=1,
        source_text="Сдаю дом, 2 спальни, 2 санузла, 30000. Телевизора нет, только проектор",
    )

    assert row["tv_source"] == "text"
    for requirements in (LEGACY_REQUIREMENTS, DEFAULT_REQUIREMENTS):
        result = match_requirements(row, requirements)
        assert result.verdict is not Verdict.HARD_MISS
        tv_states = [p.state for p in result.preferences if p.field == "tv"]
        assert tv_states == [FieldState.VIOLATED]


def test_a_television_the_text_describes_survives_the_guard() -> None:
    facts = HousingFacts(
        is_rental_offer=True, is_vehicle_ad=False, tv_present=True, tv_size_class="large"
    )

    row = facts.as_row(unit_version=1, source_text="Большой телевизор, кондиционер")

    assert row["tv_size_class"] == "large"
    assert row["tv_source"] == "text"


async def test_a_unit_abandoned_mid_processing_is_swept_again(store: HousingStore) -> None:
    """A daemon killed mid-_process leaves a unit in an intermediate state;
    the sweep must reclaim it instead of leaving it stuck forever."""
    key = await _queue_unit(store, "Сдаю дом 2 спальни 30000 бат")
    claimed = await store.claim_settled_units()
    assert [unit.unit_key for unit in claimed] == [key]
    # Simulate a crash right after the state moved to 'extracting'.
    await store.set_unit_state(key, UnitState.EXTRACTING)

    # Fresh abandonment is not stale yet: nothing to reclaim.
    assert await store.claim_settled_units() == []

    async with store._write_lock:
        await store._conn.execute(
            "UPDATE housing_live_units SET updated_at = datetime('now', '-16 minutes')"
            " WHERE unit_key = ?",
            (key,),
        )
        await store._conn.commit()

    reclaimed = await store.claim_settled_units()
    assert [unit.unit_key for unit in reclaimed] == [key]


async def test_a_poisoned_unit_is_retried_a_bounded_number_of_times(
    store: HousingStore,
) -> None:
    """One advertisement that keeps failing must not burn a model call per
    sweep forever."""
    key = await _queue_unit(store, "Сдаю дом 2 спальни 30000 бат")
    await store.claim_settled_units()
    for _ in range(store.MAX_SWEEP_ATTEMPTS):
        await store.set_unit_state(key, UnitState.ERROR, error="Boom")
        async with store._write_lock:
            await store._conn.execute(
                "UPDATE housing_live_units SET updated_at = datetime('now', '-16 minutes')"
                " WHERE unit_key = ?",
                (key,),
            )
            await store._conn.commit()
        await store.claim_settled_units()

    await store.set_unit_state(key, UnitState.ERROR, error="Boom")
    async with store._write_lock:
        await store._conn.execute(
            "UPDATE housing_live_units SET updated_at = datetime('now', '-16 minutes')"
            " WHERE unit_key = ?",
            (key,),
        )
        await store._conn.commit()

    assert await store.claim_settled_units() == []


async def test_vision_status_stays_pending_until_the_alert_is_queued(
    store: HousingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between the vision read and the alert must leave the unit
    eligible for another read, not stranded 'done' with a stale verdict."""
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
    await store.claim_due_alerts(lease_owner="drain")

    class Killed(BaseException):
        """A process death, which no except-Exception handler sees."""

    async def crash(**kwargs: object) -> None:
        raise Killed

    monkeypatch.setattr(store, "record_match_with_alert", crash)
    vision = FakeVision(
        VisionReading(
            bathrooms_visible_min=2,
            tv_size_class="large",
            tv_present=True,
            property_type_visible="house",
            confidence=0.9,
        )
    )
    with pytest.raises(Killed):
        await HousingVisionWorker(store=store, extractor=vision).run_once()

    facts = await store.get_facts(key)
    assert facts is not None
    assert facts["vision_status"] == "pending"
    # The unit must still be eligible for another read after a restart;
    # 'done' with no alert queued is the stranded state this prevents.
    assert key in await store.units_awaiting_vision()


async def test_a_claim_alone_does_not_consume_a_delivery_attempt(
    store: HousingStore,
) -> None:
    """Claiming is a lease; only a completed delivery attempt counts against
    the retry budget."""
    key = await _queue_unit(store, "Сдаю дом 2 спальни 30000 бат")
    await store.enqueue_alert(
        unit_key=key,
        chat_id=-1001199262612,
        chat_title=None,
        telegram_msg_id=100,
        requirements_revision=1,
        verdict=Verdict.POSSIBLE,
        kind=AlertKind.LIVE,
        body_html="<b>x</b>",
    )

    first = await store.claim_due_alerts(lease_owner="one", lease_seconds=0)
    assert first[0]["attempts"] == 0
    second = await store.claim_due_alerts(lease_owner="two", lease_seconds=0)
    assert second[0]["attempts"] == 0

    await store.settle_alert(int(second[0]["id"]), delivered=False, retry_in_seconds=0)
    third = await store.claim_due_alerts(lease_owner="three", lease_seconds=0)
    assert third[0]["attempts"] == 1


async def test_photo_requests_are_capped_at_what_vision_reads(store: HousingStore) -> None:
    """Downloading a fifteen-photo album spends budget on frames vision
    will never look at."""
    chat_id = -1001199262612
    await store.set_chat_kind(chat_id, "dedicated_housing")
    key = unit_key_for(chat_id, grouped_id=555, telegram_msg_id=200)
    for i in range(10):
        await store.record_message(
            unit_key=key,
            chat_id=chat_id,
            grouped_id=555,
            message_id=i + 1,
            telegram_msg_id=200 + i,
            text="Сдаю дом 2 спальни 30000 бат" if i == 0 else None,
            has_media=True,
            telegram_photo_id=9000 + i,
        )
    worker = HousingWorker(
        store=store,
        extractor=FakeExtractor(
            HousingFacts(
                is_rental_offer=True, is_vehicle_ad=False, bedrooms=2, monthly_price_thb=30000
            )
        ),
    )
    await worker.run_once()

    async with store._write_lock:
        cursor = await store._conn.execute(
            "SELECT COUNT(*) FROM housing_media WHERE unit_key = ?", (key,)
        )
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 6


async def test_a_vision_verdict_and_its_alert_land_together(
    store: HousingStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A death inside the verdict write must not leave a recorded verdict
    whose alert never existed — the retry would then see 'nothing changed'
    and the owner would never hear about the upgrade."""
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
    await store.claim_due_alerts(lease_owner="drain")

    class Killed(BaseException):
        pass

    real_execute = store._conn.execute

    async def sabotaged(sql: str, *args: object, **kwargs: object) -> object:
        # Die exactly between the two writes of the one transaction.
        if "INSERT INTO housing_alerts" in sql:
            raise Killed
        return await real_execute(sql, *args, **kwargs)

    monkeypatch.setattr(store._conn, "execute", sabotaged)
    vision = FakeVision(
        VisionReading(
            bathrooms_visible_min=2,
            tv_size_class="large",
            tv_present=True,
            property_type_visible="house",
            confidence=0.9,
        )
    )
    with pytest.raises(Killed):
        await HousingVisionWorker(store=store, extractor=vision).run_once()
    monkeypatch.setattr(store._conn, "execute", real_execute)

    # Neither write survived: the pass rolled back whole, so the retry will
    # compare against the OLD verdict and queue the upgrade alert properly.
    match = await store.latest_match(key)
    assert match is not None
    assert match["verdict"] == Verdict.POSSIBLE.value
    assert key in await store.units_awaiting_vision()


def test_a_fabricated_apartment_claim_is_refused_by_the_guard() -> None:
    """The model's measured failure mode, ported to the one new field that
    can reject: a property type of apartment/room/hotel must be backed by
    the text actually containing that vocabulary."""
    from pipeline.housing.extractor import property_typed

    fabricated = property_typed("Сдаю жильё 2 спальни 30000 бат", claimed="apartment")
    corroborated = property_typed("Сдаётся квартира в кондо, 2 спальни", claimed="apartment")
    house = property_typed("Сдаю жильё у моря", claimed="house")

    assert fabricated is None
    assert corroborated == "apartment"
    # "house" cannot reject anything, so it passes without corroboration.
    assert house == "house"


def test_the_guard_reaches_the_stored_row() -> None:
    """as_row is where the guard must fire — a guard nobody calls is prose."""
    facts = HousingFacts(
        is_rental_offer=True,
        is_vehicle_ad=False,
        bedrooms=2,
        property_type="apartment",
    )

    row = facts.as_row(unit_version=1, source_text="Сдаю жильё 2 спальни 30000 бат")

    assert row["property_type"] is None
    assert row["property_type_source"] == "unknown"


def test_preference_weights_shape_the_score() -> None:
    """The score is the weighted share of CONFIRMED preferences."""
    facts = _text_facts(
        terrace=None,
        terrace_source="unknown",
        private_setting=None,
        nature_setting=None,
    )

    result = match_requirements(facts, DEFAULT_REQUIREMENTS)

    # Only the TV (weight 30 of 100) is confirmed.
    assert result.preference_score == 30


def test_a_mixed_vocabulary_listing_cannot_be_rejected_on_its_type() -> None:
    """34.9% of the corpus mentions both vocabularies in one message — the
    exact case the model mis-keys on. There the only honest answer is
    unknown, which can reject nothing."""
    from pipeline.housing.extractor import property_typed

    mixed = property_typed("Сдаю дом рядом с кондо-комплексом, 2 спальни", claimed="apartment")
    clean = property_typed("Сдаётся студия в кондо", claimed="apartment")

    assert mixed is None
    assert clean == "apartment"


def test_a_room_offered_inside_a_house_still_rejects() -> None:
    """ "Комната в общем доме" mentions both vocabularies, but the meaning is
    unambiguous: the house is the container, the offer is the room."""
    from pipeline.housing.extractor import property_typed

    ru = property_typed("Сдаётся комната в общем доме на Шритану", claimed="room")
    en = property_typed("Private room in our shared villa, 8000 THB", claimed="room")
    house_of_rooms = property_typed("Сдаю дом с 3 комнатами", claimed="room")

    assert ru == "room"
    assert en == "room"
    # A house described by its room count is not a room offer; the guard
    # refuses the fabrication-prone reading.
    assert house_of_rooms is None


async def test_a_loosened_requirement_digests_the_newly_admitted(
    store: HousingStore,
) -> None:
    """An edit re-judges the archive without model calls and reports the
    newly admitted listings once, as one message — not one alert each."""
    key = await _queue_unit(store, "Сдаётся квартира в кондо, 2 спальни, 30000 бат")
    worker = HousingWorker(
        store=store,
        extractor=FakeExtractor(
            HousingFacts(
                is_rental_offer=True,
                is_vehicle_ad=False,
                bedrooms=2,
                monthly_price_thb=30000,
                property_type="apartment",
            )
        ),
    )
    await worker.run_once()
    match = await store.latest_match(key)
    assert match is not None
    assert match["verdict"] == Verdict.HARD_MISS.value
    assert await store.claim_due_alerts(lease_owner="drain") == []

    # The owner decides an apartment is fine after all.
    revision, _ = (await store.active_requirements()) or (0, {})
    await store.save_requirements(
        definition={
            "bedrooms": {"operator": "at_least", "value": 2},
            "monthly_rent_thb": {"min": 20000, "max": 40000},
        },
        created_by="test",
        expected_revision=revision,
    )

    await worker.run_once()

    alerts = await store.claim_due_alerts(lease_owner="test")
    assert len(alerts) == 1
    assert alerts[0]["kind"] == AlertKind.DIGEST.value
    body = str(alerts[0]["body_html"])
    assert "Теперь подходят ещё 1" in body
    assert "30 000 THB" in body

    # The sweep is complete and never repeats itself: the generation is
    # marked, not merely deduplicated away — an unmarked sweep would re-read
    # the whole archive on every one-second poll forever.
    assert await store.pending_rematch_generation() is None
    await store.settle_alert(int(alerts[0]["id"]), delivered=True)
    await worker.run_once()
    assert await store.claim_due_alerts(lease_owner="again") == []


async def test_an_interrupted_rematch_sweep_loses_nothing(
    store: HousingStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A death mid-sweep must repeat the whole sweep: verdicts committed
    without their digest would read as 'nothing changed' on the retry and
    the newly admitted listing would never be reported."""
    await _queue_unit(store, "Сдаётся квартира в кондо, 2 спальни, 30000 бат")
    worker = HousingWorker(
        store=store,
        extractor=FakeExtractor(
            HousingFacts(
                is_rental_offer=True,
                is_vehicle_ad=False,
                bedrooms=2,
                monthly_price_thb=30000,
                property_type="apartment",
            )
        ),
    )
    await worker.run_once()
    revision, _ = (await store.active_requirements()) or (0, {})
    await store.save_requirements(
        definition={"bedrooms": {"operator": "at_least", "value": 2}},
        created_by="test",
        expected_revision=revision,
    )

    class Killed(BaseException):
        pass

    real_execute = store._conn.execute

    async def sabotaged(sql: str, *args: object, **kwargs: object) -> object:
        if "INSERT INTO housing_alerts" in sql:
            raise Killed
        return await real_execute(sql, *args, **kwargs)

    monkeypatch.setattr(store._conn, "execute", sabotaged)
    with pytest.raises(Killed):
        await worker.run_once()
    monkeypatch.setattr(store._conn, "execute", real_execute)

    # Nothing landed: the sweep is still pending and the retry reports it.
    assert await store.pending_rematch_generation() is not None
    await worker.run_once()
    alerts = await store.claim_due_alerts(lease_owner="test")
    assert [a["kind"] for a in alerts] == [AlertKind.DIGEST.value]
    assert await store.pending_rematch_generation() is None


async def test_old_settled_housing_units_age_out_with_the_retention_window(
    db: Database, store: HousingStore
) -> None:
    """Without housing purge the archive grows forever and every
    requirements edit re-judges an ever-larger sweep."""
    old_key = await _queue_unit(store, "Сдаю дом 2 спальни 30000 бат")
    await store.claim_settled_units()
    await store.set_unit_state(old_key, UnitState.DONE)
    async with store._write_lock:
        await store._conn.execute(
            "UPDATE housing_live_units SET created_at = datetime('now', '-40 days')"
            " WHERE unit_key = ?",
            (old_key,),
        )
        await store._conn.commit()
    fresh_key = unit_key_for(-1001199262612, grouped_id=None, telegram_msg_id=999)
    await store.record_message(
        unit_key=fresh_key,
        chat_id=-1001199262612,
        grouped_id=None,
        message_id=2,
        telegram_msg_id=999,
        text="Сдаю виллу 35000",
        has_media=False,
        telegram_photo_id=None,
    )

    await db.purge_expired_data(30)

    assert await store.get_unit(old_key) is None
    assert await store.get_unit(fresh_key) is not None
