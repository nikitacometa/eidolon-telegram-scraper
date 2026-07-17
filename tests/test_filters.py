"""Tests for pipeline/filters.py — rule-based keyword filter."""

import pytest

from config.watchers import Watcher, WatcherRules
from pipeline.filters import RuleFilter


@pytest.fixture
def housing_watcher() -> Watcher:
    """A watcher mimicking the phangan-housing config."""
    return Watcher(
        name="phangan-housing",
        chats=[-100123],
        rules=WatcherRules(
            keywords=["house", "villa", "rent", "сдаю", "аренда"],
            keywords_negative=["looking for", "ищу", "need"],
            min_length=20,
        ),
    )


@pytest.fixture
def empty_watcher() -> Watcher:
    """A watcher with no keyword rules."""
    return Watcher(
        name="catch-all",
        chats=[-100999],
        rules=WatcherRules(),
    )


class TestRuleFilter:
    def test_keyword_match_english(self, housing_watcher: Watcher) -> None:
        """Should pass when a positive keyword is found."""
        f = RuleFilter(housing_watcher)
        result = f.check("Beautiful villa for rent near the beach, 3 bedrooms")
        assert result.passed is True
        assert result.reason == "keyword_match"

    def test_keyword_match_russian(self, housing_watcher: Watcher) -> None:
        """Should match Russian keywords."""
        f = RuleFilter(housing_watcher)
        result = f.check("Сдаю дом на Пангане, 2 спальни, рядом с морем")
        assert result.passed is True
        assert result.matched_keyword == "сдаю"

    def test_keyword_case_insensitive(self, housing_watcher: Watcher) -> None:
        """Keywords should match regardless of case."""
        f = RuleFilter(housing_watcher)
        result = f.check("VILLA FOR RENT — spacious 2BR near Srithanu")
        assert result.passed is True

    def test_negative_keyword_blocks(self, housing_watcher: Watcher) -> None:
        """Negative keywords should block even if positive keywords match."""
        f = RuleFilter(housing_watcher)
        result = f.check("Looking for a house to rent near the beach for January")
        assert result.passed is False
        assert result.reason == "negative_keyword"

    def test_negative_keyword_russian(self, housing_watcher: Watcher) -> None:
        """Russian negative keywords should also block."""
        f = RuleFilter(housing_watcher)
        result = f.check("Ищу дом на аренду в районе Шритану, бюджет 15к")
        assert result.passed is False
        assert result.reason == "negative_keyword"

    def test_min_length_rejects_short(self, housing_watcher: Watcher) -> None:
        """Messages shorter than min_length should be rejected."""
        f = RuleFilter(housing_watcher)
        result = f.check("rent house")  # 10 chars < 20
        assert result.passed is False
        assert result.reason == "too_short"

    def test_no_keyword_match(self, housing_watcher: Watcher) -> None:
        """Messages without any keyword should not pass."""
        f = RuleFilter(housing_watcher)
        result = f.check("Hey everyone, what's the best restaurant near Thong Sala?")
        assert result.passed is False
        assert result.reason == "no_keyword_match"

    def test_empty_message(self, housing_watcher: Watcher) -> None:
        """None or empty text should not pass."""
        f = RuleFilter(housing_watcher)
        assert not f.check(None)
        assert not f.check("")

    def test_no_rules_passes_everything(self, empty_watcher: Watcher) -> None:
        """Watcher with no keywords should pass all messages."""
        f = RuleFilter(empty_watcher)
        result = f.check("Any random message at all")
        assert result.passed is True
        assert result.reason == "no_rules"

    def test_bool_conversion(self, housing_watcher: Watcher) -> None:
        """FilterResult should be truthy when passed, falsy when not."""
        f = RuleFilter(housing_watcher)
        assert f.check("Beautiful villa for rent, 3 bedrooms, Srithanu area")
        assert not f.check("Hey, what's up?")

    @pytest.mark.parametrize(
        "text,should_pass",
        [
            ("Spacious house available for rent on Phangan island", True),
            ("Villa near the beach, 25000 THB per month", True),
            ("Need a bungalow for February, any suggestions?", False),
            ("Сдаю квартиру на Пангане, 3 спальни, бассейн, вид на море", True),
            ("Нужен дом рядом с пляжем", False),
            ("Just photos from yesterday's sunset", False),
        ],
    )
    def test_various_messages(self, housing_watcher: Watcher, text: str, should_pass: bool) -> None:
        """Parametrized test with various real-world-like messages."""
        f = RuleFilter(housing_watcher)
        result = f.check(text)
        assert result.passed is should_pass, f"Expected {should_pass} for: {text!r}, got {result}"
