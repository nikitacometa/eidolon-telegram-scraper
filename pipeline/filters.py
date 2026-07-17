"""Rule-based message filter (Level 1) — zero-cost keyword/regex matching."""

from __future__ import annotations

import logging
import re

from config.watchers import Watcher

logger = logging.getLogger(__name__)


class RuleFilter:
    """Level 1 filter: keyword matching, negative keywords, min length."""

    def __init__(self, watcher: Watcher) -> None:
        self.watcher = watcher
        self.rules = watcher.rules

        # Match complete words/phrases so `rent` does not match `current`.
        self._positive_patterns = [
            (keyword, _keyword_pattern(keyword)) for keyword in self.rules.keywords
        ]
        self._negative_patterns = [
            (keyword, _keyword_pattern(keyword)) for keyword in self.rules.keywords_negative
        ]

    def check(self, text: str | None) -> FilterResult:
        """Run all rules against the message text.

        Returns a FilterResult indicating pass/fail and the reason.
        """
        if not text:
            return FilterResult(passed=False, reason="empty_message")

        # Min length check
        if len(text) < self.rules.min_length:
            return FilterResult(passed=False, reason="too_short")

        # Negative keyword check (reject if found)
        for keyword, pattern in self._negative_patterns:
            if pattern.search(text):
                return FilterResult(
                    passed=False,
                    reason="negative_keyword",
                    matched_keyword=keyword,
                )

        # Positive keyword check (accept if any found)
        if self._positive_patterns:
            for keyword, pattern in self._positive_patterns:
                if pattern.search(text):
                    return FilterResult(
                        passed=True,
                        reason="keyword_match",
                        matched_keyword=keyword,
                    )
            return FilterResult(passed=False, reason="no_keyword_match")

        # No keywords configured — pass everything
        return FilterResult(passed=True, reason="no_rules")


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword)
    if keyword[0].isalnum():
        escaped = rf"(?<!\w){escaped}"
    if keyword[-1].isalnum():
        escaped = rf"{escaped}(?!\w)"
    return re.compile(escaped, re.IGNORECASE)


class FilterResult:
    """Result of a filter check."""

    __slots__ = ("passed", "reason", "matched_keyword")

    def __init__(
        self,
        *,
        passed: bool,
        reason: str,
        matched_keyword: str | None = None,
    ) -> None:
        self.passed = passed
        self.reason = reason
        self.matched_keyword = matched_keyword

    def __repr__(self) -> str:
        return f"FilterResult(passed={self.passed}, reason={self.reason!r})"

    def __bool__(self) -> bool:
        return self.passed
