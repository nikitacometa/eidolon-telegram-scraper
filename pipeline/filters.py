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

        # Pre-compile keyword patterns for performance
        self._positive_patterns = [
            re.compile(re.escape(kw), re.IGNORECASE) for kw in self.rules.keywords
        ]
        self._negative_patterns = [
            re.compile(re.escape(kw), re.IGNORECASE) for kw in self.rules.keywords_negative
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
        for pattern in self._negative_patterns:
            if pattern.search(text):
                return FilterResult(
                    passed=False,
                    reason="negative_keyword",
                    matched_keyword=pattern.pattern,
                )

        # Positive keyword check (accept if any found)
        if self._positive_patterns:
            for pattern in self._positive_patterns:
                if pattern.search(text):
                    return FilterResult(
                        passed=True,
                        reason="keyword_match",
                        matched_keyword=pattern.pattern,
                    )
            return FilterResult(passed=False, reason="no_keyword_match")

        # No keywords configured — pass everything
        return FilterResult(passed=True, reason="no_rules")


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
