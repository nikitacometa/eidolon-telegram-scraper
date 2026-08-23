"""Deciding which discovered chats are worth the account's standing.

Scoring is deterministic on purpose. A language model may later explain or
re-rank a candidate, but it must not be what earns a join: the text it reads
was written by whoever owns the chat, and joining is an action taken with the
owner's real account.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from pipeline.discovery import DiscoveredChat
from pipeline.recon_models import CandidateState, ChatVisibility

# Clusters that are almost never what someone asking about a city wants, and
# that dominate unfiltered Telegram search results.
JUNK_PATTERNS: tuple[str, ...] = (
    "pump",
    "airdrop",
    "1000x",
    "100x",
    "casino",
    "betting",
    "escort",
    "massage 24",
    "forex signal",
    "binary option",
    "crypto signal",
    "nude",
    "onlyfans",
    "порно",
    "ставки",
    "казино",
    "эскорт",
)

# Telegram's own labels. These are assertions by the platform, not guesses.
PLATFORM_RISK_FLAGS: frozenset[str] = frozenset({"scam", "fake"})

MAX_SCORE = 100.0

# Weights are calibrated so that the ordinary good candidate — a group that
# names the city once, names the subject once, is a reasonable size, and was
# found by two independent routes — lands just above the automatic-join
# threshold, and anything weaker asks first. They are a starting point to be
# retuned from real discovery output, not measured constants.
LOCATION_FIRST_HIT = 35.0
LOCATION_EXTRA_HIT = 8.0
LOCATION_MAX = 45.0
TOPIC_FIRST_HIT = 18.0
TOPIC_EXTRA_HIT = 6.0
TOPIC_MAX = 30.0
SOURCE_POINTS = 5.0
SOURCE_MAX = 15.0
GROUP_POINTS = 10.0
BROADCAST_POINTS = 4.0
SIZE_MAX = 10.0


@dataclass(frozen=True, slots=True)
class ScoringPolicy:
    """What a particular reconnaissance job counts as relevant."""

    location_keywords: tuple[str, ...] = ()
    topic_keywords: tuple[str, ...] = ()
    auto_join_threshold: float = 80.0
    approval_threshold: float = 55.0
    min_independent_sources: int = 2
    max_auto_join_wave: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.approval_threshold <= self.auto_join_threshold <= MAX_SCORE:
            raise ValueError("thresholds must satisfy 0 <= approval <= auto_join <= 100")
        if self.min_independent_sources < 1:
            raise ValueError("min_independent_sources must be at least 1")


@dataclass(frozen=True, slots=True)
class Score:
    """A candidate's deterministic score and what produced it."""

    value: float
    decision: CandidateState
    breakdown: dict[str, float] = field(default_factory=dict)
    risk_flags: tuple[str, ...] = ()
    reason: str = ""

    @property
    def auto_joinable(self) -> bool:
        """Whether this candidate may be joined without asking."""
        return self.decision is CandidateState.APPROVED


def _haystack(chat: DiscoveredChat) -> str:
    parts = [chat.title or "", chat.username or ""]
    return " ".join(parts).lower()


def _keyword_hits(haystack: str, keywords: tuple[str, ...]) -> int:
    hits = 0
    for keyword in keywords:
        needle = keyword.strip().lower()
        if not needle:
            continue
        # Word-boundary matching keeps "hoi an" from matching inside a longer
        # word, while still handling non-latin scripts that \b handles poorly.
        pattern = re.escape(needle)
        if re.search(rf"(?<!\w){pattern}(?!\w)", haystack):
            hits += 1
    return hits


def score_candidate(
    chat: DiscoveredChat,
    *,
    policy: ScoringPolicy,
    independent_sources: int,
    wave: int = 0,
) -> Score:
    """Score one candidate and decide what happens to it.

    The public-scope check runs first and is terminal: a chat whose public
    access was never proven is out of scope no matter how relevant it looks,
    and no approval can bring it back.
    """
    if chat.visibility is not ChatVisibility.PUBLIC:
        return Score(
            value=0.0,
            decision=CandidateState.BLOCKED_PRIVATE,
            reason="public scope was not proven",
        )

    haystack = _haystack(chat)
    risk: list[str] = [flag for flag in chat.flags if flag in PLATFORM_RISK_FLAGS]
    junk = [pattern for pattern in JUNK_PATTERNS if pattern in haystack]
    if junk:
        risk.append("junk_topic")

    location_hits = _keyword_hits(haystack, policy.location_keywords)
    topic_hits = _keyword_hits(haystack, policy.topic_keywords)

    breakdown: dict[str, float] = {
        "location": _tiered(location_hits, LOCATION_FIRST_HIT, LOCATION_EXTRA_HIT, LOCATION_MAX),
        "topic": _tiered(topic_hits, TOPIC_FIRST_HIT, TOPIC_EXTRA_HIT, TOPIC_MAX),
        "independent_sources": min(independent_sources * SOURCE_POINTS, SOURCE_MAX),
        "chat_type": (
            GROUP_POINTS if chat.chat_type in {"supergroup", "group"} else BROADCAST_POINTS
        ),
        "size": _size_points(chat.participants),
    }
    value = min(sum(breakdown.values()), MAX_SCORE)

    if risk:
        return Score(
            value=0.0,
            decision=CandidateState.REJECTED,
            breakdown=breakdown,
            risk_flags=tuple(sorted(set(risk))),
            reason="risk flags: " + ", ".join(sorted(set(risk))),
        )

    if policy.location_keywords and location_hits == 0:
        # Being on-topic somewhere else in the world is not being relevant.
        value = min(value, policy.approval_threshold - 1)
        breakdown["location_penalty"] = -1.0

    decision = _decide(value, policy=policy, independent_sources=independent_sources, wave=wave)
    return Score(
        value=value,
        decision=decision,
        breakdown=breakdown,
        reason=_explain(decision, value, independent_sources, policy, wave),
    )


def _tiered(hits: int, first: float, extra: float, ceiling: float) -> float:
    """Score the first match generously and further ones with less weight.

    Naming the city at all is the signal; naming it three ways mostly means
    the title is long.
    """
    if hits <= 0:
        return 0.0
    return min(first + (hits - 1) * extra, ceiling)


def _size_points(participants: int | None) -> float:
    """Reward activity without letting one huge channel dominate."""
    if not participants or participants < 50:
        return 0.0
    # log10(50) ≈ 1.7 → ~3 points; log10(100000) = 5 → 10 points (capped).
    return min(math.log10(participants) * 2.0, SIZE_MAX)


def _decide(
    value: float,
    *,
    policy: ScoringPolicy,
    independent_sources: int,
    wave: int,
) -> CandidateState:
    if value >= policy.auto_join_threshold:
        # Two conditions guard automatic joining beyond the score itself: a
        # single origin can be manufactured by whoever owns the chat, and a
        # later wave is further from anything the owner actually asked for.
        if (
            independent_sources >= policy.min_independent_sources
            and wave <= policy.max_auto_join_wave
        ):
            return CandidateState.APPROVED
        return CandidateState.AWAITING_APPROVAL
    if value >= policy.approval_threshold:
        return CandidateState.AWAITING_APPROVAL
    return CandidateState.REJECTED


def _explain(
    decision: CandidateState,
    value: float,
    independent_sources: int,
    policy: ScoringPolicy,
    wave: int,
) -> str:
    if decision is CandidateState.APPROVED:
        return f"score {value:.0f} with {independent_sources} independent sources"
    if decision is CandidateState.AWAITING_APPROVAL:
        if value >= policy.auto_join_threshold:
            if independent_sources < policy.min_independent_sources:
                return f"score {value:.0f} but only {independent_sources} independent source(s)"
            return f"score {value:.0f} but wave {wave} is past the automatic limit"
        return f"score {value:.0f} needs a decision"
    return f"score {value:.0f} below the {policy.approval_threshold:.0f} threshold"


def build_policy(
    *,
    topic: str,
    location: str | None,
    auto_join_threshold: float | None = None,
    approval_threshold: float | None = None,
) -> ScoringPolicy:
    """Derive a scoring policy from what the owner asked for.

    Location keywords are expanded with common transliterations so that a
    Russian-speaking chat about Da Nang is not missed because it spells the
    city in Cyrillic.
    """
    location_keywords: list[str] = []
    if location:
        primary = location.split(",")[0].strip().lower()
        if primary:
            location_keywords.append(primary)
            location_keywords.append(primary.replace(" ", ""))
            location_keywords.extend(_TRANSLITERATIONS.get(primary, ()))

    topic_keywords = [word for word in re.split(r"[\s,]+", topic.lower()) if len(word) > 3]

    # Thresholds are only overridden when a caller asks, so the defaults live
    # in exactly one place and cannot drift apart.
    policy = ScoringPolicy(
        location_keywords=tuple(dict.fromkeys(location_keywords)),
        topic_keywords=tuple(dict.fromkeys(topic_keywords)),
    )
    if auto_join_threshold is not None:
        policy = replace(policy, auto_join_threshold=auto_join_threshold)
    if approval_threshold is not None:
        policy = replace(policy, approval_threshold=approval_threshold)
    return policy


# Cities where the owner's chats are likely to be multilingual. Kept small and
# explicit rather than pulling in a transliteration dependency for a handful
# of place names.
_TRANSLITERATIONS: dict[str, tuple[str, ...]] = {
    "da nang": ("danang", "дананг", "да нанг", "đà nẵng", "đà nang"),
    "da lat": ("dalat", "далат", "да лат", "đà lạt", "lam dong", "lâm đồng"),
    "hoi an": ("hoian", "хойан", "хой ан", "hội an"),
    "nha trang": ("nhatrang", "нячанг", "нha trang"),
    "ho chi minh": ("saigon", "сайгон", "хошимин"),
    "hanoi": ("ханой", "hà nội"),
    "phangan": ("panganm", "панган", "ко панган", "koh phangan"),
    "bali": ("бали",),
}
