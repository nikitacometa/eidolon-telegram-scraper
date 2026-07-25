"""Tests for pipeline/scoring.py — which chats earn the account's standing."""

import pytest

from pipeline.discovery import DiscoveredChat
from pipeline.recon_models import (
    CandidateState,
    ChatIdentity,
    ChatVisibility,
    DiscoverySource,
    Evidence,
)
from pipeline.scoring import (
    ScoringPolicy,
    build_policy,
    score_candidate,
)


def _chat(
    *,
    title: str = "Da Nang Housing",
    username: str = "danang_housing",
    visibility: ChatVisibility = ChatVisibility.PUBLIC,
    chat_type: str = "supergroup",
    participants: int | None = 5000,
    flags: tuple[str, ...] = (),
) -> DiscoveredChat:
    return DiscoveredChat(
        identity=ChatIdentity(username=username, title=title),
        evidence=Evidence(source=DiscoverySource.HASHTAG_SEARCH, origin_key="hashtag:danang"),
        visibility=visibility,
        title=title,
        username=username,
        chat_type=chat_type,
        participants=participants,
        flags=flags,
    )


DANANG = ScoringPolicy(
    location_keywords=("da nang", "danang", "дананг"),
    topic_keywords=("housing", "rent", "аренда"),
)


# ----------------------------------------------------------------------
# The public-scope gate
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "visibility",
    [ChatVisibility.UNKNOWN, ChatVisibility.PRIVATE],
)
def test_unproven_public_scope_is_terminal(visibility: ChatVisibility) -> None:
    """Out of scope means out of scope, however relevant the chat looks."""
    score = score_candidate(
        _chat(visibility=visibility),
        policy=DANANG,
        independent_sources=3,
    )

    assert score.decision is CandidateState.BLOCKED_PRIVATE
    assert score.value == 0.0
    assert not score.auto_joinable


def test_public_chat_passes_the_gate() -> None:
    """A public username is the proof the gate looks for."""
    score = score_candidate(_chat(), policy=DANANG, independent_sources=2)

    assert score.decision is not CandidateState.BLOCKED_PRIVATE


# ----------------------------------------------------------------------
# Risk
# ----------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["scam", "fake"])
def test_platform_risk_labels_reject_outright(flag: str) -> None:
    """Telegram's own label is an assertion, not a hint."""
    score = score_candidate(
        _chat(flags=(flag,)),
        policy=DANANG,
        independent_sources=3,
    )

    assert score.decision is CandidateState.REJECTED
    assert flag in score.risk_flags


@pytest.mark.parametrize(
    "title",
    [
        "Da Nang PUMP signals 100x",
        "Da Nang escort service",
        "Дананг казино бонус",
    ],
)
def test_junk_clusters_are_rejected_even_when_on_topic(title: str) -> None:
    """Scam chats name the city too; that is the whole point of them."""
    score = score_candidate(
        _chat(title=title),
        policy=DANANG,
        independent_sources=3,
    )

    assert score.decision is CandidateState.REJECTED
    assert "junk_topic" in score.risk_flags


# ----------------------------------------------------------------------
# Relevance
# ----------------------------------------------------------------------


def test_location_and_topic_match_scores_highest() -> None:
    """A chat naming both the city and the subject is the ideal candidate."""
    score = score_candidate(_chat(), policy=DANANG, independent_sources=3)

    assert score.value >= DANANG.auto_join_threshold
    assert score.breakdown["location"] > 0
    assert score.breakdown["topic"] > 0


def test_on_topic_elsewhere_cannot_reach_approval_on_its_own() -> None:
    """Housing in another city is not what was asked for."""
    score = score_candidate(
        _chat(title="Bali Housing and Rent", username="bali_housing"),
        policy=DANANG,
        independent_sources=3,
    )

    assert score.value < DANANG.approval_threshold
    assert score.decision is CandidateState.REJECTED


def test_bigger_chats_score_higher_but_not_without_limit() -> None:
    """Size is a weak signal and must not outweigh relevance."""
    small = score_candidate(_chat(participants=60), policy=DANANG, independent_sources=1)
    large = score_candidate(_chat(participants=200_000), policy=DANANG, independent_sources=1)

    assert large.breakdown["size"] > small.breakdown["size"]
    assert large.breakdown["size"] <= 10.0


def test_groups_are_preferred_over_broadcast_channels() -> None:
    """The owner asked for chats, where people talk back."""
    group = score_candidate(_chat(chat_type="supergroup"), policy=DANANG, independent_sources=1)
    channel = score_candidate(_chat(chat_type="channel"), policy=DANANG, independent_sources=1)

    assert group.breakdown["chat_type"] > channel.breakdown["chat_type"]


def test_score_never_exceeds_the_maximum() -> None:
    """A saturated candidate must stay comparable to the thresholds."""
    score = score_candidate(
        _chat(title="Da Nang Danang Дананг housing rent аренда", participants=500_000),
        policy=DANANG,
        independent_sources=10,
    )

    assert score.value <= 100.0


# ----------------------------------------------------------------------
# The auto-join decision
# ----------------------------------------------------------------------


def test_one_origin_cannot_earn_an_automatic_join() -> None:
    """Whoever owns a chat controls what one source says about it.

    Reposting the same message into two crawled chats is cheap; being found
    by two genuinely different routes is not. The candidate here scores above
    the automatic threshold on its own merits, so the source count is the only
    thing holding it back.
    """
    score = score_candidate(
        _chat(title="Da Nang Дананг Housing", username="danang_housing"),
        policy=DANANG,
        independent_sources=1,
    )

    assert score.value >= DANANG.auto_join_threshold
    assert score.decision is CandidateState.AWAITING_APPROVAL
    assert "1 independent source" in score.reason


def test_two_origins_earn_an_automatic_join() -> None:
    """Independent corroboration is what the threshold is really about."""
    score = score_candidate(_chat(), policy=DANANG, independent_sources=2)

    assert score.decision is CandidateState.APPROVED
    assert score.auto_joinable


def test_later_waves_always_ask_first() -> None:
    """Wave two is far from anything the owner named."""
    score = score_candidate(_chat(), policy=DANANG, independent_sources=3, wave=2)

    assert score.decision is CandidateState.AWAITING_APPROVAL
    assert "wave 2" in score.reason


def test_middling_score_asks_for_a_decision() -> None:
    """The grey band is exactly what the owner is for."""
    policy = ScoringPolicy(
        location_keywords=("da nang",),
        topic_keywords=("housing",),
        auto_join_threshold=95.0,
        approval_threshold=20.0,
    )

    score = score_candidate(
        _chat(title="Da Nang random chat", username="danang_chat", participants=100),
        policy=policy,
        independent_sources=1,
    )

    assert score.decision is CandidateState.AWAITING_APPROVAL


# ----------------------------------------------------------------------
# Policy construction
# ----------------------------------------------------------------------


def test_policy_expands_city_transliterations() -> None:
    """A Russian-speaking Da Nang chat spells the city in Cyrillic."""
    policy = build_policy(topic="housing and rent", location="Da Nang, Vietnam")

    assert "da nang" in policy.location_keywords
    assert "дананг" in policy.location_keywords
    assert "danang" in policy.location_keywords


def test_policy_keeps_meaningful_topic_words_only() -> None:
    """Short filler words match everything and mean nothing."""
    policy = build_policy(topic="housing and rent in the city", location=None)

    assert "housing" in policy.topic_keywords
    assert "and" not in policy.topic_keywords
    assert "the" not in policy.topic_keywords


def test_cyrillic_city_name_matches_a_russian_chat() -> None:
    """The transliteration list has to actually work end to end."""
    policy = build_policy(topic="жильё аренда", location="Da Nang, Vietnam")

    score = score_candidate(
        _chat(title="Дананг аренда жилья", username="danang_arenda"),
        policy=policy,
        independent_sources=2,
    )

    assert score.breakdown["location"] > 0


def test_impossible_thresholds_are_rejected() -> None:
    """A policy where approval outranks auto-join would never ask anyone."""
    with pytest.raises(ValueError):
        ScoringPolicy(auto_join_threshold=50.0, approval_threshold=80.0)
