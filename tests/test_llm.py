"""Tests for pipeline/llm.py — LLM classification filter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.llm import LLMClassifier, Verdict


@pytest.fixture
async def classifier() -> LLMClassifier:
    """Create a classifier with mocked OpenAI client."""
    c = LLMClassifier()
    c._client = AsyncMock()
    yield c
    c._client = None


def _mock_response(content: str) -> MagicMock:
    """Build a mock ChatCompletion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestVerdict:
    def test_verdict_values(self) -> None:
        assert Verdict.OFFER.value == "OFFER"
        assert Verdict.SEEK.value == "SEEK"
        assert Verdict.IRRELEVANT.value == "IRRELEVANT"


class TestLLMClassifier:
    async def test_classify_offer(self, classifier: LLMClassifier) -> None:
        """OFFER verdict should be returned for rental offers."""
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_mock_response("OFFER")
        )
        result = await classifier.classify("Beautiful villa for rent, 3BR, pool")
        assert result == Verdict.OFFER

    async def test_classify_seek(self, classifier: LLMClassifier) -> None:
        """SEEK verdict should be returned for search requests."""
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_mock_response("SEEK")
        )
        result = await classifier.classify("Looking for a house near the beach")
        assert result == Verdict.SEEK

    async def test_classify_irrelevant(self, classifier: LLMClassifier) -> None:
        """IRRELEVANT verdict for off-topic messages."""
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_mock_response("IRRELEVANT")
        )
        result = await classifier.classify("What time is sunset today?")
        assert result == Verdict.IRRELEVANT

    async def test_classify_with_watcher_prompt(self, classifier: LLMClassifier) -> None:
        """Watcher prompt should be prepended to user content."""
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_mock_response("OFFER")
        )
        await classifier.classify("Villa 2BR", watcher_prompt="Focus on Phangan rentals")

        call_args = classifier._client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "Focus on Phangan rentals" in user_msg
        assert "Villa 2BR" in user_msg

    async def test_text_truncated_to_1000(self, classifier: LLMClassifier) -> None:
        """Long text should be truncated to 1000 chars."""
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_mock_response("OFFER")
        )
        long_text = "x" * 2000
        await classifier.classify(long_text)

        call_args = classifier._client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert len(user_msg) == 1000

    async def test_timeout_returns_offer(self, classifier: LLMClassifier) -> None:
        """On timeout, fail-open: return OFFER."""
        classifier._client.chat.completions.create = AsyncMock(
            side_effect=TimeoutError("Request timed out")
        )
        result = await classifier.classify("Some message text here")
        assert result == Verdict.OFFER

    async def test_api_error_returns_offer(self, classifier: LLMClassifier) -> None:
        """On API error, fail-open: return OFFER."""
        classifier._client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("API error")
        )
        result = await classifier.classify("Some message text here")
        assert result == Verdict.OFFER

    async def test_unexpected_response_returns_offer(self, classifier: LLMClassifier) -> None:
        """Unexpected LLM output should default to OFFER."""
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_mock_response("MAYBE_RELEVANT")
        )
        result = await classifier.classify("Ambiguous message")
        assert result == Verdict.OFFER

    async def test_no_client_returns_offer(self) -> None:
        """Without API key (no client), should fail-open as OFFER."""
        classifier = LLMClassifier()
        result = await classifier.classify("Test message text")
        assert result == Verdict.OFFER

    async def test_lowercase_response_normalized(self, classifier: LLMClassifier) -> None:
        """LLM response should be normalized to uppercase."""
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_mock_response("seek")
        )
        result = await classifier.classify("I need a house")
        assert result == Verdict.SEEK
