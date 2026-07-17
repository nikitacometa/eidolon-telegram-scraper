"""Tests for pipeline/summarizer.py — daily digest generation."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.dispatcher import _split_message
from pipeline.summarizer import DailySummarizer, _build_transcript


@pytest.fixture
def sample_messages() -> list[dict]:
    return [
        {
            "chat_id": -100123,
            "chat_title": "Phangan Expats",
            "sender_name": "Alice",
            "text": "Beautiful villa for rent, 3BR, pool, 25k THB",
            "date": "2026-03-05 10:00:00",
        },
        {
            "chat_id": -100123,
            "chat_title": "Phangan Expats",
            "sender_name": "Bob",
            "text": "Full moon party this Saturday at Haad Rin!",
            "date": "2026-03-05 12:00:00",
        },
        {
            "chat_id": -100456,
            "chat_title": "Housing KPG",
            "sender_name": "Eve",
            "text": "Сдаю бунгало на Шритану, 15000 бат/месяц",
            "date": "2026-03-05 14:00:00",
        },
    ]


@pytest.fixture
async def summarizer() -> DailySummarizer:
    s = DailySummarizer()
    s._client = AsyncMock()
    yield s
    s._client = None


def _mock_completion(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestBuildTranscript:
    def test_basic_transcript(self, sample_messages: list[dict]) -> None:
        transcript = _build_transcript(sample_messages)
        assert "[Phangan Expats] Alice:" in transcript
        assert "[Housing KPG] Eve:" in transcript
        assert "villa for rent" in transcript

    def test_empty_messages(self) -> None:
        assert _build_transcript([]) == ""

    def test_truncation(self) -> None:
        messages = [
            {"chat_title": "Chat", "sender_name": "X", "text": "a" * 5000} for _ in range(10)
        ]
        transcript = _build_transcript(messages)
        assert "truncated" in transcript
        assert len(transcript) < 15000


class TestSplitMessage:
    def test_short_message(self) -> None:
        assert _split_message("Hello", 4096) == ["Hello"]

    def test_exact_limit(self) -> None:
        text = "x" * 4096
        assert _split_message(text, 4096) == [text]

    def test_long_message_splits_at_newline(self) -> None:
        text = "Line 1\n" + "x" * 4090 + "\nLine 3"
        chunks = _split_message(text, 4096)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_no_newline_splits_at_limit(self) -> None:
        text = "x" * 5000
        chunks = _split_message(text, 4096)
        assert len(chunks) == 2
        assert len(chunks[0]) == 4096


class TestDailySummarizer:
    async def test_summarize_success(
        self,
        summarizer: DailySummarizer,
        sample_messages: list[dict],
    ) -> None:
        summarizer._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion(
                "• Villa 3BR available in Phangan, 25k THB\n• Full moon party Saturday"
            )
        )
        result = await summarizer.summarize(
            messages=sample_messages,
            watcher_name="phangan-housing",
            target_date=date(2026, 3, 5),
        )
        assert result is not None
        assert "Villa" in result or "villa" in result

    async def test_summarize_empty_messages(self, summarizer: DailySummarizer) -> None:
        result = await summarizer.summarize(
            messages=[],
            watcher_name="test",
            target_date=date(2026, 3, 5),
        )
        assert result is None

    async def test_summarize_no_client(self, sample_messages: list[dict]) -> None:
        summarizer = DailySummarizer()
        result = await summarizer.summarize(
            messages=sample_messages,
            watcher_name="test",
        )
        assert result is None

    async def test_summarize_api_error(
        self,
        summarizer: DailySummarizer,
        sample_messages: list[dict],
    ) -> None:
        summarizer._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))
        result = await summarizer.summarize(
            messages=sample_messages,
            watcher_name="test",
        )
        assert result is None

    async def test_summarize_includes_watcher_name(
        self,
        summarizer: DailySummarizer,
        sample_messages: list[dict],
    ) -> None:
        summarizer._client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("Summary text")
        )
        await summarizer.summarize(
            messages=sample_messages,
            watcher_name="phangan-housing",
            target_date=date(2026, 3, 5),
        )
        call_args = summarizer._client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "phangan-housing" in user_msg
        assert "2026-03-05" in user_msg
