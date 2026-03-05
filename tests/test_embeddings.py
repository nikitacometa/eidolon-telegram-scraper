"""Tests for pipeline/embeddings.py — embedding similarity filter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.watchers import Watcher, WatcherRules
from pipeline.embeddings import EmbeddingFilter, _build_reference_texts


@pytest.fixture
def housing_watcher() -> Watcher:
    return Watcher(
        name="phangan-housing",
        chats=[-100123],
        rules=WatcherRules(
            keywords=["house", "villa", "rent"],
            min_length=20,
        ),
        llm_level=3,
        prompt="Is someone offering a house for rent on Phangan?",
    )


@pytest.fixture
def empty_watcher() -> Watcher:
    return Watcher(
        name="empty",
        chats=[-100999],
        rules=WatcherRules(),
        llm_level=1,
    )


class TestBuildReferenceTexts:
    def test_generates_from_keywords(self, housing_watcher: Watcher) -> None:
        texts = _build_reference_texts(housing_watcher)
        assert len(texts) > 0
        assert any("house" in t.lower() for t in texts)
        assert any("villa" in t.lower() for t in texts)

    def test_includes_prompt(self, housing_watcher: Watcher) -> None:
        texts = _build_reference_texts(housing_watcher)
        assert any("offering a house" in t for t in texts)

    def test_includes_phangan_examples(self, housing_watcher: Watcher) -> None:
        texts = _build_reference_texts(housing_watcher)
        assert any("Srithanu" in t for t in texts)

    def test_empty_watcher_no_keywords(self, empty_watcher: Watcher) -> None:
        texts = _build_reference_texts(empty_watcher)
        assert texts == []


class TestEmbeddingFilter:
    async def test_no_client_returns_true(self) -> None:
        """Without OpenAI client, fail-open: return True."""
        f = EmbeddingFilter()
        result = await f.check("Any text here", "some-watcher")
        assert result is True

    async def test_unknown_watcher_returns_true(self) -> None:
        """Unknown watcher name should fail-open."""
        f = EmbeddingFilter()
        f._openai = AsyncMock()
        result = await f.check("Any text here", "nonexistent-watcher")
        assert result is True

    async def test_check_high_similarity_passes(self) -> None:
        """Message with high similarity should pass."""
        f = EmbeddingFilter()
        f._openai = AsyncMock()

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "distances": [[0.15]],  # cosine distance → similarity = 0.85
            "documents": [["ref text"]],
        }
        f._collections["test-watcher"] = mock_collection

        mock_emb_resp = MagicMock()
        mock_emb_resp.data = [MagicMock(embedding=[0.1] * 10)]
        f._openai.embeddings.create = AsyncMock(return_value=mock_emb_resp)

        result = await f.check("Villa for rent on Phangan", "test-watcher")
        assert result is True

    async def test_check_low_similarity_drops(self) -> None:
        """Message with low similarity should be dropped."""
        f = EmbeddingFilter()
        f._openai = AsyncMock()

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "distances": [[0.65]],  # cosine distance → similarity = 0.35
            "documents": [["ref text"]],
        }
        f._collections["test-watcher"] = mock_collection

        mock_emb_resp = MagicMock()
        mock_emb_resp.data = [MagicMock(embedding=[0.1] * 10)]
        f._openai.embeddings.create = AsyncMock(return_value=mock_emb_resp)

        result = await f.check("What time is sunset?", "test-watcher")
        assert result is False

    async def test_embed_error_failopen(self) -> None:
        """On embedding error, fail-open: return True."""
        f = EmbeddingFilter()
        f._openai = AsyncMock()
        f._collections["test-watcher"] = MagicMock()

        f._openai.embeddings.create = AsyncMock(side_effect=RuntimeError("API down"))

        result = await f.check("Some text", "test-watcher")
        assert result is True

    async def test_empty_distances_failopen(self) -> None:
        """Empty distances result should fail-open."""
        f = EmbeddingFilter()
        f._openai = AsyncMock()

        mock_collection = MagicMock()
        mock_collection.query.return_value = {"distances": [[]], "documents": [[]]}
        f._collections["test-watcher"] = mock_collection

        mock_emb_resp = MagicMock()
        mock_emb_resp.data = [MagicMock(embedding=[0.1] * 10)]
        f._openai.embeddings.create = AsyncMock(return_value=mock_emb_resp)

        result = await f.check("Some text", "test-watcher")
        assert result is True

    async def test_text_truncated_to_1000(self) -> None:
        """Long text should be truncated before embedding."""
        f = EmbeddingFilter()
        f._openai = AsyncMock()

        mock_collection = MagicMock()
        mock_collection.query.return_value = {"distances": [[0.1]], "documents": [["ref"]]}
        f._collections["test-watcher"] = mock_collection

        mock_emb_resp = MagicMock()
        mock_emb_resp.data = [MagicMock(embedding=[0.1] * 10)]
        f._openai.embeddings.create = AsyncMock(return_value=mock_emb_resp)

        long_text = "x" * 2000
        await f.check(long_text, "test-watcher")

        call_args = f._openai.embeddings.create.call_args
        assert len(call_args.kwargs["input"]) == 1000
