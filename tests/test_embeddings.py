"""Tests for the explainable Level 2 embedding decision contract."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from config.watchers import Watcher, WatcherExamples, WatcherRules
from pipeline.embeddings import (
    EmbeddingFilter,
    ReferenceText,
    _build_reference_records,
    _build_reference_texts,
    _collection_name,
    _reference_fingerprint,
)
from pipeline.models import StageStatus


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
        prompt="Alert on Phangan housing offers.",
        examples=WatcherExamples(
            positive=[
                "Villa for rent in Srithanu, 25,000 THB per month",
                "Сдаю дом на Пангане на длительный срок",
            ],
            negative=[
                "Looking for a villa to rent",
                "What is the current weather?",
            ],
        ),
        embedding_threshold=0.72,
    )


def _embedding_response(*vectors: list[float]) -> MagicMock:
    response = MagicMock()
    response.data = [
        MagicMock(index=index, embedding=vector) for index, vector in enumerate(vectors)
    ]
    return response


def _filter_with_results(
    results: dict[str, object],
    *,
    reference_count: int = 2,
    threshold: float = 0.70,
) -> tuple[EmbeddingFilter, MagicMock, MagicMock]:
    client = MagicMock()
    client.embeddings.create = AsyncMock(return_value=_embedding_response([0.1, 0.2, 0.3]))
    collection = MagicMock()
    collection.count.return_value = reference_count
    collection.query.return_value = results
    collection.metadata = {"eidolon:threshold": threshold}
    embedding_filter = EmbeddingFilter(openai_client=client)
    embedding_filter._collections["test-watcher"] = collection
    return embedding_filter, client, collection


class TestReferenceCorpus:
    def test_preserves_positive_and_negative_examples(
        self,
        housing_watcher: Watcher,
    ) -> None:
        references = _build_reference_records(housing_watcher)

        assert {reference.label for reference in references} == {"positive", "negative"}
        assert (
            ReferenceText(
                "Villa for rent in Srithanu, 25,000 THB per month",
                "positive",
            )
            in references
        )
        assert ReferenceText("Looking for a villa to rent", "negative") in references

    def test_falls_back_to_keywords_and_objective_without_positive_examples(self) -> None:
        watcher = Watcher(
            name="fallback-watcher",
            chats=[-100456],
            rules=WatcherRules(keywords=["villa", "bungalow"]),
            llm_level=2,
            prompt="Alert on rental offers.",
        )

        assert _build_reference_texts(watcher) == [
            "villa",
            "bungalow",
            "Alert on rental offers.",
        ]

    def test_empty_watcher_has_no_references(self) -> None:
        watcher = Watcher(
            name="empty-watcher",
            chats=[-100999],
            rules=WatcherRules(),
        )

        assert _build_reference_records(watcher) == []

    def test_fingerprint_is_stable_and_includes_reference_labels(self) -> None:
        positive = [ReferenceText("Villa for rent", "positive")]
        relabeled = [ReferenceText("Villa for rent", "negative")]

        assert _reference_fingerprint(positive) == _reference_fingerprint(list(positive))
        assert _reference_fingerprint(positive) != _reference_fingerprint(relabeled)

    def test_collection_names_are_stable_safe_and_collision_resistant(self) -> None:
        first = _collection_name("Housing_Watcher_With_A_Very_Long_Name_And_Suffix_A")
        second = _collection_name("Housing_Watcher_With_A_Very_Long_Name_And_Suffix_B")

        assert first != second
        assert len(first) <= 63
        assert "_" not in first


class TestEmbeddingDecision:
    async def test_high_positive_similarity_passes_with_explanation(self) -> None:
        embedding_filter, _, collection = _filter_with_results(
            {
                "distances": [[0.15, 0.80]],
                "documents": [["Villa for rent", "Looking for a villa"]],
                "metadatas": [[{"label": "positive"}, {"label": "negative"}]],
            }
        )

        decision = await embedding_filter.check(
            "Villa available on Phangan",
            "test-watcher",
        )

        assert decision.passed is True
        assert decision.status is StageStatus.OK
        assert decision.score == 0.85
        assert decision.matched_reference == "Villa for rent"
        assert decision.error_code is None
        collection.query.assert_called_once()

    async def test_strong_negative_reference_blocks_positive_match(self) -> None:
        embedding_filter, _, _ = _filter_with_results(
            {
                "distances": [[0.20, 0.22]],
                "documents": [["Villa for rent", "Looking for a villa"]],
                "metadatas": [[{"label": "positive"}, {"label": "negative"}]],
            }
        )

        decision = await embedding_filter.check(
            "Looking for a villa",
            "test-watcher",
        )

        assert decision.passed is False
        assert decision.status is StageStatus.OK
        assert decision.score == 0.8
        assert decision.matched_reference == "Villa for rent"
        assert "negative margin" in decision.reason

    async def test_score_below_threshold_is_rejected(self) -> None:
        embedding_filter, _, _ = _filter_with_results(
            {
                "distances": [[0.40]],
                "documents": [["Villa for rent"]],
                "metadatas": [[{"label": "positive"}]],
            },
            reference_count=1,
            threshold=0.70,
        )

        decision = await embedding_filter.check("Unrelated message", "test-watcher")

        assert decision.passed is False
        assert decision.status is StageStatus.OK
        assert decision.score == 0.6

    async def test_missing_provider_or_index_is_explicitly_degraded(self) -> None:
        decision = await EmbeddingFilter().check("Any text", "unknown-watcher")

        assert decision.passed is True
        assert decision.status is StageStatus.DEGRADED
        assert decision.score is None
        assert decision.matched_reference is None
        assert decision.error_code == "provider_disabled"

    async def test_provider_error_is_fail_open_but_explicitly_degraded(self) -> None:
        embedding_filter, client, _ = _filter_with_results(
            {
                "distances": [[0.1]],
                "documents": [["Reference"]],
                "metadatas": [[{"label": "positive"}]],
            },
            reference_count=1,
        )
        client.embeddings.create = AsyncMock(side_effect=RuntimeError("provider down"))

        decision = await embedding_filter.check("Potential offer", "test-watcher")

        assert decision.passed is True
        assert decision.status is StageStatus.DEGRADED
        assert decision.error_code == "RuntimeError"
        assert decision.score is None

    async def test_empty_index_result_is_explicitly_degraded(self) -> None:
        embedding_filter, _, _ = _filter_with_results(
            {
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
            },
            reference_count=1,
        )

        decision = await embedding_filter.check("Potential offer", "test-watcher")

        assert decision.passed is True
        assert decision.status is StageStatus.DEGRADED
        assert decision.error_code == "empty_index_result"

    async def test_missing_positive_reference_is_explicitly_degraded(self) -> None:
        embedding_filter, _, _ = _filter_with_results(
            {
                "distances": [[0.10]],
                "documents": [["Looking for a villa"]],
                "metadatas": [[{"label": "negative"}]],
            },
            reference_count=1,
        )

        decision = await embedding_filter.check("Looking for a villa", "test-watcher")

        assert decision.passed is True
        assert decision.status is StageStatus.DEGRADED
        assert decision.error_code == "missing_positive_reference"

    async def test_message_is_truncated_to_2000_characters_before_embedding(self) -> None:
        embedding_filter, client, _ = _filter_with_results(
            {
                "distances": [[0.10]],
                "documents": [["Reference"]],
                "metadatas": [[{"label": "positive"}]],
            },
            reference_count=1,
        )

        await embedding_filter.check("x" * 2500, "test-watcher")

        assert len(client.embeddings.create.await_args.kwargs["input"]) == 2000


class TestIndexLifecycle:
    async def test_batch_embedding_restores_provider_order(self) -> None:
        client = MagicMock()
        response = MagicMock()
        response.data = [
            MagicMock(index=1, embedding=[2.0]),
            MagicMock(index=0, embedding=[1.0]),
        ]
        client.embeddings.create = AsyncMock(return_value=response)
        embedding_filter = EmbeddingFilter(openai_client=client)

        embeddings = await embedding_filter._embed_many(["first", "second"])

        assert embeddings == [[1.0], [2.0]]
        assert client.embeddings.create.await_args.kwargs["input"] == ["first", "second"]

    async def test_stale_fingerprint_rebuilds_and_batches_reference_embeddings(
        self,
        housing_watcher: Watcher,
    ) -> None:
        references = _build_reference_records(housing_watcher)
        client = MagicMock()
        client.embeddings.create = AsyncMock(
            return_value=_embedding_response(*[[float(index)] for index in range(len(references))])
        )

        stale_collection = MagicMock()
        stale_collection.metadata = {"eidolon:fingerprint": "stale"}
        stale_collection.count.return_value = 1
        fresh_collection = MagicMock()
        fresh_collection.metadata = {}
        fresh_collection.count.return_value = 0

        chroma = MagicMock()
        chroma.get_or_create_collection.return_value = stale_collection
        chroma.create_collection.return_value = fresh_collection
        embedding_filter = EmbeddingFilter(
            openai_client=client,
            chroma_client=chroma,
        )

        await embedding_filter._seed_collection(housing_watcher)

        collection_name = _collection_name(housing_watcher.name)
        chroma.delete_collection.assert_called_once_with(collection_name)
        chroma.create_collection.assert_called_once()
        assert client.embeddings.create.await_count == 1
        assert client.embeddings.create.await_args.kwargs["input"] == [
            reference.text for reference in references
        ]
        add_call = fresh_collection.add.call_args
        assert add_call.kwargs["metadatas"] == [
            {"label": reference.label} for reference in references
        ]
        assert embedding_filter._collections[housing_watcher.name] is fresh_collection
