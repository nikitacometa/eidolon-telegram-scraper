"""Offline quality gate for open-taxonomy entity retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from evals.place_retrieval import (
    DEFAULT_BASELINE,
    DEFAULT_DATASET,
    DEFAULT_FIXTURE,
    DEFAULT_VECTORS,
    evaluate,
    load_golden_cases,
    load_query_vectors,
    smoke,
)
from pipeline.indexer import SYSTEM_PROMPT
from storage.search import SearchDatabase


def test_golden_set_has_48_cases_and_reports_pending_evidence() -> None:
    cases = load_golden_cases(DEFAULT_DATASET)

    assert len(cases) == 48
    assert sum(case.label_status == "pending_evidence" for case in cases) == 6
    assert all(case.label_status == "pending_evidence" for case in cases if "hookah" in case.tags)


def test_offline_fixture_passes_the_retrieval_gate() -> None:
    search = SearchDatabase(DEFAULT_FIXTURE)
    search.connect(read_only=True)
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))["metrics"]
    try:
        report = evaluate(
            search=search,
            cases=load_golden_cases(DEFAULT_DATASET),
            query_vectors=load_query_vectors(DEFAULT_VECTORS),
            baseline=baseline,
        )
    finally:
        search.close()

    assert report["passed"], report["failures"]
    assert report["pending_evidence_count"] == 6


def test_offline_fixture_passes_deployment_smoke_without_provider_calls() -> None:
    search = SearchDatabase(DEFAULT_FIXTURE)
    search.connect(read_only=True)
    try:
        report = smoke(search, timeout_seconds=5.0)
    finally:
        search.close()

    assert report["passed"], report["failed_checks"]
    assert report["checks"]["known_name"]["status"] == "passed"
    assert report["checks"]["synchouse"]["status"] == "passed"
    assert report["checks"]["honest_empty"]["status"] == "passed"


def test_honest_empty_proctologist_case_returns_no_nearest_descriptor() -> None:
    search = SearchDatabase(DEFAULT_FIXTURE)
    search.connect(read_only=True)
    try:
        rows = search.search_places(
            query="проктолог в Дананге",
            city_area="Da Nang",
            expanded_fts=True,
            semantic_enabled=True,
            query_vector=[0.0] * 8,
            embedding_model="fixture-descriptors-v1",
            semantic_cutoff=0.55,
            include_contacts=False,
        )
    finally:
        search.close()

    assert rows == []


def test_marketplace_ads_are_negative_eval_cases_and_prompt_examples() -> None:
    path = Path("evals/data/entity-extraction-policy-v1.jsonl")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    marketplace = [case for case in cases if case["boundary"] == "one_off_marketplace"]

    assert len(marketplace) >= 2
    assert all(case["expected_entity_count"] == 0 for case in marketplace)
    assert "Продам iPhone 13, 128GB, пишите @seller" in SYSTEM_PROMPT
    assert "Сдам байк на три дня, телефон +84 905 123 456" in SYSTEM_PROMPT


def test_author_policy_has_measured_positive_and_negative_cases() -> None:
    path = Path("evals/data/entity-extraction-policy-v1.jsonl")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_id = {case["id"]: case for case in cases}

    positive = by_id["author-self-promo-barber"]
    assert positive["author"] == "@barber_danang (Иван)"
    assert positive["expected_entity_count"] == 1
    assert positive["expected_entity_kind"] == "person"

    negative = by_id["author-ordinary-chatter"]
    assert negative["author"] == "@weather_person (Пётр)"
    assert negative["expected_entity_count"] == 0
    assert "Сегодня отличная погода" in SYSTEM_PROMPT
