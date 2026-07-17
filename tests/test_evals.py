"""Tests for deterministic evaluation metrics and dataset validation."""

from pathlib import Path

import pytest

from config.watchers import load_watchers
from evals.runner import (
    EvaluationCase,
    Prediction,
    calculate_metrics,
    evaluate_cases,
    load_cases,
)
from pipeline.models import Intent


def _case(id_: str, should_alert: bool) -> EvaluationCase:
    return EvaluationCase(
        id=id_,
        watcher="test",
        language="en",
        text=f"message {id_}",
        should_alert=should_alert,
        intent=Intent.OFFER if should_alert else Intent.OTHER,
    )


def _prediction(id_: str, predicted: bool) -> Prediction:
    return Prediction(
        case_id=id_,
        predicted_alert=predicted,
        stopped_at="accepted" if predicted else "rules",
        latency_ms=1,
        reason="test",
    )


def test_calculate_metrics() -> None:
    cases = [_case("tp", True), _case("fn", True), _case("fp", False), _case("tn", False)]
    predictions = [
        _prediction("tp", True),
        _prediction("fn", False),
        _prediction("fp", True),
        _prediction("tn", False),
    ]

    metrics = calculate_metrics(
        dataset="test",
        level=1,
        cases=cases,
        predictions=predictions,
        min_precision=0.5,
        min_recall=0.5,
    )

    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.accuracy == 0.5
    assert metrics.passed_gates is True


def test_metrics_reject_mismatched_case_ids() -> None:
    with pytest.raises(ValueError, match="case IDs"):
        calculate_metrics(
            dataset="test",
            level=1,
            cases=[_case("expected", True)],
            predictions=[_prediction("different", True)],
            min_precision=0,
            min_recall=0,
        )


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="exactly one prediction"):
        calculate_metrics(
            dataset="test",
            level=1,
            cases=[_case("expected", True)],
            predictions=[],
            min_precision=0,
            min_recall=0,
        )


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    line = (
        '{"id":"duplicate","watcher":"test","language":"en","text":"message",'
        '"should_alert":true,"intent":"offer","tags":[]}\n'
    )
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(line + line)

    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(dataset, "test")


def test_load_cases_rejects_invalid_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"id": "not valid enough"}\n')

    with pytest.raises(ValueError, match=r"cases\.jsonl:1"):
        load_cases(dataset, "test")


def test_load_cases_requires_matching_watcher(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"id":"one","watcher":"other","language":"en","text":"message",'
        '"should_alert":true,"intent":"offer","tags":[]}\n'
    )

    with pytest.raises(ValueError, match="no cases"):
        load_cases(dataset, "test")


def test_committed_dataset_is_valid() -> None:
    cases = load_cases(Path("evals/data/relevance-v1.jsonl"), "phangan-housing")
    assert len(cases) >= 20
    assert any("prompt-injection" in case.tags for case in cases)
    assert {case.language for case in cases} >= {"en", "ru"}


async def test_offline_baseline_meets_regression_gate() -> None:
    watcher = load_watchers(Path("config/watchers.example.yml"))[0]
    cases = load_cases(Path("evals/data/relevance-v1.jsonl"), watcher.name)

    predictions = await evaluate_cases(
        watcher=watcher,
        cases=cases,
        level=1,
    )
    metrics = calculate_metrics(
        dataset="relevance-v1",
        level=1,
        cases=cases,
        predictions=predictions,
        min_precision=0.80,
        min_recall=0.80,
    )

    assert metrics.total == 20
    assert metrics.precision == 0.8889
    assert metrics.recall == 1.0
    assert metrics.f1 == 0.9412
    assert metrics.passed_gates is True
