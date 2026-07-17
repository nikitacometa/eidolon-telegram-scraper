"""Tests for deterministic evaluation metrics and dataset validation."""

import json
import sys
from pathlib import Path

import pytest

from config.watchers import load_watchers
from evals.calibrate import calibrate_threshold
from evals.calibrate import main as calibrate_main
from evals.runner import (
    EvaluationCase,
    Prediction,
    calculate_metrics,
    evaluate_cases,
    load_cases,
)
from evals.runner import (
    main as evaluation_main,
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
    assert metrics.total_input_tokens == 0
    assert metrics.total_output_tokens == 0
    assert metrics.provider_calls == 0
    assert metrics.degraded_predictions == 0
    assert metrics.passed_gates is True


def test_degraded_online_prediction_fails_quality_gate() -> None:
    case = _case("degraded", True)
    prediction = _prediction("degraded", True).model_copy(
        update={"degraded_stages": ["embeddings"]}
    )

    metrics = calculate_metrics(
        dataset="test",
        level=2,
        cases=[case],
        predictions=[prediction],
        min_precision=0,
        min_recall=0,
    )

    assert metrics.degraded_predictions == 1
    assert metrics.passed_gates is False


def test_calibration_selects_observed_score_boundary() -> None:
    cases = [
        _case("positive-high", True),
        _case("positive-low", True),
        _case("negative-close", False),
        _case("negative-low", False),
    ]
    predictions = [
        _prediction("positive-high", True).model_copy(update={"score": 0.8}),
        _prediction("positive-low", True).model_copy(update={"score": 0.6}),
        _prediction("negative-close", True).model_copy(update={"score": 0.55}),
        _prediction("negative-low", True).model_copy(update={"score": 0.2}),
    ]

    result = calibrate_threshold(
        cases=cases,
        predictions=predictions,
        negative_margin=0.05,
        min_precision=0.8,
        min_recall=0.9,
    )

    assert result.threshold == 0.6
    assert result.metrics.precision == 1.0
    assert result.metrics.recall == 1.0
    assert result.metrics.passed_gates is True


def test_calibration_rejects_degraded_or_mismatched_artifacts() -> None:
    cases = [_case("expected", True)]
    degraded = _prediction("expected", True).model_copy(
        update={"score": 0.8, "degraded_stages": ["embeddings"]}
    )
    with pytest.raises(ValueError, match="degraded"):
        calibrate_threshold(
            cases=cases,
            predictions=[degraded],
            negative_margin=0,
            min_precision=0,
            min_recall=0,
        )

    with pytest.raises(ValueError, match="IDs"):
        calibrate_threshold(
            cases=cases,
            predictions=[_prediction("different", True).model_copy(update={"score": 0.8})],
            negative_margin=0,
            min_precision=0,
            min_recall=0,
        )


def test_level1_cli_writes_reproducible_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "level1.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eidolon-eval", "--level", "1", "--output", str(output)],
    )

    with pytest.raises(SystemExit) as exit_info:
        evaluation_main()

    assert exit_info.value.code == 0
    artifact = json.loads(output.read_text())
    assert artifact["manifest"]["dataset_sha256"]
    assert artifact["manifest"]["watcher_config_sha256"]
    assert artifact["metrics"]["passed_gates"] is True


def test_calibration_cli_replays_committed_online_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "calibration.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eidolon-calibrate",
            "docs/evaluation-calibration-level2-2026-07-17.json",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        calibrate_main()

    assert exit_info.value.code == 0
    result = json.loads(output.read_text())
    assert result["metrics"]["recall"] == 1.0
    assert result["metrics"]["passed_gates"] is True


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
    assert metrics.total_input_tokens == 0
    assert metrics.provider_calls == 0
    assert metrics.passed_gates is True
