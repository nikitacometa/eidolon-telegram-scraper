"""Evaluate relevance quality against a versioned, anonymized JSONL corpus."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from config.settings import settings
from config.watchers import Watcher, load_watchers
from pipeline.embeddings import EmbeddingFilter
from pipeline.filters import RuleFilter
from pipeline.llm import LLMClassifier
from pipeline.models import Intent

DEFAULT_DATASET = Path("evals/data/relevance-v1.jsonl")


class EvaluationCase(BaseModel):
    """One human-labeled relevance example."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    watcher: str
    language: str
    text: str = Field(min_length=1)
    should_alert: bool
    intent: Intent
    tags: list[str] = Field(default_factory=list)


class Prediction(BaseModel):
    """Prediction and stage provenance for one case."""

    case_id: str
    predicted_alert: bool
    stopped_at: Literal["rules", "embeddings", "llm", "accepted"]
    latency_ms: float = Field(ge=0)
    score: float | None = None
    reason: str


class EvaluationMetrics(BaseModel):
    """Binary relevance metrics used as CI regression gates."""

    dataset: str
    level: int
    total: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    mean_latency_ms: float
    passed_gates: bool


def load_cases(path: Path, watcher_name: str) -> list[EvaluationCase]:
    """Load and validate all cases for one watcher."""
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                case = EvaluationCase.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if case.id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate case id {case.id}")
            seen_ids.add(case.id)
            if case.watcher == watcher_name:
                cases.append(case)
    if not cases:
        raise ValueError(f"no cases for watcher {watcher_name!r} in {path}")
    return cases


def calculate_metrics(
    *,
    dataset: str,
    level: int,
    cases: list[EvaluationCase],
    predictions: list[Prediction],
    min_precision: float,
    min_recall: float,
) -> EvaluationMetrics:
    """Calculate deterministic binary classification metrics."""
    if len(cases) != len(predictions):
        raise ValueError("each evaluation case must have exactly one prediction")

    expected = {case.id: case.should_alert for case in cases}
    actual = {prediction.case_id: prediction.predicted_alert for prediction in predictions}
    if expected.keys() != actual.keys():
        raise ValueError("prediction case IDs do not match the dataset")

    true_positive = sum(actual[id_] and label for id_, label in expected.items())
    false_positive = sum(actual[id_] and not label for id_, label in expected.items())
    true_negative = sum(not actual[id_] and not label for id_, label in expected.items())
    false_negative = sum(not actual[id_] and label for id_, label in expected.items())

    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    accuracy = _safe_ratio(true_positive + true_negative, len(cases))
    mean_latency = _safe_ratio(
        sum(prediction.latency_ms for prediction in predictions),
        len(predictions),
    )

    return EvaluationMetrics(
        dataset=dataset,
        level=level,
        total=len(cases),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        mean_latency_ms=round(mean_latency, 3),
        passed_gates=precision >= min_precision and recall >= min_recall,
    )


async def evaluate_cases(
    *,
    watcher: Watcher,
    cases: list[EvaluationCase],
    level: int,
) -> list[Prediction]:
    """Replay cases through the configured pipeline stages."""
    rule_filter = RuleFilter(watcher)
    embedding_filter = EmbeddingFilter()
    classifier = LLMClassifier()

    if level >= 2:
        await embedding_filter.start([watcher])
    if level >= 3:
        await classifier.start()

    predictions: list[Prediction] = []
    try:
        for case in cases:
            started = time.perf_counter()
            rule_result = rule_filter.check(case.text)
            if not rule_result:
                predictions.append(
                    _prediction(
                        case=case,
                        started=started,
                        predicted=False,
                        stopped_at="rules",
                        reason=rule_result.reason,
                    )
                )
                continue

            score: float | None = None
            if level >= 2:
                embedding = await embedding_filter.check(case.text, watcher.name)
                score = embedding.score
                if not embedding.passed:
                    predictions.append(
                        _prediction(
                            case=case,
                            started=started,
                            predicted=False,
                            stopped_at="embeddings",
                            score=score,
                            reason=embedding.reason,
                        )
                    )
                    continue

            if level >= 3:
                objective = "\n\n".join(
                    part for part in (watcher.description.strip(), watcher.prompt.strip()) if part
                )
                decision = await classifier.classify(case.text, objective)
                if not decision.result.relevant:
                    predictions.append(
                        _prediction(
                            case=case,
                            started=started,
                            predicted=False,
                            stopped_at="llm",
                            score=score,
                            reason=decision.result.reason,
                        )
                    )
                    continue

            predictions.append(
                _prediction(
                    case=case,
                    started=started,
                    predicted=True,
                    stopped_at="accepted",
                    score=score,
                    reason="all configured stages passed",
                )
            )
    finally:
        await classifier.close()
        await embedding_filter.close()

    return predictions


def _prediction(
    *,
    case: EvaluationCase,
    started: float,
    predicted: bool,
    stopped_at: Literal["rules", "embeddings", "llm", "accepted"],
    reason: str,
    score: float | None = None,
) -> Prediction:
    return Prediction(
        case_id=case.id,
        predicted_alert=predicted,
        stopped_at=stopped_at,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        score=score,
        reason=reason,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _find_watcher(path: Path, name: str) -> Watcher:
    for watcher in load_watchers(path):
        if watcher.name == name:
            return watcher
    raise ValueError(f"watcher {name!r} not found in {path}")


async def _run(arguments: argparse.Namespace) -> int:
    if arguments.level >= 2 and not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for level 2 or 3 evaluation")

    watcher = _find_watcher(arguments.watchers, arguments.watcher)
    cases = load_cases(arguments.dataset, arguments.watcher)
    predictions = await evaluate_cases(
        watcher=watcher,
        cases=cases,
        level=arguments.level,
    )
    metrics = calculate_metrics(
        dataset=str(arguments.dataset),
        level=arguments.level,
        cases=cases,
        predictions=predictions,
        min_precision=arguments.min_precision,
        min_recall=arguments.min_recall,
    )

    payload = {
        "metrics": metrics.model_dump(mode="json"),
        "predictions": [prediction.model_dump(mode="json") for prediction in predictions],
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(metrics.model_dump(mode="json"), indent=2))
    return 0 if metrics.passed_gates else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--watchers", type=Path, default=Path("config/watchers.example.yml"))
    parser.add_argument("--watcher", default="phangan-housing")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    try:
        exit_code = asyncio.run(_run(arguments))
    except (OSError, ValueError) as error:
        print(f"evaluation error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
