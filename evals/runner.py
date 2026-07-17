"""Evaluate relevance quality against a versioned, anonymized JSONL corpus."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from pipeline.llm import (
    PROMPT_VERSION,
    LLMClassifier,
    build_watcher_objective,
    classification_passes,
)
from pipeline.models import Intent, StageStatus

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
    negative_score: float | None = None
    threshold: float | None = None
    negative_margin: float | None = None
    intent: Intent | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    models: list[str] = Field(default_factory=list)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    provider_calls: int = Field(default=0, ge=0)
    degraded_stages: list[str] = Field(default_factory=list)
    stage_errors: dict[str, str] = Field(default_factory=dict)
    reason: str


class EvaluationManifest(BaseModel):
    """Reproducibility inputs for one committed evaluation run."""

    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    watcher_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    watcher: str
    level: int
    prompt_version: str
    embedding_model: str
    llm_model: str
    embedding_threshold: float
    embedding_negative_margin: float
    target_intents: list[str]
    # Older committed artifacts predate the explicit policy and used
    # availability-first acceptance.
    degraded_policy: str = "accept"


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
    total_input_tokens: int
    total_output_tokens: int
    provider_calls: int
    degraded_predictions: int
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
    degraded_predictions = sum(bool(prediction.degraded_stages) for prediction in predictions)

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
        total_input_tokens=sum(prediction.input_tokens for prediction in predictions),
        total_output_tokens=sum(prediction.output_tokens for prediction in predictions),
        provider_calls=sum(prediction.provider_calls for prediction in predictions),
        degraded_predictions=degraded_predictions,
        passed_gates=(
            precision >= min_precision and recall >= min_recall and degraded_predictions == 0
        ),
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
            models: list[str] = []
            input_tokens = 0
            output_tokens = 0
            provider_calls = 0
            degraded_stages: list[str] = []
            stage_errors: dict[str, str] = {}
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
            negative_score: float | None = None
            threshold: float | None = None
            negative_margin: float | None = None
            if level >= 2:
                embedding = await embedding_filter.check(case.text, watcher.name)
                score = embedding.score
                negative_score = embedding.negative_score
                threshold = embedding.threshold
                negative_margin = embedding.negative_margin
                input_tokens += embedding.input_tokens
                if embedding.model:
                    models.append(embedding.model)
                if embedding.error_code != "provider_disabled":
                    provider_calls += 1
                if embedding.status is StageStatus.DEGRADED:
                    degraded_stages.append("embeddings")
                if embedding.error_code:
                    stage_errors["embeddings"] = embedding.error_code
                if embedding.status is not StageStatus.OK and watcher.degraded_policy == "reject":
                    predictions.append(
                        _prediction(
                            case=case,
                            started=started,
                            predicted=False,
                            stopped_at="embeddings",
                            score=score,
                            negative_score=negative_score,
                            threshold=threshold,
                            negative_margin=negative_margin,
                            models=models,
                            input_tokens=input_tokens,
                            provider_calls=provider_calls,
                            degraded_stages=degraded_stages,
                            stage_errors=stage_errors,
                            reason="watcher rejected degraded embedding stage",
                        )
                    )
                    continue
                if not embedding.passed:
                    predictions.append(
                        _prediction(
                            case=case,
                            started=started,
                            predicted=False,
                            stopped_at="embeddings",
                            score=score,
                            negative_score=negative_score,
                            threshold=threshold,
                            negative_margin=negative_margin,
                            models=models,
                            input_tokens=input_tokens,
                            provider_calls=provider_calls,
                            degraded_stages=degraded_stages,
                            stage_errors=stage_errors,
                            reason=embedding.reason,
                        )
                    )
                    continue

            intent: Intent | None = None
            confidence: float | None = None
            if level >= 3:
                objective = build_watcher_objective(
                    description=watcher.description,
                    prompt=watcher.prompt,
                    target_intents=watcher.target_intents,
                )
                decision = await classifier.classify(case.text, objective)
                intent = decision.result.intent
                confidence = decision.result.confidence
                input_tokens += decision.input_tokens
                output_tokens += decision.output_tokens
                if decision.model:
                    models.append(decision.model)
                if decision.error_code != "provider_disabled":
                    provider_calls += 1
                if decision.status is StageStatus.DEGRADED:
                    degraded_stages.append("llm")
                if decision.error_code:
                    stage_errors["llm"] = decision.error_code
                if not classification_passes(
                    decision,
                    watcher.target_intents,
                    watcher.degraded_policy,
                ):
                    reason = decision.result.reason
                    if (
                        decision.result.relevant
                        and decision.result.intent.value not in watcher.target_intents
                    ):
                        reason = (
                            f"intent {decision.result.intent.value!r} is outside the watcher policy"
                        )
                    predictions.append(
                        _prediction(
                            case=case,
                            started=started,
                            predicted=False,
                            stopped_at="llm",
                            score=score,
                            negative_score=negative_score,
                            threshold=threshold,
                            negative_margin=negative_margin,
                            intent=intent,
                            confidence=confidence,
                            models=models,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            provider_calls=provider_calls,
                            degraded_stages=degraded_stages,
                            stage_errors=stage_errors,
                            reason=reason,
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
                    negative_score=negative_score,
                    threshold=threshold,
                    negative_margin=negative_margin,
                    intent=intent,
                    confidence=confidence,
                    models=models,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provider_calls=provider_calls,
                    degraded_stages=degraded_stages,
                    stage_errors=stage_errors,
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
    negative_score: float | None = None,
    threshold: float | None = None,
    negative_margin: float | None = None,
    intent: Intent | None = None,
    confidence: float | None = None,
    models: list[str] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    provider_calls: int = 0,
    degraded_stages: list[str] | None = None,
    stage_errors: dict[str, str] | None = None,
) -> Prediction:
    return Prediction(
        case_id=case.id,
        predicted_alert=predicted,
        stopped_at=stopped_at,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        score=score,
        negative_score=negative_score,
        threshold=threshold,
        negative_margin=negative_margin,
        intent=intent,
        confidence=confidence,
        models=models or [],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider_calls=provider_calls,
        degraded_stages=degraded_stages or [],
        stage_errors=stage_errors or {},
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
    effective_threshold = (
        watcher.embedding_threshold
        if watcher.embedding_threshold is not None
        else settings.embedding_similarity_threshold
    )
    manifest = EvaluationManifest(
        dataset_sha256=_sha256_file(arguments.dataset),
        watcher_config_sha256=_sha256_file(arguments.watchers),
        watcher=watcher.name,
        level=arguments.level,
        prompt_version=PROMPT_VERSION,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        embedding_threshold=effective_threshold,
        embedding_negative_margin=settings.embedding_negative_margin,
        target_intents=list(watcher.target_intents),
        degraded_policy=watcher.degraded_policy,
    )

    payload = {
        "manifest": manifest.model_dump(mode="json"),
        "metrics": metrics.model_dump(mode="json"),
        "predictions": [prediction.model_dump(mode="json") for prediction in predictions],
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(metrics.model_dump(mode="json"), indent=2))
    return 0 if metrics.passed_gates else 1


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
