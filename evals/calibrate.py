"""Select an embedding threshold from a disjoint, labeled calibration run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from evals.runner import (
    EvaluationCase,
    EvaluationManifest,
    EvaluationMetrics,
    Prediction,
    calculate_metrics,
    load_cases,
)


class EvaluationArtifact(BaseModel):
    """Subset of an online evaluation artifact needed for calibration."""

    manifest: EvaluationManifest
    predictions: list[Prediction]


class CalibrationResult(BaseModel):
    """Best threshold and the metrics it produces on the calibration corpus."""

    threshold: float
    negative_margin: float
    evaluated_thresholds: int
    metrics: EvaluationMetrics


def calibrate_threshold(
    *,
    cases: list[EvaluationCase],
    predictions: list[Prediction],
    negative_margin: float,
    min_precision: float,
    min_recall: float,
) -> CalibrationResult:
    """Sweep every observed score boundary and prefer the safest tied gate."""
    if any(prediction.degraded_stages for prediction in predictions):
        raise ValueError("cannot calibrate from degraded provider predictions")
    by_id = {prediction.case_id: prediction for prediction in predictions}
    if set(by_id) != {case.id for case in cases}:
        raise ValueError("calibration prediction IDs do not match the dataset")

    scored = [prediction.score for prediction in predictions if prediction.stopped_at != "rules"]
    if not scored or any(score is None for score in scored):
        raise ValueError("every rule-passing calibration case needs an embedding score")
    thresholds = sorted({0.0, 1.0, *(float(score) for score in scored if score is not None)})

    candidates: list[tuple[EvaluationMetrics, float]] = []
    for threshold in thresholds:
        recalibrated = [
            _apply_threshold(
                prediction=by_id[case.id],
                threshold=threshold,
                negative_margin=negative_margin,
            )
            for case in cases
        ]
        metrics = calculate_metrics(
            dataset="calibration",
            level=2,
            cases=cases,
            predictions=recalibrated,
            min_precision=min_precision,
            min_recall=min_recall,
        )
        candidates.append((metrics, threshold))

    passing = [candidate for candidate in candidates if candidate[0].passed_gates]
    search_space = passing or candidates
    metrics, threshold = max(
        search_space,
        key=lambda candidate: (
            candidate[0].f1,
            candidate[0].precision,
            candidate[0].recall,
            candidate[1],
        ),
    )
    return CalibrationResult(
        threshold=round(threshold, 6),
        negative_margin=negative_margin,
        evaluated_thresholds=len(thresholds),
        metrics=metrics,
    )


def _apply_threshold(
    *,
    prediction: Prediction,
    threshold: float,
    negative_margin: float,
) -> Prediction:
    if prediction.stopped_at == "rules":
        predicted_alert = False
        stopped_at = "rules"
    else:
        if prediction.score is None:
            raise ValueError(f"prediction {prediction.case_id} has no embedding score")
        clears_negative = (
            prediction.negative_score is None
            or prediction.score >= prediction.negative_score + negative_margin
        )
        predicted_alert = prediction.score >= threshold and clears_negative
        stopped_at = "accepted" if predicted_alert else "embeddings"
    return prediction.model_copy(
        update={
            "predicted_alert": predicted_alert,
            "stopped_at": stopped_at,
            "threshold": threshold,
            "negative_margin": negative_margin,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/data/relevance-calibration-v1.jsonl"),
    )
    parser.add_argument("--watcher", default="phangan-housing")
    parser.add_argument("--min-precision", type=float, default=0.60)
    parser.add_argument("--min-recall", type=float, default=0.98)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    try:
        artifact = EvaluationArtifact.model_validate_json(
            arguments.artifact.read_text(encoding="utf-8")
        )
        if artifact.manifest.level != 2:
            raise ValueError("threshold calibration requires a Level 2 artifact")
        cases = load_cases(arguments.dataset, arguments.watcher)
        result = calibrate_threshold(
            cases=cases,
            predictions=artifact.predictions,
            negative_margin=artifact.manifest.embedding_negative_margin,
            min_precision=arguments.min_precision,
            min_recall=arguments.min_recall,
        )
    except (OSError, ValueError) as error:
        print(f"calibration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    rendered = json.dumps(result.model_dump(mode="json"), indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result.metrics.passed_gates else 1)


if __name__ == "__main__":
    main()
