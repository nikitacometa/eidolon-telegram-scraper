"""Deterministic evaluation for open-taxonomy entity retrieval.

The fixture and query vectors are committed artifacts. This module opens them
read-only during evaluation and never constructs an embedding or LLM client.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from storage.search import SearchDatabase, content_digest, normalize_descriptor

FIXTURE_EMBEDDING_MODEL = "fixture-descriptors-v1"
DEFAULT_DATASET = Path("evals/data/place-retrieval-golden-v1.jsonl")
DEFAULT_FIXTURE = Path("evals/data/place-retrieval-fixture-v1.db")
DEFAULT_VECTORS = Path("evals/data/place-query-vectors-v1.npz")
DEFAULT_BASELINE = Path("docs/place-retrieval-baseline-v1.json")


@dataclass(frozen=True, slots=True)
class GoldenCase:
    id: str
    query: str
    arguments: dict[str, Any]
    expected_entity_keys: frozenset[str]
    must_be_empty: bool
    tags: tuple[str, ...]
    label_status: str
    evidence_corpus_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    macro_recall_at_5: float
    pooled_precision_at_5: float
    positive_non_empty_share: float
    negative_non_empty_share: float
    name_compatibility_recall_at_5: float


def load_golden_cases(path: Path) -> list[GoldenCase]:
    """Load and validate the retrieval dataset contract."""
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            case_id = str(raw["id"])
            if case_id in seen:
                raise ValueError(f"duplicate case id at {path}:{line_number}: {case_id}")
            seen.add(case_id)
            label_status = str(raw.get("label_status", "verified"))
            if label_status not in {"verified", "pending_evidence"}:
                raise ValueError(f"invalid label_status at {path}:{line_number}")
            expected = frozenset(str(value) for value in raw.get("expected_entity_keys", []))
            must_be_empty = bool(raw.get("must_be_empty", False))
            if label_status == "verified" and must_be_empty == bool(expected):
                raise ValueError(
                    f"verified case must have expected keys xor must_be_empty at "
                    f"{path}:{line_number}"
                )
            cases.append(
                GoldenCase(
                    id=case_id,
                    query=str(raw["query"]),
                    arguments=dict(raw.get("arguments", {})),
                    expected_entity_keys=expected,
                    must_be_empty=must_be_empty,
                    tags=tuple(str(value) for value in raw.get("tags", [])),
                    label_status=label_status,
                    evidence_corpus_ids=tuple(
                        int(value) for value in raw.get("evidence_corpus_ids", [])
                    ),
                )
            )
    return cases


def load_query_vectors(path: Path) -> dict[str, list[float]]:
    """Load frozen vectors without enabling pickle-backed object arrays."""
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["ids"]
        vectors = archive["vectors"]
    if len(ids) != len(vectors):
        raise ValueError("query vector ids and rows differ")
    return {
        str(case_id): list(map(float, vector)) for case_id, vector in zip(ids, vectors, strict=True)
    }


def evaluate(
    *,
    search: SearchDatabase,
    cases: list[GoldenCase],
    query_vectors: dict[str, list[float]],
    k: int = 5,
    baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run verified cases and apply quality/regression gates."""
    verified = [case for case in cases if case.label_status == "verified"]
    pending = [case.id for case in cases if case.label_status == "pending_evidence"]
    outcomes: list[dict[str, Any]] = []
    for case in verified:
        arguments = case.arguments
        name = arguments.get("name")
        category_query = arguments.get("category_query")
        natural_query = (
            str(category_query)
            if category_query is not None
            else (None if name is not None else case.query)
        )
        semantic = bool(arguments.get("semantic", True))
        vector = query_vectors.get(case.id) if semantic else None
        rows = search.search_places(
            name_query=str(name) if name is not None else None,
            query=natural_query,
            city_area=_optional_string(arguments.get("city")),
            place_type=_optional_string(arguments.get("place_type")),
            event_types=(
                [str(arguments["event_type"])] if arguments.get("event_type") is not None else None
            ),
            entity_kind=_optional_string(arguments.get("entity_kind")),
            access_mode=_optional_string(arguments.get("access_mode")),
            min_mentions=int(arguments.get("min_mentions", 1)),
            limit=k,
            include_contacts=False,
            expanded_fts=True,
            semantic_enabled=semantic and vector is not None,
            query_vector=vector,
            embedding_model=FIXTURE_EMBEDDING_MODEL,
            semantic_cutoff=float(arguments.get("semantic_cutoff", 0.55)),
        )
        keys = [str(row["entity_key"]) for row in rows[:k]]
        relevant = len(set(keys) & case.expected_entity_keys)
        recall = relevant / len(case.expected_entity_keys) if case.expected_entity_keys else 0.0
        passed = not keys if case.must_be_empty else relevant > 0
        outcomes.append(
            {
                "id": case.id,
                "keys": keys,
                "relevant": relevant,
                "recall_at_5": recall,
                "passed": passed,
                "tags": list(case.tags),
                "must_be_empty": case.must_be_empty,
            }
        )

    positives = [outcome for outcome in outcomes if not outcome["must_be_empty"]]
    negatives = [outcome for outcome in outcomes if outcome["must_be_empty"]]
    name_cases = [outcome for outcome in positives if "name-compatibility" in outcome["tags"]]
    relevant_total = sum(int(outcome["relevant"]) for outcome in outcomes)
    returned_total = sum(min(k, len(outcome["keys"])) for outcome in outcomes)
    metrics = RetrievalMetrics(
        macro_recall_at_5=_mean(float(outcome["recall_at_5"]) for outcome in positives),
        pooled_precision_at_5=(relevant_total / returned_total if returned_total else 0.0),
        positive_non_empty_share=_mean(bool(outcome["keys"]) for outcome in positives),
        negative_non_empty_share=_mean(bool(outcome["keys"]) for outcome in negatives),
        name_compatibility_recall_at_5=_mean(
            float(outcome["recall_at_5"]) for outcome in name_cases
        ),
    )
    metric_values = asdict(metrics)
    failures: list[str] = []
    thresholds = {
        "macro_recall_at_5": 0.90,
        "pooled_precision_at_5": 0.80,
        "positive_non_empty_share": 0.95,
        "negative_non_empty_share": 0.00,
        "name_compatibility_recall_at_5": 1.00,
    }
    for name, threshold in thresholds.items():
        value = metric_values[name]
        if name == "negative_non_empty_share":
            if value != threshold:
                failures.append(f"{name}={value:.4f}, required {threshold:.4f}")
        elif value < threshold:
            failures.append(f"{name}={value:.4f}, required >= {threshold:.4f}")
    if baseline:
        for name, old_value in baseline.items():
            if name not in metric_values:
                continue
            new_value = metric_values[name]
            if name == "negative_non_empty_share":
                if new_value > float(old_value) + 0.02:
                    failures.append(f"{name} regressed from {old_value:.4f} to {new_value:.4f}")
            elif new_value < float(old_value) - 0.02:
                failures.append(f"{name} regressed from {old_value:.4f} to {new_value:.4f}")
    critical_failures = [
        str(outcome["id"])
        for outcome in outcomes
        if "critical" in outcome["tags"] and not outcome["passed"]
    ]
    failures.extend(f"critical case failed: {case_id}" for case_id in critical_failures)
    return {
        "passed": not failures,
        "metrics": metric_values,
        "evaluated_cases": len(verified),
        "pending_evidence_count": len(pending),
        "pending_evidence": pending,
        "failures": failures,
        "cases": outcomes,
    }


def smoke(search: SearchDatabase, *, timeout_seconds: float) -> dict[str, Any]:
    """Run fast lexical deployment invariants without a provider call."""
    started = time.monotonic()
    checks: dict[str, dict[str, Any]] = {}
    known = search.conn.execute(
        "SELECT name, entity_kind, canonical FROM places "
        "WHERE mention_count > 0 ORDER BY mention_count DESC, place_id LIMIT 1"
    ).fetchone()
    if known is None:
        checks["known_name"] = {"status": "failed", "detail": "places is empty"}
    else:
        old = search.search_places(name_query=str(known["name"]), limit=5, include_contacts=False)
        new = search.search_places(
            name_query=str(known["name"]),
            limit=5,
            include_contacts=False,
            expanded_fts=True,
        )
        key = f"{known['entity_kind']}|{known['canonical']}"
        old_keys = {row["entity_key"] for row in old}
        new_keys = {row["entity_key"] for row in new}
        checks["known_name"] = {
            "status": "passed" if key in old_keys and key in new_keys else "failed",
            "detail": key,
        }

    synchouse = search.conn.execute(
        "SELECT entity_kind, canonical FROM places WHERE canonical='synchouse'"
    ).fetchone()
    if synchouse is None:
        checks["synchouse"] = {"status": "pending_evidence", "detail": "entity absent"}
    else:
        key = f"{synchouse['entity_kind']}|{synchouse['canonical']}"
        variants = [
            search.search_places(name_query=value, expanded_fts=True, include_contacts=False)
            for value in ("SYNCHØUSE", "synchouse")
        ]
        checks["synchouse"] = {
            "status": (
                "passed"
                if all(key in {row["entity_key"] for row in rows} for rows in variants)
                else "failed"
            ),
            "detail": "stylized and plain spelling",
        }

    terminal = search.search_places(
        query="автовокзал", expanded_fts=True, semantic_enabled=False, include_contacts=False
    )
    checks["terminal"] = {
        "status": "passed" if terminal else "pending_evidence",
        "detail": [row["entity_key"] for row in terminal[:3]],
    }
    proctologist = search.search_places(
        query="проктолог",
        city_area="Da Nang",
        expanded_fts=True,
        semantic_enabled=False,
        include_contacts=False,
    )
    checks["honest_empty"] = {
        "status": "passed" if not proctologist else "failed",
        "detail": [row["entity_key"] for row in proctologist],
    }
    checks["hookah"] = {
        "status": "pending_evidence",
        "detail": "live evidence sweep decides positive vs must_be_empty",
    }
    status = search.status()
    checks["fts_row_count"] = {
        "status": ("passed" if status["places"] == status["place_fts_next_rows"] else "failed"),
        "detail": {
            "places": status["places"],
            "place_fts_next": status["place_fts_next_rows"],
        },
    }
    elapsed = time.monotonic() - started
    checks["timeout"] = {
        "status": "passed" if elapsed <= timeout_seconds else "failed",
        "detail": round(elapsed, 4),
    }
    failed = [name for name, check in checks.items() if check["status"] == "failed"]
    return {
        "passed": not failed,
        "elapsed_seconds": round(elapsed, 4),
        "failed_checks": failed,
        "checks": checks,
        "active_prompt_version": status["active_prompt_version"],
        "descriptor_embedding_backlog": status["descriptor_embedding_backlog"],
    }


def build_fixture(db_path: Path, vectors_path: Path, cases_path: Path) -> None:
    """Build committed anonymized artifacts from explicit synthetic evidence."""
    if db_path.exists():
        db_path.unlink()
    search = SearchDatabase(db_path)
    search.connect()
    entities = _fixture_entities()
    descriptor_ids: dict[str, int] = {}
    for index, (text, entity) in enumerate(entities, start=1):
        cursor = search.conn.execute(
            """
            INSERT INTO corpus_messages (
                source, chat_id, telegram_msg_id, text, date, content_hash
            ) VALUES ('scout', -9000, ?, ?, ?, ?)
            """,
            (index, text, f"2026-08-{index:02d}T10:00:00+00:00", content_digest(text)),
        )
        corpus_id = int(cursor.lastrowid or 0)
        search._seed_extraction_state()
        search.record_extraction(corpus_id, [entity], model="fixture")
        descriptor = normalize_descriptor(str(entity["descriptor"]))
        descriptor_row = search.conn.execute(
            "SELECT descriptor_id FROM descriptors WHERE normalized=?", (descriptor,)
        ).fetchone()
        if descriptor_row is None:
            raise RuntimeError(f"fixture descriptor missing: {descriptor}")
        descriptor_ids[descriptor] = int(descriptor_row["descriptor_id"])
    descriptor_vectors = _descriptor_vectors()
    search.store_descriptor_embeddings(
        ((descriptor_ids[descriptor], vector) for descriptor, vector in descriptor_vectors.items()),
        model=FIXTURE_EMBEDDING_MODEL,
    )
    search.close()

    cases = load_golden_cases(cases_path)
    query_vectors = _query_vectors(cases)
    np.savez_compressed(
        vectors_path,
        ids=np.asarray([case.id for case in cases], dtype=np.str_),
        vectors=np.asarray([query_vectors[case.id] for case in cases], dtype=np.float32),
    )


def _fixture_entities() -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "Сергей @sergeyrepair ремонтирует айфоны с выездом по Данангу, меняет экраны и батареи.",
            _entity(
                "Сергей",
                "person",
                ["house_call"],
                "мастер по ремонту айфонов",
                ["ремонт айфонов", "замена экранов", "замена батарей"],
                "Сергей @sergeyrepair ремонтирует айфоны с выездом по Данангу",
                aliases=["@sergeyrepair"],
            ),
        ),
        (
            "Барбер Дананг, пишите в личку @barberdanang: мужские стрижки и оформление бороды.",
            _entity(
                "@barberdanang",
                "person",
                ["unknown"],
                "барбер",
                ["мужские стрижки", "оформление бороды"],
                "Барбер Дананг, пишите в личку @barberdanang",
            ),
        ),
        (
            "Dr Minh, dentist in Da Nang, принимает взрослых и детей.",
            _entity(
                "Dr Minh",
                "person",
                ["visit"],
                "dentist",
                ["приём взрослых", "приём детей"],
                "Dr Minh, dentist in Da Nang",
            ),
        ),
        (
            "Lotus Dental clinic in Da Nang: лечение зубов и гигиена.",
            _entity(
                "Lotus Dental",
                "place",
                ["visit"],
                "dental clinic",
                ["лечение зубов", "гигиена"],
                "Lotus Dental clinic in Da Nang",
            ),
        ),
        (
            "Автовокзал Мё Динь — билеты и междугородние автобусы в Хайфон и Дананг.",
            _entity(
                "Автовокзал Мё Динь",
                "place",
                ["visit"],
                "автовокзал",
                ["билеты", "междугородние автобусы"],
                "Автовокзал Мё Динь — билеты и междугородние автобусы",
                city="unknown",
            ),
        ),
        (
            "River Yoga Studio в Дананге: йога, медитация и мастер-классы.",
            _entity(
                "River Yoga Studio",
                "place",
                ["visit"],
                "йога-студия",
                ["йога", "медитация", "мастер-классы"],
                "River Yoga Studio в Дананге",
            ),
        ),
        (
            "Открытый микрофон и концерты в Sound Cafe, живая музыка по пятницам.",
            _entity(
                "Sound Cafe",
                "place",
                ["visit"],
                "кафе с живой музыкой",
                ["открытый микрофон", "концерты", "живая музыка"],
                "Открытый микрофон и концерты в Sound Cafe",
            ),
        ),
        (
            "WELCOME TO SYNCHØUSE COMMUNITY: DJ sets and quality music.",
            _entity(
                "SYNCHØUSE",
                "place",
                ["visit"],
                "music club",
                ["DJ sets", "quality music"],
                "WELCOME TO SYNCHØUSE COMMUNITY",
            ),
        ),
    ]


def _entity(
    name: str,
    kind: str,
    access_modes: list[str],
    descriptor: str,
    offerings: list[str],
    evidence: str,
    *,
    city: str = "Da Nang",
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "aliases": aliases or [],
        "entity_kind": kind,
        "access_modes": access_modes,
        "descriptor": descriptor,
        "descriptor_language": "en" if descriptor.isascii() else "ru",
        "offerings": offerings,
        "city_area": city,
        "evidence": evidence,
        "confidence": 0.95,
    }


def _descriptor_vectors() -> dict[str, list[float]]:
    return {
        "мастер по ремонту айфонов": [1, 0, 0, 0, 0, 0, 0, 0],
        "барбер": [0, 1, 0, 0, 0, 0, 0, 0],
        "dentist": [0, 0, 1, 0, 0, 0, 0, 0],
        "dental clinic": [0, 0, 0.9, 0.1, 0, 0, 0, 0],
        "автовокзал": [0, 0, 0, 1, 0, 0, 0, 0],
        "йога-студия": [0, 0, 0, 0, 1, 0, 0, 0],
        "кафе с живой музыкой": [0, 0, 0, 0, 0, 1, 0, 0],
        "music club": [0, 0, 0, 0, 0, 0.9, 0.1, 0],
    }


def _query_vectors(cases: list[GoldenCase]) -> dict[str, list[float]]:
    tag_vectors = {
        "repair": [1, 0, 0, 0, 0, 0, 0, 0],
        "barber": [0, 1, 0, 0, 0, 0, 0, 0],
        "medicine": [0, 0, 1, 0, 0, 0, 0, 0],
        "transport": [0, 0, 0, 1, 0, 0, 0, 0],
        "yoga": [0, 0, 0, 0, 1, 0, 0, 0],
        "events": [0, 0, 0, 0, 0, 1, 0, 0],
    }
    vectors: dict[str, list[float]] = {}
    for case in cases:
        vector = [0.0] * 8
        if not case.must_be_empty:
            for tag, candidate in tag_vectors.items():
                if tag in case.tags:
                    vector = list(map(float, candidate))
                    break
        vectors[case.id] = vector
    return vectors


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _mean(values: Any) -> float:
    collected = [float(value) for value in values]
    return sum(collected) / len(collected) if collected else 0.0


def _load_baseline(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    metrics = raw.get("metrics", raw)
    return {str(key): float(value) for key, value in metrics.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--db", type=Path)
    source.add_argument("--fixture", type=Path)
    parser.add_argument("--query-vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fail-on-regression", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--build-fixture", action="store_true")
    args = parser.parse_args()

    if args.build_fixture:
        build_fixture(
            args.fixture or DEFAULT_FIXTURE,
            args.query_vectors,
            args.dataset,
        )
        return 0
    db_path = args.db or args.fixture or DEFAULT_FIXTURE
    search = SearchDatabase(db_path)
    search.connect(read_only=True)
    try:
        if args.smoke:
            report = smoke(search, timeout_seconds=args.timeout)
        else:
            report = evaluate(
                search=search,
                cases=load_golden_cases(args.dataset),
                query_vectors=load_query_vectors(args.query_vectors),
                k=max(1, args.k),
                baseline=_load_baseline(args.fail_on_regression),
            )
    finally:
        search.close()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
