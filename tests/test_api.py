"""Tests for the read-only FastAPI control plane."""

import asyncio
import json
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from api import ReadOnlySQLite, create_app
from storage.db import Database


def _write_watchers(path: Path) -> None:
    path.write_text(
        """
watchers:
  - name: housing-watch
    description: Long-term island housing offers
    chats: [-100123456789]
    rules:
      keywords: [villa, house, rent]
      keywords_negative: [looking for, need]
      min_length: 10
    alert: immediate
    llm_level: 3
    prompt: Alert only when somebody offers a property for rent.
    examples:
      positive: [Villa for rent for six months]
      negative: [Looking for a villa]
""".strip(),
        encoding="utf-8",
    )


async def _seed_stats(db_path: Path, selected_date: date) -> None:
    database = Database(db_path)
    await database.connect()
    try:
        await database.conn.execute(
            """
            INSERT INTO filter_stats (
                watcher_name,
                date,
                messages_total,
                passed_level1,
                passed_level2,
                passed_level3,
                alerts_sent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("housing-watch", selected_date.isoformat(), 20, 12, 8, 5, 4),
        )
        await database.conn.execute(
            """
            INSERT INTO filter_stats (
                watcher_name,
                date,
                messages_total,
                passed_level1,
                passed_level2,
                passed_level3,
                alerts_sent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("archived-watch", selected_date.isoformat(), 3, 2, 0, 0, 1),
        )
        await database.conn.commit()
    finally:
        await database.close()


async def _initialize_database(db_path: Path) -> None:
    database = Database(db_path)
    await database.connect()
    await database.close()


def test_liveness_has_no_database_or_config_dependency(tmp_path: Path) -> None:
    db_path = tmp_path / "control.db"
    app = create_app(
        db_path=db_path,
        watchers_path=tmp_path / "missing.yml",
    )

    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "eidolon-control-plane",
    }
    assert not db_path.exists()


def test_readiness_reports_missing_config_without_leaking_paths(tmp_path: Path) -> None:
    db_path = tmp_path / "control.db"
    asyncio.run(_initialize_database(db_path))
    missing_path = tmp_path / "private" / "watchers.yml"
    app = create_app(
        db_path=db_path,
        watchers_path=missing_path,
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body == {
        "status": "not_ready",
        "components": {
            "database": {"ready": True, "code": None, "item_count": None},
            "configuration": {
                "ready": False,
                "code": "watchers_config_missing",
                "item_count": 0,
            },
        },
    }
    assert str(missing_path) not in json.dumps(body)


def test_readiness_reports_database_failure_without_exception(tmp_path: Path) -> None:
    watchers_path = tmp_path / "watchers.yml"
    _write_watchers(watchers_path)
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("occupied", encoding="utf-8")
    app = create_app(
        db_path=not_a_directory / "control.db",
        watchers_path=watchers_path,
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")
        stats_response = client.get("/stats")

    assert response.status_code == 503
    body = response.json()
    assert body["components"]["database"] == {
        "ready": False,
        "code": "database_start_failed",
        "item_count": None,
    }
    assert body["components"]["configuration"] == {
        "ready": True,
        "code": None,
        "item_count": 1,
    }
    assert stats_response.status_code == 503
    assert stats_response.json()["detail"] == {"code": "database_start_failed"}


def test_ready_with_valid_database_and_configuration(tmp_path: Path) -> None:
    watchers_path = tmp_path / "watchers.yml"
    _write_watchers(watchers_path)
    db_path = tmp_path / "control.db"
    asyncio.run(_initialize_database(db_path))
    app = create_app(
        database=ReadOnlySQLite(db_path),
        watchers_path=watchers_path,
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["components"]["configuration"]["item_count"] == 1


def test_stats_returns_daily_rows_totals_and_provenance(tmp_path: Path) -> None:
    selected_date = date(2026, 7, 17)
    db_path = tmp_path / "control.db"
    asyncio.run(_seed_stats(db_path, selected_date))
    watchers_path = tmp_path / "watchers.yml"
    _write_watchers(watchers_path)
    app = create_app(db_path=db_path, watchers_path=watchers_path)

    with TestClient(app) as client:
        response = client.get("/stats", params={"date": selected_date.isoformat()})

    assert response.status_code == 200
    assert response.json() == {
        "date": "2026-07-17",
        "watchers": [
            {
                "watcher_name": "archived-watch",
                "messages_total": 3,
                "passed_level1": 2,
                "passed_level2": 0,
                "passed_level3": 0,
                "alerts_sent": 1,
            },
            {
                "watcher_name": "housing-watch",
                "messages_total": 20,
                "passed_level1": 12,
                "passed_level2": 8,
                "passed_level3": 5,
                "alerts_sent": 4,
            },
        ],
        "totals": {
            "messages_total": 23,
            "passed_level1": 14,
            "passed_level2": 8,
            "passed_level3": 5,
            "alerts_sent": 5,
        },
        "provenance": {
            "source": "sqlite.filter_stats",
            "timezone": "UTC",
        },
    }


def test_stats_includes_configured_watcher_with_zero_activity(tmp_path: Path) -> None:
    watchers_path = tmp_path / "watchers.yml"
    _write_watchers(watchers_path)
    db_path = tmp_path / "control.db"
    asyncio.run(_initialize_database(db_path))
    app = create_app(
        db_path=db_path,
        watchers_path=watchers_path,
    )

    with TestClient(app) as client:
        response = client.get("/stats", params={"date": "2026-07-17"})

    assert response.status_code == 200
    assert response.json()["watchers"] == [
        {
            "watcher_name": "housing-watch",
            "messages_total": 0,
            "passed_level1": 0,
            "passed_level2": 0,
            "passed_level3": 0,
            "alerts_sent": 0,
        }
    ]


def test_stats_rejects_invalid_date(tmp_path: Path) -> None:
    watchers_path = tmp_path / "watchers.yml"
    _write_watchers(watchers_path)
    db_path = tmp_path / "control.db"
    asyncio.run(_initialize_database(db_path))
    app = create_app(
        db_path=db_path,
        watchers_path=watchers_path,
    )

    with TestClient(app) as client:
        response = client.get("/stats", params={"date": "not-a-date"})

    assert response.status_code == 422


def test_analyze_returns_explainable_rule_only_provenance(tmp_path: Path) -> None:
    watchers_path = tmp_path / "watchers.yml"
    _write_watchers(watchers_path)
    app = create_app(
        db_path=tmp_path / "control.db",
        watchers_path=watchers_path,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze",
            json={
                "watcher_name": "housing-watch",
                "text": "Beautiful villa for rent near the beach",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["watcher_name"] == "housing-watch"
    assert body["accepted"] is True
    assert body["stages"] == [
        {
            "stage": "rules",
            "status": "ok",
            "passed": True,
            "reason": "keyword_match",
            "matched_keyword": "villa",
        },
        {
            "stage": "embeddings",
            "status": "skipped",
            "passed": None,
            "reason": "control_plane_rule_only",
            "matched_keyword": None,
        },
        {
            "stage": "llm",
            "status": "skipped",
            "passed": None,
            "reason": "control_plane_rule_only",
            "matched_keyword": None,
        },
    ]
    assert body["provenance"]["execution_mode"] == "rule_only"
    assert body["provenance"]["deterministic"] is True
    assert body["provenance"]["external_calls"] == 0
    assert body["provenance"]["persisted"] is False
    assert len(body["provenance"]["watcher_config_fingerprint"]) == 16


def test_analyze_explains_rejection_and_never_runs_ai(tmp_path: Path) -> None:
    watchers_path = tmp_path / "watchers.yml"
    _write_watchers(watchers_path)
    app = create_app(
        db_path=tmp_path / "control.db",
        watchers_path=watchers_path,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze",
            json={
                "watcher_name": "housing-watch",
                "text": "Looking for a villa to rent near the beach",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is False
    assert body["stages"][0]["reason"] == "negative_keyword"
    assert body["stages"][0]["matched_keyword"] == "looking for"
    assert all(stage["status"] == "skipped" for stage in body["stages"][1:])
    assert body["provenance"]["external_calls"] == 0


def test_analyze_reports_unknown_watcher_without_listing_configuration(tmp_path: Path) -> None:
    watchers_path = tmp_path / "watchers.yml"
    _write_watchers(watchers_path)
    app = create_app(
        db_path=tmp_path / "control.db",
        watchers_path=watchers_path,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze",
            json={"watcher_name": "unknown-watch", "text": "Villa for rent"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": {"code": "watcher_not_found"}}


def test_analyze_is_unavailable_when_configuration_is_missing(tmp_path: Path) -> None:
    app = create_app(
        db_path=tmp_path / "control.db",
        watchers_path=tmp_path / "missing.yml",
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze",
            json={"watcher_name": "housing-watch", "text": "Villa for rent"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "watchers_config_missing"}}
