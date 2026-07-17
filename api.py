"""Read-only operational control plane for Eidolon.

The API intentionally owns no Telegram or LLM clients. It exposes health,
daily aggregate metrics, and a deterministic rule-only analysis endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

import aiosqlite
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from config.watchers import Watcher, WatcherConfigError, load_watchers
from pipeline.filters import RuleFilter
from pipeline.models import StageStatus

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(os.environ.get("DB_PATH", "data/eidolon.db"))
DEFAULT_WATCHERS_PATH = Path(os.environ.get("WATCHERS_PATH", "config/watchers.yml"))


class ControlPlaneDatabase(Protocol):
    """Minimal lifecycle and query contract required by the control plane."""

    @property
    def conn(self) -> aiosqlite.Connection: ...

    async def connect(self) -> None: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class ReadOnlySQLite:
    """SQLite adapter that cannot create or mutate the operational database."""

    path: Path
    _conn: aiosqlite.Connection | None = field(default=None, init=False)

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Read-only database is not connected")
        return self._conn

    async def connect(self) -> None:
        uri = f"{self.path.resolve().as_uri()}?mode=ro"
        self._conn = await aiosqlite.connect(uri, uri=True)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA query_only=ON")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


class LivenessResponse(BaseModel):
    """Dependency-free process liveness."""

    status: Literal["ok"] = "ok"
    service: Literal["eidolon-control-plane"] = "eidolon-control-plane"


class ComponentReadiness(BaseModel):
    """Safe readiness metadata for one local component."""

    ready: bool
    code: str | None = None
    item_count: int | None = None


class ReadinessComponents(BaseModel):
    database: ComponentReadiness
    configuration: ComponentReadiness


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    components: ReadinessComponents


class MetricCounts(BaseModel):
    messages_total: int = Field(ge=0)
    passed_level1: int = Field(ge=0)
    passed_level2: int = Field(ge=0)
    passed_level3: int = Field(ge=0)
    alerts_sent: int = Field(ge=0)


class WatcherMetrics(MetricCounts):
    watcher_name: str


class StatsProvenance(BaseModel):
    source: Literal["sqlite.filter_stats"] = "sqlite.filter_stats"
    timezone: Literal["UTC"] = "UTC"


class StatsResponse(BaseModel):
    date: date
    watchers: list[WatcherMetrics]
    totals: MetricCounts
    provenance: StatsProvenance = Field(default_factory=StatsProvenance)


class AnalyzeRequest(BaseModel):
    """A side-effect-free message analysis request."""

    model_config = ConfigDict(extra="forbid")

    watcher_name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    text: str = Field(max_length=20_000)


class StageTrace(BaseModel):
    stage: Literal["rules", "embeddings", "llm"]
    status: StageStatus
    passed: bool | None
    reason: str
    matched_keyword: str | None = None


class AnalyzeProvenance(BaseModel):
    execution_mode: Literal["rule_only"] = "rule_only"
    pipeline_version: Literal["rules-v1"] = "rules-v1"
    deterministic: Literal[True] = True
    watcher_config_fingerprint: str
    external_calls: Literal[0] = 0
    persisted: Literal[False] = False


class AnalyzeResponse(BaseModel):
    watcher_name: str
    accepted: bool
    stages: list[StageTrace]
    provenance: AnalyzeProvenance


@dataclass(slots=True)
class ControlPlane:
    """Injected local dependencies and their readiness state."""

    database: ControlPlaneDatabase
    watchers_path: Path
    injected_watchers: tuple[Watcher, ...] | None = None
    database_started: bool = False
    database_error: str | None = None
    configuration_error: str | None = None
    watchers: dict[str, Watcher] = field(default_factory=dict)

    async def start(self) -> None:
        try:
            await self.database.connect()
        except Exception as error:
            self.database_started = False
            self.database_error = "database_start_failed"
            logger.warning("Control-plane database unavailable: %s", type(error).__name__)
        else:
            self.database_started = True
            self.database_error = None

        self._load_configuration()

    async def stop(self) -> None:
        try:
            await self.database.close()
        except Exception as error:
            logger.warning("Control-plane database close failed: %s", type(error).__name__)
        finally:
            self.database_started = False

    def _load_configuration(self) -> None:
        if self.injected_watchers is not None:
            loaded = list(self.injected_watchers)
            if not loaded:
                self.watchers = {}
                self.configuration_error = "watchers_config_empty"
                return
        else:
            try:
                loaded = load_watchers(self.watchers_path)
            except WatcherConfigError:
                self.watchers = {}
                self.configuration_error = (
                    "watchers_config_missing"
                    if not self.watchers_path.exists()
                    else "watchers_config_invalid"
                )
                return

        self.watchers = {watcher.name: watcher for watcher in loaded}
        self.configuration_error = None

    async def database_readiness(self) -> ComponentReadiness:
        if not self.database_started:
            return ComponentReadiness(
                ready=False,
                code=self.database_error or "database_not_started",
            )
        try:
            async with self.database.conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'filter_stats'
                """
            ) as cursor:
                row = await cursor.fetchone()
        except Exception as error:
            logger.warning("Control-plane database readiness failed: %s", type(error).__name__)
            return ComponentReadiness(ready=False, code="database_query_failed")
        return ComponentReadiness(
            ready=row is not None,
            code=None if row is not None else "database_schema_missing",
        )

    def configuration_readiness(self) -> ComponentReadiness:
        return ComponentReadiness(
            ready=self.configuration_error is None and bool(self.watchers),
            code=self.configuration_error,
            item_count=len(self.watchers),
        )


def _control_plane(request: Request) -> ControlPlane:
    return cast(ControlPlane, request.app.state.control_plane)


ControlPlaneDependency = Annotated[ControlPlane, Depends(_control_plane)]
StatsDateQuery = Annotated[date | None, Query(alias="date")]


async def _live() -> LivenessResponse:
    return LivenessResponse()


async def _ready(
    response: Response,
    control_plane: ControlPlaneDependency,
) -> ReadinessResponse:
    database = await control_plane.database_readiness()
    configuration = control_plane.configuration_readiness()
    ready = database.ready and configuration.ready
    response.status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    readiness_status: Literal["ready", "not_ready"] = "ready" if ready else "not_ready"
    return ReadinessResponse(
        status=readiness_status,
        components=ReadinessComponents(
            database=database,
            configuration=configuration,
        ),
    )


async def _stats(
    control_plane: ControlPlaneDependency,
    target_date: StatsDateQuery = None,
) -> StatsResponse:
    database = await control_plane.database_readiness()
    if not database.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": database.code or "database_unavailable"},
        )

    selected_date = target_date or datetime.now(UTC).date()
    metrics = await _read_daily_metrics(control_plane.database, selected_date)
    for watcher_name in control_plane.watchers:
        metrics.setdefault(watcher_name, _empty_counts())

    watcher_metrics = [
        _watcher_metrics(watcher_name, counts) for watcher_name, counts in sorted(metrics.items())
    ]
    return StatsResponse(
        date=selected_date,
        watchers=watcher_metrics,
        totals=_sum_counts(metrics.values()),
    )


async def _analyze(
    payload: AnalyzeRequest,
    control_plane: ControlPlaneDependency,
) -> AnalyzeResponse:
    configuration = control_plane.configuration_readiness()
    if not configuration.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": configuration.code or "configuration_unavailable"},
        )

    watcher = control_plane.watchers.get(payload.watcher_name)
    if watcher is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "watcher_not_found"},
        )

    result = RuleFilter(watcher).check(payload.text)
    return AnalyzeResponse(
        watcher_name=watcher.name,
        accepted=result.passed,
        stages=[
            StageTrace(
                stage="rules",
                status=StageStatus.OK,
                passed=result.passed,
                reason=result.reason,
                matched_keyword=result.matched_keyword,
            ),
            StageTrace(
                stage="embeddings",
                status=StageStatus.SKIPPED,
                passed=None,
                reason="control_plane_rule_only",
            ),
            StageTrace(
                stage="llm",
                status=StageStatus.SKIPPED,
                passed=None,
                reason="control_plane_rule_only",
            ),
        ],
        provenance=AnalyzeProvenance(
            watcher_config_fingerprint=_watcher_fingerprint(watcher),
        ),
    )


async def _read_daily_metrics(
    database: ControlPlaneDatabase,
    selected_date: date,
) -> dict[str, MetricCounts]:
    async with database.conn.execute(
        """
        SELECT
            watcher_name,
            messages_total,
            passed_level1,
            passed_level2,
            passed_level3,
            alerts_sent
        FROM filter_stats
        WHERE date = ?
        ORDER BY watcher_name
        """,
        (selected_date.isoformat(),),
    ) as cursor:
        rows = await cursor.fetchall()

    return {
        str(row["watcher_name"]): MetricCounts(
            messages_total=int(row["messages_total"] or 0),
            passed_level1=int(row["passed_level1"] or 0),
            passed_level2=int(row["passed_level2"] or 0),
            passed_level3=int(row["passed_level3"] or 0),
            alerts_sent=int(row["alerts_sent"] or 0),
        )
        for row in rows
    }


def _empty_counts() -> MetricCounts:
    return MetricCounts(
        messages_total=0,
        passed_level1=0,
        passed_level2=0,
        passed_level3=0,
        alerts_sent=0,
    )


def _watcher_metrics(watcher_name: str, counts: MetricCounts) -> WatcherMetrics:
    return WatcherMetrics(
        watcher_name=watcher_name,
        messages_total=counts.messages_total,
        passed_level1=counts.passed_level1,
        passed_level2=counts.passed_level2,
        passed_level3=counts.passed_level3,
        alerts_sent=counts.alerts_sent,
    )


def _sum_counts(counts: Iterable[MetricCounts]) -> MetricCounts:
    return MetricCounts(
        messages_total=sum(item.messages_total for item in counts),
        passed_level1=sum(item.passed_level1 for item in counts),
        passed_level2=sum(item.passed_level2 for item in counts),
        passed_level3=sum(item.passed_level3 for item in counts),
        alerts_sent=sum(item.alerts_sent for item in counts),
    )


def _watcher_fingerprint(watcher: Watcher) -> str:
    payload = json.dumps(
        watcher.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def create_app(
    *,
    database: ControlPlaneDatabase | None = None,
    db_path: Path | None = None,
    watchers_path: Path | None = None,
    watchers: Sequence[Watcher] | None = None,
) -> FastAPI:
    """Build a control-plane app with injectable local dependencies."""

    if database is not None and db_path is not None:
        raise ValueError("Pass either database or db_path, not both")

    control_plane = ControlPlane(
        database=database if database is not None else ReadOnlySQLite(db_path or DEFAULT_DB_PATH),
        watchers_path=watchers_path or DEFAULT_WATCHERS_PATH,
        injected_watchers=tuple(watchers) if watchers is not None else None,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await control_plane.start()
        try:
            yield
        finally:
            await control_plane.stop()

    application = FastAPI(
        title="Eidolon Control Plane",
        summary="Read-only health, metrics, and deterministic analysis",
        description=(
            "Operational API for the Eidolon Telegram intelligence pipeline. "
            "It never sends Telegram messages and never invokes an AI provider."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.control_plane = control_plane
    application.add_api_route(
        "/health/live",
        _live,
        methods=["GET"],
        response_model=LivenessResponse,
        tags=["health"],
        summary="Process liveness",
    )
    application.add_api_route(
        "/health/ready",
        _ready,
        methods=["GET"],
        response_model=ReadinessResponse,
        responses={503: {"description": "Database or watcher configuration is unavailable"}},
        tags=["health"],
        summary="Dependency readiness",
    )
    application.add_api_route(
        "/stats",
        _stats,
        methods=["GET"],
        response_model=StatsResponse,
        responses={503: {"description": "SQLite is unavailable"}},
        tags=["observability"],
        summary="Daily watcher metrics",
    )
    application.add_api_route(
        "/v1/analyze",
        _analyze,
        methods=["POST"],
        response_model=AnalyzeResponse,
        responses={
            404: {"description": "Watcher was not found"},
            503: {"description": "Watcher configuration is unavailable"},
        },
        tags=["analysis"],
        summary="Explain a deterministic rule-only decision",
    )
    return application


app = create_app()
