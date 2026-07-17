# Changelog

Notable changes are documented here using
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions.

## [Unreleased]

No unreleased changes.

## [0.2.0] - 2026-07-17

### Added

- Durable pre-queue ingestion, restartable watcher jobs, leased alert outbox, bounded
  workers, retry backoff, and continuous retention sweeps.
- Typed model provenance with latency/token usage, watcher intent and degradation
  policies, verbatim evidence checks, and citation-validated structured digests.
- Calibration and evaluation commands with hashed manifests, EN/RU corpora, committed
  online results, and fail-closed degraded-provider gates.
- Read-only FastAPI health, stats, backlog, and deterministic analysis endpoints.
- Non-root, health-checked container, hardened Compose services, and deployment scripts.
- Locked `uv` CI with Ruff, strict mypy, coverage, Bandit, pip-audit, shell, and image
  build gates.

### Changed

- Renamed the project to `eidolon-telegram-scraper`.
- Rebuilt public configuration, documentation, contributor workflow, and dependency
  policy for a reproducible portfolio release.
- Added watcher-policy fingerprints for recovery and fencing tokens for leased alert
  delivery; degraded required AI stages now reject by default.

## [0.1.0] - 2026-03-04

### Added

- Initial read-only Telegram monitor with rules, embeddings, LLM classification,
  alerts, daily summaries, and SQLite persistence.
