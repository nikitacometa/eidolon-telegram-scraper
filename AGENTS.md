# Repository Guidelines

## Project Structure & Module Organization

`main.py` owns the Telethon session, ingress, workers, recovery, and summaries.
`pipeline/` contains typed stages: rules, embeddings, LLM
classification, orchestration, and Telegram delivery. SQLite code and migrations live in
`storage/`; validated settings and watcher policy live in `config/`. `api.py` is a
separate read-only FastAPI control plane. Keep tests in `tests/test_<module>.py`,
evaluation corpora in `evals/data/`, committed results in `docs/`, and deployment
scripts in `deploy/`.

## Build, Test, and Development Commands

- `uv sync --locked --dev` installs the locked dependency graph.
- `uv run eidolon-auth` creates the dedicated account session and saves it to `.env`.
- `uv run eidolon-scraper` starts the worker; never run two copies for one session.
- `uv run uvicorn api:app --host 127.0.0.1 --port 8000` starts the control plane.
- `uv run pytest --cov` runs unit and integration tests with the 80% coverage gate.
- `uv run ruff format --check . && uv run ruff check .` checks style and lint rules.
- `uv run mypy` runs strict static type checks.
- `uv run eidolon-eval --level 1` runs the credential-free relevance baseline.
- `docker compose -f compose.yml config --quiet` validates container configuration.

## Coding Style & Naming Conventions

Target Python 3.12, four-space indentation, and a 100-character line limit. Use
`snake_case` for functions/modules, `PascalCase` for classes, frozen or slotted typed
contracts where practical, and explicit return types. Keep blocking work outside the
event loop. Preserve the rules → embeddings → LLM order and make degraded provider
behavior explicit rather than swallowing exceptions.

## Testing Guidelines

Pytest uses strict markers, asyncio support, timeouts, and branch coverage. Cover accepted,
rejected, duplicate, cancellation, retry, malformed-output, and provider-failure paths.
Mock Telegram/OpenAI in ordinary tests; only explicit evaluation commands may call live
providers. Add anonymized EN/RU cases for relevance changes, calibrate on a development
set, and reserve a new holdout for final measurement.

## Commit & Pull Request Guidelines

History uses lowercase imperative subjects such as `fix retry accounting`. Update the
related `E-NNN` row in `BOARD.md`, then commit each logical unit. Pull requests must
describe behavior, tests, configuration/storage/cost impact, and linked tasks. Include
screenshots only for user-visible alert changes.

## Security & Operational Safety

Copy `.env.example` and `config/watchers.example.yml`; never commit credentials, session
strings, real chat IDs, databases, or message data. Keep Telegram behavior read-only
unless an approved task defines consent and rate limits. Bind the unauthenticated API to
loopback, and document at-least-once—not exactly-once—delivery semantics.
