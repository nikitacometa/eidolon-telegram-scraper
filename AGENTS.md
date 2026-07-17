# Repository Guidelines

## Project Structure & Module Organization

`main.py` starts the Telethon listener; `auth.py` creates the dedicated account session.
Message processing lives in `pipeline/`: ingestion, filters, LLM analysis, summaries, and
alert dispatch. Configuration models and the watcher source of truth are under `config/`,
especially `config/watchers.yml`. SQLite access and schemas live in `storage/`. Keep tests
in `tests/`, using names such as `test_filters.py`. Deployment files are in `deploy/`;
longer technical notes belong in `docs/`.

## Build, Test, and Development Commands

- `python3 -m venv .venv && source .venv/bin/activate` creates a local environment.
- `pip install -r requirements.txt -r requirements-dev.txt` installs runtime and tooling
  dependencies.
- `python3 auth.py` generates the initial Telegram session; run it only for the dedicated
  account and never concurrently with the listener.
- `python3 main.py` runs the monitoring daemon locally.
- `python3 -m pytest tests/` runs the complete test suite.
- `ruff check . && ruff format --check .` verifies linting and formatting.
- `mypy .` runs strict static type checks.

## Coding Style & Naming Conventions

Target Python 3.12 and use four-space indentation. Ruff enforces a 100-character line
length and checks imports, naming, modernization, bug patterns, and async usage. Use
`snake_case` for functions and modules, `PascalCase` for classes, and explicit type
annotations for new code. Keep async I/O non-blocking and preserve the pipeline order:
rules, embeddings, then LLM.

## Testing Guidelines

Pytest uses strict markers and automatic asyncio support. Name files `test_<module>.py`
and functions `test_<behavior>`. Test every filter level, including accepted, rejected,
malformed, and provider-failure cases. Mock Telegram and OpenAI calls; tests must not
contact live services or require secrets. Run the full suite before submitting changes.

## Commit & Pull Request Guidelines

Follow the existing history: lowercase imperative subjects such as `add daily digest`
or `fix watcher validation`. Update the relevant `E-NNN` entry in `BOARD.md`, then commit
each logical unit. Pull requests should explain the behavior change, link the task or
issue, list tests run, and call out configuration or deployment impact. Include logs or
screenshots only when they clarify user-visible alert behavior.

## Security & Operational Safety

Copy `.env.example` to `.env`; never commit credentials, session strings, databases, or
generated state. Use one process per Telethon session. Keep monitoring read-only unless
the task explicitly authorizes Telegram writes, and treat `config/watchers.yml` as the
authoritative monitoring scope.
