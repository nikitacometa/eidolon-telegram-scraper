# Contributing

Thank you for improving Eidolon. Keep changes focused, typed, tested, and safe
for a daemon that handles private Telegram data.

## Local setup

Use Python 3.12 and the committed lockfile:

```bash
uv sync --locked --dev
cp .env.example .env
cp config/watchers.example.yml config/watchers.yml
```

Use a dedicated Telegram account. Never place real credentials, session
strings, chat IDs, databases, or message samples in commits.

## Development checks

Run the same gates as CI before opening a pull request:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov
uv run bandit -r . -c pyproject.toml
uv run pip-audit --local --skip-editable
```

Tests must mock Telegram and model providers. Add accepted, rejected,
malformed-input, timeout, and provider-failure cases where relevant.

## Commits and pull requests

Use a lowercase imperative subject, for example `fix retry accounting`.
Update the related `E-NNN` entry in `BOARD.md`. Pull requests should explain
the behavior change, tests run, and any configuration, storage, cost, or
deployment impact.

Keep Telegram behavior read-only unless a task explicitly authorizes writes
and defines an approval workflow. Report vulnerabilities through GitHub's
private vulnerability reporting instead of a public issue.

## Containers

`docker compose up --build` runs one MTProto worker and a separate read-only
control plane. The unauthenticated API binds to `127.0.0.1:8000`; keep it on
loopback, or place authentication and TLS at a trusted reverse proxy before
exposing it to any network.
