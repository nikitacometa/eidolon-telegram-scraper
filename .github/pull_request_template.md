## Summary

Describe the problem and the behavior changed by this pull request.

## Validation

- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run mypy`
- [ ] `uv run pytest --cov`
- [ ] Tests cover failure and malformed-input paths

## Safety and operations

- [ ] No credentials, session strings, personal chat IDs, or message data are included
- [ ] Telegram monitoring remains read-only, or write behavior has explicit approval
- [ ] Configuration, storage, cost, or deployment impact is documented below
- [ ] The relevant `BOARD.md` task is updated

Operational notes:

<!-- Include migrations, environment changes, rollout risks, or "None". -->
