Run the full linting and type checking pipeline:

```bash
ruff check . && ruff format --check . && python3 -m mypy pipeline/ storage/ config/ --strict
```

Report all issues grouped by severity:
1. Type errors (mypy)
2. Bugs and style violations (ruff)
3. Formatting issues

Fix auto-fixable issues with `ruff check --fix .` and `ruff format .` if the user approves.
