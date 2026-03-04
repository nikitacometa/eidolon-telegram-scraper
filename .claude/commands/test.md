Run the full test suite with coverage:

```bash
python3 -m pytest tests/ -v --cov=pipeline --cov=storage --cov=config --cov-report=term-missing --timeout=30
```

Report:
1. Number of tests passed/failed/skipped
2. Coverage percentage per module
3. Full tracebacks for any failures
4. Suggest missing test coverage areas
