Run security audit on the codebase:

```bash
python3 -m bandit -r pipeline/ storage/ config/ -c pyproject.toml && pip-audit
```

Report:
1. Code vulnerabilities (bandit) — grouped by severity (High/Medium/Low)
2. Dependency CVEs (pip-audit)
3. Actionable remediation steps for each finding

Pay special attention to: hardcoded secrets, SQL injection in aiosqlite queries, unsafe deserialization, and Telegram session handling.
