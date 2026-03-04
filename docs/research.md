# Eidolon — Technology Research

> Date: 2026-03-04

## Telegram Client Libraries

| Library | Stars | Status | Last Release | Recommendation |
|---------|-------|--------|-------------|----------------|
| **Telethon** | 11.9k | Active (moved to Codeberg) | v1.42.0 (Nov 2025) | **Use this** |
| Pyrogram | ~8k | **Archived Dec 2024** | v2.0.106 | Dead, do not use |
| Hydrogram | 214 | Moderate | v0.2.0 (Jun 2024) | Pyrogram fork, backup option |
| PyroFork | 277 | Active | Active (126 tags) | Pyrogram fork, alternative |
| python-telegram-bot | ~27k | Active | v22.0 | Bot API only, not for userbots |

**Key facts:**
- Telethon moved from GitHub to Codeberg: https://codeberg.org/Lonami/Telethon
- v2 alpha (2.0.0a0) exists but is a complete rewrite — not production-ready
- 1.79M monthly PyPI downloads — dominant in the space
- Built-in FloodWaitError handling (`flood_sleep_threshold`)

## Existing OSS Projects

| Project | Stars | Stack | LLM |
|---------|-------|-------|-----|
| [telegram-chat-summarizer](https://github.com/dev0x13/telegram-chat-summarizer) | 91 | Telethon + LangChain | GPT-4 Turbo |
| [telegram-summary-bot](https://github.com/asukaminato0721/telegram-summary-bot) | 182 | TypeScript, Cloudflare | Gemini 2.5 Flash |
| [elizaOS/discord-summarizer](https://github.com/elizaOS/discord-summarizer) | 90 | Python, LangChain | Ollama (local) |

**Conclusion:** No production-quality "userbot + LLM monitoring" solution exists. This is a gap.

## Architecture Decision

Chose **Pattern 2: Buffer + Periodic Processing** over direct processing or message queue.

Three-level filter pipeline:
1. **Rule-based** (free) — keywords, regex, sender lists → drops 80-90%
2. **Embedding similarity** (~$0.02/1M tokens) — cosine similarity vs reference vectors → drops 50-70%
3. **LLM classification** (expensive) — only for filtered candidates

## Cost Model

| Component | Cost/1M tokens | Use case |
|-----------|---------------|----------|
| OpenAI text-embedding-3-small | $0.02 input | Pre-filtering |
| Claude Haiku 3.5 | $0.80 input / $4 output | Quick classification |
| Claude Sonnet (Batch API) | $1.50 input / $7.50 output | Summaries (50% off) |
| GPT-4o-mini | $0.15 input / $0.60 output | Cheapest alternative |

## Telegram Account Safety

| Action | Risk Level |
|--------|-----------|
| Passive message reading | Very low |
| Sending < 1 msg/min | Low |
| Sending to > 20 chats/min | High |
| Mass group joining | High |
| Identical messages to many | Critical |

**Best practices:** dedicated phone number, StringSession (reuse), single client instance, human-like delays.

## Sources

- [Telethon Codeberg](https://codeberg.org/Lonami/Telethon)
- [Telethon docs v1.42](https://docs.telethon.dev/en/stable/)
- [telegram-chat-summarizer](https://github.com/dev0x13/telegram-chat-summarizer)
- [Anthropic Batch API](https://docs.anthropic.com/en/docs/build-with-claude/message-batches)
- [GPTCache](https://github.com/zilliztech/GPTCache)
- [SentryDock](https://www.sentrydock.com/) — commercial analog
- [PyPI stats Telethon](https://www.pypistats.org/packages/telethon)

---

## Python Dev Tooling for Claude Code (2026-03-04)

### MCP Servers для Python

| Сервер | GitHub | Что делает | Зрелость |
|--------|--------|-----------|---------|
| **mcp-code-checker** | [MarcusJellinghaus/mcp-code-checker](https://github.com/MarcusJellinghaus/mcp-code-checker) | pylint + pytest + mypy через MCP, LLM-friendly prompts | Stable (69 commits, активен) |
| **mcp_python_toolbox** | [gianlucamazza/mcp_python_toolbox](https://github.com/gianlucamazza/mcp_python_toolbox) | AST analysis, venv management, Black/Pylint | Archived (Feb 2026, не использовать) |
| **mcp_pytest_service** | [kieranlal/mcp_pytest_service](https://github.com/kieranlal/mcp_pytest_service) | pytest results → MCP memory entities | Experimental (PoC, Node.js backend) |
| **modelcontextprotocol/python-sdk** | [python-sdk](https://github.com/modelcontextprotocol/python-sdk) | Official Python SDK для создания MCP серверов | Stable (spec 2025-11-25) |

**Рекомендация:** Из готовых Python MCP серверов только `mcp-code-checker` production-ready. Для pytest/linting интеграции проще сделать custom Bash-инструмент через Claude Code Skills.

### Claude Code Skills для Python

**Готовая skill** из [playbooks.com/skills/laurigates/claude-plugins/python-testing](https://playbooks.com/skills/laurigates/claude-plugins/python-testing):
- Команды: `uv run pytest`, `uv run pytest --cov --cov-report=term-missing`
- Покрывает: fixtures, parametrize, asyncio, mocking, pytest-mock
- Model: haiku, Tools: Bash, Read, Grep, Glob

**awesome-claude-code subagents**: [VoltAgent/awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents) — 127+ агентов, включая `python-pro` в категории Language Specialists. 12.4k stars.

### Конфигурация .claude/agents/ (формат)

```markdown
---
name: python-reviewer
description: Python code review specialist. Proactively reviews after any .py file changes.
tools: Read, Grep, Glob, Bash
model: sonnet
memory: project
---

System prompt here...
```

Поддерживаемые поля frontmatter: `name`, `description`, `tools`, `disallowedTools`, `model` (sonnet/opus/haiku/inherit), `permissionMode`, `maxTurns`, `skills`, `mcpServers`, `hooks`, `memory` (user/project/local), `background`, `isolation`.

### Pre-commit Hooks (Python 3.12+)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.4
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.8.3
    hooks:
      - id: bandit
        args: [-c, pyproject.toml]
```

**Не включать pytest в pre-commit** — слишком долго. Запускать отдельно в CI или через skill.

### pyproject.toml (полная конфигурация)

```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM", "ASYNC"]
ignore = ["E501"]
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "ARG"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
plugins = ["pydantic.mypy"]  # если используется Pydantic

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"
asyncio_mode = "auto"  # для pytest-asyncio

[tool.bandit]
exclude_dirs = ["tests", ".venv"]
skips = ["B101"]  # assert в тестах
```

**Ключевое правило ruff**: rule group `ASYNC` (AIO) специально для async-кода — ловит `asyncio.sleep(0)`, блокирующие вызовы в async-контексте.

### Сравнение Python Quality Tools (2026)

#### Linting / Formatting

| Tool | Скорость | Заменяет | Рекомендация |
|------|---------|---------|-------------|
| **Ruff** | 200x быстрее flake8 (Rust) | flake8, isort, pyupgrade, eradicate | **Использовать** |
| flake8 | Медленно | — | Устарел, не нужен с Ruff |
| pylint | Очень медленно | — | Только если нужны специфичные проверки |
| black | Медленно | — | Заменён `ruff format` |

#### Type Checking

| Tool | Скорость | Async support | Рекомендация |
|------|---------|--------------|-------------|
| **mypy** | Медленно на больших кодовых базах | Хорошо | Стабильный выбор, reference implementation |
| **pyright** | Быстро | Отлично | VS Code / Pylance, лучше для IDE |
| ty (Astral) | Очень быстро (Rust) | — | Experimental, планируется интеграция с ruff |
| pyrefly (Meta) | Быстро (Rust) | — | Beta, замена pyre |

**Рекомендация для Eidolon:** mypy + strict mode. Для IDE — pyright/Pylance.

#### Dependency Management

| Tool | Скорость | Функции | Рекомендация |
|------|---------|---------|-------------|
| **uv** | 10-100x быстрее pip (Rust) | venv, lockfile, Python version mgmt | **Использовать** для новых проектов |
| poetry | Медленно | Полный lifecycle, публикация | Для библиотек с PyPI публикацией |
| pip + pip-tools | Медленно | Простой lockfile | Legacy, мигрировать на uv |

#### Security

| Tool | Что проверяет | Интеграция |
|------|--------------|-----------|
| **bandit** | SAST — уязвимые паттерны в коде (47 checks, 7 категорий) | pre-commit, CI |
| **pip-audit** | Зависимости с CVE (PyPI Advisory Database) | CI, не в pre-commit |
| **safety** | Зависимости с CVE (платная БД) | Альтернатива pip-audit |

Использовать **bandit + pip-audit** вместе — покрывают разные векторы.

#### pytest Plugins (топ для async daemon)

| Plugin | Назначение | Обязателен |
|--------|-----------|-----------|
| **pytest-asyncio** | `@pytest.mark.asyncio`, asyncio_mode="auto" | Да — async код |
| **pytest-cov** | Coverage reports | Да |
| **pytest-mock** | Удобный `mocker` fixture | Да |
| **pytest-xdist** | Параллельный запуск | По желанию |
| **pytest-timeout** | Предотвращение висящих тестов | Рекомендуется для daemon |
| **respx** | Mock для httpx async запросов | Если используется httpx |

### Sources (tooling research)

- [mcp-code-checker GitHub](https://github.com/MarcusJellinghaus/mcp-code-checker)
- [awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents)
- [Claude Code Sub-agents docs](https://code.claude.com/docs/en/sub-agents)
- [python-testing skill](https://playbooks.com/skills/laurigates/claude-plugins/python-testing)
- [ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit)
- [Ruff configuration docs](https://docs.astral.sh/ruff/configuration/)
- [Future Python type checkers comparison](https://sinon.github.io/future-python-type-checkers/)
- [uv vs Poetry 2026](https://scopir.com/posts/best-python-package-managers-2026/)
- [bandit GitHub](https://github.com/PyCQA/bandit)
- [pip-audit PyPI](https://pypi.org/project/pip-audit/)
