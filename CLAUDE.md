# Eidolon

AI-powered digital presence for Telegram. Connects to a real account via MTProto, monitors group chats, filters messages with multi-level intelligence (rules → embeddings → LLM), sends alerts, generates summaries, and eventually responds on behalf of the user.

Named after εἴδωλον — the phantom double of a living person in Greek mythology.

## Stack

- **Runtime**: Python 3.12+
- **Telegram**: Telethon v1.42 (MTProto userbot, from Codeberg)
- **LLM**: OpenAI (GPT-4.1-mini for filtering, GPT-4.1 for analysis)
- **Embeddings**: OpenAI text-embedding-3-small (pre-filtering)
- **Storage**: SQLite + aiosqlite (messages, alerts, chat metadata)
- **Vector store**: ChromaDB embedded (semantic filtering)
- **Deployment**: launchd (macOS) / systemd (Linux) — persistent daemon

## Key Commands

```bash
# Run the main listener
python3 main.py

# Generate session string (first time only)
python3 auth.py

# Run tests
python3 -m pytest tests/

# Lint & format
ruff check . && ruff format --check .
```

## Architecture

```
Telegram Groups (MTProto listener)
    ↓ events.NewMessage
Level 1: Rule-based filter (0 cost)
    │  Keywords, regex, sender allow/deny, message length
    │  Drops ~80-90% of messages
    ↓
Level 2: Embedding similarity (cheap)
    │  text-embedding-3-small → ChromaDB cosine similarity
    │  Drops ~50-70% of remaining
    ↓
Level 3: LLM analysis (expensive, rare)
    │  GPT-4.1-mini: relevance classification
    │  GPT-4.1: summaries, deep analysis
    ↓
Alert Dispatcher
    │  → Telegram (via @ClaudePantheon_Bot)
    │  → Future: email, webhook, dashboard
    ↓
SQLite (message log, alert history, filter stats)
```

## Configuration

All secrets in `.env` (never committed):
```
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_SESSION_STRING=...
TELEGRAM_PHONE=...
OPENAI_API_KEY=...          # LLM + embeddings
PANTHEON_BOT_TOKEN=...      # alert delivery via existing bot
PANTHEON_CHAT_ID=...        # Nikita's chat ID
```

Chat filters and watch rules in `config/watchers.yml`:
```yaml
watchers:
  - name: phangan-housing
    chats: [-100123456789, -100987654321]
    rules:
      keywords: [house, villa, rent, сдаю, дом, аренда, bungalow]
      languages: [en, ru, th]
    alert: immediate
    llm_level: 3  # full LLM analysis

  - name: crypto-signals
    chats: [-100111222333]
    rules:
      keywords: [signal, buy, sell, entry]
    alert: digest  # batch every 4 hours
    llm_level: 2   # embedding filter only
```

## Project Structure

```
eidolon-telegram/
├── main.py                 # Entry point: Telethon client + event loop
├── auth.py                 # Session string generator (run once)
├── pipeline/
│   ├── __init__.py
│   ├── ingestion.py        # Telethon event handlers, message extraction
│   ├── filters.py          # Rule-based filter (Level 1)
│   ├── embeddings.py       # Embedding similarity filter (Level 2)
│   ├── llm.py              # LLM calls: classification, summarization (Level 3)
│   └── dispatcher.py       # Alert routing (Telegram bot, future: email/webhook)
├── storage/
│   ├── __init__.py
│   ├── db.py               # aiosqlite wrapper, migrations
│   └── schema.sql          # DDL: messages, alerts, chats, filter_stats
├── config/
│   ├── __init__.py
│   ├── settings.py         # Pydantic settings from .env
│   └── watchers.yml        # Chat watch rules and filter config
├── tests/
│   ├── test_filters.py
│   ├── test_pipeline.py
│   └── test_storage.py
├── docs/
│   └── research.md         # Tech research findings
├── .env.example            # Template for secrets
├── .gitignore
├── requirements.txt
├── requirements-dev.txt    # Dev dependencies (pytest, ruff, mypy)
├── pyproject.toml
├── CLAUDE.md               # AI agent instructions
├── BOARD.md                # Task board (E-NNN)
├── README.md
└── tmp/                    # Working files (gitignored)
```

## Rules

- **Never commit `.env` or session strings.** All secrets live in `.env` (gitignored).
- **Telethon session is sacred.** Never create multiple concurrent clients with the same session. One process, one session.
- **Read-only first.** MVP monitors only. Sending messages is a future phase with explicit user approval per chat.
- **Cost-conscious LLM usage.** Always run rule-based + embedding filters before LLM. Use GPT-4.1-mini for classification, GPT-4.1 only for deep analysis.
- **Respect Telegram rate limits.** No mass joins, no spam, human-like pauses for any writes.
- **Separate account.** Never connect to the user's primary Telegram account. Dedicated number only.
- **Test filters independently.** Each filter level must have unit tests with real message samples.
- **Watchers config is the source of truth** for what chats to monitor and how to filter.
- **Alerts through existing Pantheon bot** (@ClaudePantheon_Bot) — no new bot needed for MVP.

## Phases

1. **MVP**: Read-only monitor → keyword filter → alert via Telegram bot
2. **Smart Filter**: Add embedding pre-filter + LLM classification
3. **Summarizer**: Periodic chat digests (morning/evening)
4. **Responder**: Reply to specific patterns (with user approval workflow)
5. **Autonomous**: Join chats, initiate conversations, manage personas

## Task Board

Tasks in `BOARD.md`. Format: Pantheon table (E-NNN IDs).

## Commit Discipline

- Always commit after completing a task or logical unit of work — never leave finished work uncommitted
- Use lowercase verb, concise English: `add`, `fix`, `update`, `remove`, `refactor`
- Push after committing unless explicitly told not to
- If changes need review, commit anyway — better to fix in a follow-up than leave uncommitted
- Update task board status before committing the related work

## Pantheon Integration

- **Registered in:** `~/dev/pantheon-command/projects.yml`
- **Task file:** `BOARD.md` (pantheon format, E-NNN IDs)
- **Cross-project status:** `/status` from any project
- **Alerts:** via @ClaudePantheon_Bot (shared infrastructure)
