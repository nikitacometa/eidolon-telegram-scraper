# Eidolon — Task Board

> Last updated: 2026-03-05

## Conventions

- **ID format**: `E-NNN` (sequential, never reuse)
- **Statuses**: `todo` | `in_progress` | `blocked` | `done`
- **Priorities**: `critical` | `high` | `medium` | `low`
- Next available ID: **E-019**

---

## Active

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-018 | Live test: summary digest at 20:00 ICT | todo | medium | Verify daily summary arrives tonight |

## Backlog

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-009 | Pipeline: embedding similarity filter (Level 2) | todo | medium | `pipeline/embeddings.py` — ChromaDB + text-embedding-3-small |
| E-013 | Monitoring: filter stats dashboard | todo | low | Track pass/drop rates per watcher per level |

## Done

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-001 | Telegram auth: session string generator | done | critical | `auth.py` |
| E-002 | Core: Telethon client + event loop | done | critical | `main.py` with graceful shutdown |
| E-003 | Storage: SQLite wrapper + migrations | done | critical | `storage/db.py` — 8 tests |
| E-004 | Pipeline: message ingestion handler | done | critical | `pipeline/ingestion.py` |
| E-005 | Pipeline: rule-based keyword filter (Level 1) | done | high | `pipeline/filters.py` — 16 tests |
| E-006 | Pipeline: alert dispatcher via Pantheon bot | done | high | `pipeline/dispatcher.py` — 6 tests |
| E-007 | Integration: wire pipeline into main event loop | done | high | Full pipeline in main.py — 4 integration tests |
| E-008 | First watcher: Phangan housing monitor | done | high | Live with real chat IDs |
| E-010 | Pipeline: LLM classification filter (Level 2+) | done | medium | `pipeline/llm.py` — GPT-4.1-mini, fail-open |
| E-011 | LLM: daily digest summarization | done | medium | `pipeline/summarizer.py` — evening digest at 20:00 ICT |
| E-012 | Digest mode: periodic chat summaries | done | low | Scheduler in main.py, summary via dispatcher |
| E-015 | Deploy: systemd unit for Linux server | done | low | `deploy/` — install.sh, deploy.sh, eidolon.service |
| E-016 | Deploy Eidolon to Hostinger VPS | done | high | Live on srv1327676, systemd user service |
| E-017 | Create Eidolon bot via @BotFather | done | high | @EidolonSpyBot |
