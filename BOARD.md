# Eidolon — Task Board

> Last updated: 2026-03-04

## Conventions

- **ID format**: `E-NNN` (sequential, never reuse)
- **Statuses**: `todo` | `in_progress` | `blocked` | `done`
- **Priorities**: `critical` | `high` | `medium` | `low`
- Next available ID: **E-016**

---

## Active

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-008 | First watcher: Phangan housing monitor | todo | high | Need real chat IDs, test with live messages |

## Backlog

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-009 | Pipeline: embedding similarity filter (Level 2) | todo | medium | `pipeline/embeddings.py` — ChromaDB + text-embedding-3-small |
| E-010 | Pipeline: LLM classification filter (Level 3) | todo | medium | `pipeline/llm.py` — GPT-4.1-mini relevance check |
| E-011 | LLM: structured extraction + summarization | todo | medium | GPT-4.1 for deep analysis |
| E-012 | Digest mode: periodic chat summaries | todo | low | Cron-like scheduler, batch alerts every N hours |
| E-013 | Monitoring: filter stats dashboard | todo | low | Track pass/drop rates per watcher per level |
| E-014 | Deploy: launchd plist for macOS daemon | todo | low | Auto-start, restart on crash, log rotation |
| E-015 | Deploy: systemd unit for Linux server | todo | low | Alternative deployment on Hostinger VPS |

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
