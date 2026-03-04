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
| E-001 | Telegram auth: session string generator | todo | critical | `auth.py` — Telethon StringSession, saves to .env |
| E-002 | Core: Telethon client + event loop | todo | critical | `main.py` — connect, listen NewMessage, graceful shutdown |
| E-003 | Storage: SQLite wrapper + migrations | todo | critical | `storage/db.py` — aiosqlite, run schema.sql on init |
| E-004 | Pipeline: message ingestion handler | todo | critical | `pipeline/ingestion.py` — extract msg fields, store in DB |
| E-005 | Pipeline: rule-based keyword filter (Level 1) | todo | high | `pipeline/filters.py` — watchers.yml rules, regex, keywords |
| E-006 | Pipeline: alert dispatcher via Pantheon bot | todo | high | `pipeline/dispatcher.py` — send alerts through @ClaudePantheon_Bot |
| E-007 | Integration: wire pipeline into main event loop | todo | high | message → ingestion → filter → dispatch |
| E-008 | First watcher: Phangan housing monitor | todo | high | Real chat IDs, test with live messages |

## Backlog

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-009 | Pipeline: embedding similarity filter (Level 2) | todo | medium | `pipeline/embeddings.py` — ChromaDB + text-embedding-3-small |
| E-010 | Pipeline: LLM classification filter (Level 3) | todo | medium | `pipeline/llm.py` — Claude Haiku relevance check |
| E-011 | LLM: structured extraction + summarization | todo | medium | Claude Sonnet via Batch API for deep analysis |
| E-012 | Digest mode: periodic chat summaries | todo | low | Cron-like scheduler, batch alerts every N hours |
| E-013 | Monitoring: filter stats dashboard | todo | low | Track pass/drop rates per watcher per level |
| E-014 | Deploy: launchd plist for macOS daemon | todo | low | Auto-start, restart on crash, log rotation |
| E-015 | Deploy: systemd unit for Linux server | todo | low | Alternative deployment on Hostinger VPS |

## Done

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
