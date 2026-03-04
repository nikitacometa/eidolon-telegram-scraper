# Eidolon — Task Board

> Last updated: 2026-03-04

## Conventions

- **ID format**: `E-NNN` (sequential, never reuse)
- **Statuses**: `todo` | `in_progress` | `blocked` | `done`
- **Priorities**: `critical` | `high` | `medium` | `low`
- Next available ID: **E-006**

---

## Active

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-001 | Set up Telegram auth flow and session generation | todo | critical | Telethon StringSession, .env storage |
| E-002 | Implement message ingestion pipeline | todo | critical | Telethon event handler → SQLite storage |
| E-003 | Build rule-based keyword filter (Level 1) | todo | high | watchers.yml config, regex + keyword matching |
| E-004 | Implement alert dispatcher via Pantheon bot | todo | high | Reuse @ClaudePantheon_Bot for notifications |
| E-005 | Phangan housing watcher (first real use case) | todo | high | Monitor Phangan chats for rental posts |

## Backlog

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|

## Done

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
