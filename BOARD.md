# Eidolon — Task Board

> Last updated: 2026-07-17

## Conventions

- **ID format**: `E-NNN` (sequential, never reuse)
- **Statuses**: `todo` | `in_progress` | `blocked` | `done`
- **Priorities**: `critical` | `high` | `medium` | `low`
- Next available ID: **E-032**

---

## Active

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-018 | Live test: summary digest at 20:00 ICT | todo | medium | Verify daily summary arrives tonight |
| E-026 | Harden correctness, security, and public configuration | in_progress | critical | Audit milestone 1 |
| E-027 | Add typed AI decisions and provider boundaries | todo | high | Structured outputs and provenance |
| E-028 | Build Evaluation Lab and regression dataset | todo | high | Precision/recall/F1, latency, cost |
| E-029 | Add durable processing, outbox, and bounded workers | todo | high | Idempotency and crash recovery |
| E-030 | Add FastAPI control plane, Docker, lockfile, and CI | todo | high | Reproducible production signal |
| E-031 | Rebrand repository and publish portfolio-grade README | todo | high | `eidolon-telegram-scraper` |

## Backlog — Phase 3: Autonomous Agent

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-019 | Entity extraction from L3-passed messages | todo | high | `pipeline/entities.py` — LLM extracts persons, properties, contacts, locations into SQLite |
| E-020 | Contact tracking and relevance scoring | todo | high | `contacts` table, auto-populate from alerts, increment `relevance_score` |
| E-021 | Cognitive loop (read-only, insights only) | todo | high | `pipeline/cognitive.py` — OBSERVE→THINK→PLAN→REFLECT, cron or threshold-triggered, sends insights via bot |
| E-022 | Group discovery with approval workflow | todo | medium | `pipeline/discovery.py` — `SearchGlobalRequest`, LLM relevance scoring, inline keyboard approve/skip |
| E-023 | Outreach pipeline (strictly gated) | todo | medium | `pipeline/outreach.py` — double approval, rate limits (5 DM/day, 3 joins/day), FloodWait budget |
| E-024 | Schema v2: entities, contacts, actions tables | todo | high | `storage/schema_v2.sql` — entities, entity_mentions, entity_relations, contacts, actions |
| E-025 | Persona config and disclosure system | todo | low | `config/persona.yml` — style per chat type, AI disclosure in first DM |

## Backlog — Other

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
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
| E-009 | Pipeline: embedding similarity filter (Level 2) | done | medium | `pipeline/embeddings.py` — ChromaDB + text-embedding-3-small |
| E-010 | Pipeline: LLM classification filter (Level 3) | done | medium | `pipeline/llm.py` — GPT-4.1-mini, fail-open |
| E-011 | LLM: daily digest summarization | done | medium | `pipeline/summarizer.py` — evening digest at 20:00 ICT |
| E-012 | Digest mode: periodic chat summaries | done | low | Scheduler in main.py, summary via dispatcher |
| E-015 | Deploy: systemd unit for Linux server | done | low | `deploy/` — install.sh, deploy.sh, eidolon.service |
| E-016 | Deploy Eidolon to Hostinger VPS | done | high | Live on srv1327676, systemd user service |
| E-017 | Create Eidolon bot via @BotFather | done | high | @EidolonSpyBot |
