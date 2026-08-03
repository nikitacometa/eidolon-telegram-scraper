# Eidolon — Task Board

> Last updated: 2026-08-03

## Conventions

- **ID format**: `E-NNN` (sequential, never reuse)
- **Statuses**: `todo` | `in_progress` | `blocked` | `done` | `superseded`
- **Priorities**: `critical` | `high` | `medium` | `low`
- Next available ID: **E-053**

---

## Active

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-036 | Split monitoring policy from chat binding | done | critical | `observed_chats` registry in DB; `Watcher.chats` becomes an optional seed; ingest routes by observation mode |
| E-018 | Live test: summary digest at 20:00 ICT | todo | medium | Verify daily summary arrives tonight |
| E-048 | Searchable corpus over both message stores | done | critical | `eidolon_search.db`: union of `messages` + `scout_messages`, FTS5 (unicode61 + prefix) and 512-dim embeddings, crosspost rollup. 21,533 messages, refreshed by `eidolon-index.timer` every 10 min |
| E-049 | Fix the zero-alert cascade | done | critical | Measured: L1 dropped 56% of real event announcements and L2 rejected the rest against bare-keyword references. Now L1 is a negative-only gate and L2 has 25 sentence examples at threshold 0.34 — 90.6% end-to-end recall on a held-out set of 64 real announcements, ~16 L3 calls/day |
| E-050 | Rebind the Phangan chats or retire them | todo | high | Telethon still receives updates for `Близкие на Пангане` (2776 msgs) and `Подслушано Панган`; neither is in `observed_chats`, so ingest has silently dropped them since the 2026-07-29 restart |
| E-051 | Run discovery for the first time | todo | high | `telegram_actions` has only `join` and `history_page` — zero `hashtag_search`, `recommendations` or `contacts_search` ever. All 14 joins came from a hand-written seed list, and the queue has been empty since 2026-08-02 |
| E-052 | Fix `ReconRunner` history storage | todo | high | Pre-existing on main: `test_a_topic_becomes_joined_chats_with_history` and `test_links_in_history_become_next_wave_candidates` fail — the runner joins but stores 0 messages, so snowball never seeds the next wave. `BackfillWorker` is unaffected |

## Backlog — Phase 4: Reconnaissance (design: `tmp/design-openclaw-recon.md`)

Scope A "Scout" is E-035 to E-040: discovery and reporting with no join at all.
Scope B "Recon" adds E-041 to E-045.

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-035 | Scout store: jobs, candidates, evidence, frontier, action budgets | done | critical | `storage/scout.py` + `scout_schema.sql`, own connection and lock — 49 tests |
| E-037 | Probe: SQLite contention under crawl load | todo | high | Baseline p99 of `claim_due_alerts`, then synthetic scout writes; accept ≤1.2× baseline |
| E-038 | Probe: MCP spawn and hot-reload on the live gateway | done | high | Answered against OpenClaw 2026.7.1: config key is `mcp.servers`, not `tools.mcp`; `mcp add` probes before saving; `mcp reload` applies without a gateway restart; stdio and streamable-http both supported |
| E-039 | Telegram action governor | done | critical | `pipeline/governor.py` — single gateway for MTProto calls, reserves budget, wall-clock timing, FloodWait ladder |
| E-040 | Discovery sources and deterministic scoring | done | critical | `channels.searchPosts` (hashtag first), recommendations, `contacts.search`; public-scope hard gate before any LLM call |
| E-041 | Command API over a unix socket | superseded | high | The join queue is already a durable DB-polled command channel; the bridge appends to it and the daemon works it on its own budget. A second RPC surface bought nothing |
| E-042 | MCP stdio bridge and OpenClaw skill | done | high | `eidolon_mcp.py`, stateless, index opened read-only, no Telethon. 7 tools on Nikita's instance, 6 read-only on Julia's over an SSH forced command. Skill in `~/.openclaw/workspace/skills/eidolon/` on both |
| E-043 | Approval flow on inline buttons | todo | high | `getUpdates` consumer on @EidolonSpyBot, persistent offset, owner-only callbacks, batched digests |
| E-044 | Safe join with membership reconciliation | todo | critical | `INVITE_REQUEST_SENT` is not membership; no blind retry after an ambiguous crash |
| E-045 | Two-wave snowball with history backfill | done | high | Runner joins, backfills pages, follows links; `recon_cli.py` runs a job end to end |
| E-047 | Enforce one process per Telegram session | done | critical | `storage/session_lock.py`; the CLI refuses to start while the daemon holds it |
| E-046 | Coordinated backups for both databases | todo | high | `sqlite3 .backup` timer; today nothing is backed up at all |

## Backlog — Phase 3: Autonomous Agent

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-019 | Entity extraction from L3-passed messages | done | high | `pipeline/indexer.py` extracts venues into `places`/`place_mentions` with an evidence quote, over the WHOLE corpus rather than only L3 survivors; ASCII-folded aliases so stylized branding is findable |
| E-020 | Contact tracking and relevance scoring | todo | high | `contacts` table, auto-populate from alerts, increment `relevance_score` |
| E-021 | Cognitive loop (read-only, insights only) | todo | high | `pipeline/cognitive.py` — OBSERVE→THINK→PLAN→REFLECT, cron or threshold-triggered, sends insights via bot |
| E-022 | Group discovery with approval workflow | superseded | medium | Replaced by E-039/E-040/E-043 with rolling budgets and a public-scope gate |
| E-023 | Outreach pipeline (strictly gated) | todo | medium | `pipeline/outreach.py` — double approval, rate limits (5 DM/day, 3 joins/day), FloodWait budget |
| E-024 | Schema v2: entities, contacts, actions tables | superseded | high | Action ledger now lives in `storage/scout_schema.sql` |
| E-025 | Persona config and disclosure system | todo | low | `config/persona.yml` — style per chat type, AI disclosure in first DM |

## Backlog — Other

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| E-013 | Monitoring: filter stats dashboard | todo | low | Track pass/drop rates per watcher per level |
| E-032 | Improve blind-set recall with uncertainty routing | todo | high | Promote v3 misses to dev; evaluate on a new blind corpus |
| E-033 | Add durable digest chunk delivery | todo | medium | Resume partial summaries through the outbox |

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
| E-010 | Pipeline: LLM classification filter (Level 3) | done | medium | `pipeline/llm.py` — typed GPT-4.1-mini decision, safe degraded default |
| E-011 | LLM: daily digest summarization | done | medium | `pipeline/summarizer.py` — evening digest at 20:00 ICT |
| E-012 | Digest mode: periodic chat summaries | done | low | Scheduler in main.py, summary via dispatcher |
| E-015 | Deploy: systemd unit for Linux server | done | low | `deploy/` — install.sh, deploy.sh, eidolon.service |
| E-016 | Deploy Eidolon to Linux VPS | done | high | Live via systemd user service |
| E-017 | Create Eidolon bot via @BotFather | done | high | Dedicated alert bot configured |
| E-026 | Harden correctness, security, and public configuration | done | critical | Regression-covered audit milestone 1 |
| E-027 | Add typed AI decisions and provider boundaries | done | high | Structured outputs and provenance |
| E-028 | Build Evaluation Lab and regression dataset | done | high | Versioned 20-case baseline, F1 0.9412 |
| E-029 | Add durable processing, outbox, and bounded workers | done | high | Atomic ingress/outcome/delivery and crash recovery |
| E-030 | Add FastAPI control plane, Docker, lockfile, and CI | done | high | Reproducible production signal |
| E-031 | Rebrand repository and publish portfolio-grade README | done | high | `eidolon-telegram-scraper`; GitHub CI passed |
| E-034 | Harden degraded AI policy, digest grounding, and lease fencing | done | critical | Safe defaults, validated citations, stale-worker protection |
