-- Eidolon reconnaissance (scout) schema.
--
-- Physically separate from the live monitoring database on purpose. The live
-- Database holds one aiosqlite connection behind one process-wide asyncio
-- lock, so every crawl write would otherwise queue in front of live message
-- ingestion and alert delivery. A second file gets its own connection and its
-- own lock, which makes that contention structurally impossible rather than
-- merely unlikely.

CREATE TABLE IF NOT EXISTS recon_jobs (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    account_id TEXT NOT NULL DEFAULT 'owner-primary',
    topic TEXT NOT NULL,
    location TEXT,
    languages TEXT NOT NULL DEFAULT '[]',
    seeds TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK(status IN (
            'queued', 'discovering', 'waiting_approval', 'joining', 'backfilling',
            'analyzing', 'paused_rate_limit', 'safety_halt', 'completed',
            'completed_partial', 'failed', 'cancelled'
        )),
    current_wave INTEGER NOT NULL DEFAULT 0,
    max_waves INTEGER NOT NULL DEFAULT 2,
    lookback_days INTEGER NOT NULL DEFAULT 30,
    max_candidates INTEGER NOT NULL DEFAULT 100,
    max_join_attempts INTEGER NOT NULL DEFAULT 6,
    stop_reason TEXT,
    error_code TEXT,
    resume_at TIMESTAMP,
    deadline_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Canonical chat identity. A chat is one row no matter how many usernames,
-- invite links, or mentions pointed at it.
CREATE TABLE IF NOT EXISTS scout_chats (
    id TEXT PRIMARY KEY,
    telegram_chat_id INTEGER UNIQUE,
    username TEXT,
    title TEXT,
    chat_type TEXT,
    participants INTEGER,
    -- Telegram's own scam/fake labels. Scoring happens after the chat is
    -- stored, so dropping these here would silently disarm the risk check.
    risk_flags TEXT NOT NULL DEFAULT '[]',
    visibility TEXT NOT NULL DEFAULT 'unknown'
        CHECK(visibility IN ('unknown', 'public', 'private')),
    visibility_source TEXT,
    visibility_checked_at TIMESTAMP,
    membership TEXT NOT NULL DEFAULT 'none'
        CHECK(membership IN ('none', 'requested', 'member', 'left', 'failed')),
    joined_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Every locator that ever pointed at a chat. Invite hashes are stored as
-- fingerprints, never raw: a raw hash is a working key to a private chat.
CREATE TABLE IF NOT EXISTS chat_aliases (
    alias_kind TEXT NOT NULL CHECK(alias_kind IN ('peer_id', 'username', 'invite')),
    alias_value TEXT NOT NULL,
    chat_uuid TEXT NOT NULL REFERENCES scout_chats(id) ON DELETE CASCADE,
    first_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (alias_kind, alias_value)
);

CREATE TABLE IF NOT EXISTS job_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES recon_jobs(id) ON DELETE CASCADE,
    chat_uuid TEXT NOT NULL REFERENCES scout_chats(id) ON DELETE CASCADE,
    wave INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'discovered'
        CHECK(state IN (
            'discovered', 'scored', 'blocked_private', 'rejected', 'awaiting_approval',
            'approved', 'join_requested', 'joined', 'join_failed'
        )),
    policy_score REAL,
    llm_score REAL,
    independent_sources INTEGER NOT NULL DEFAULT 0,
    risk_flags TEXT NOT NULL DEFAULT '[]',
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id, chat_uuid)
);

-- One row per independent signal. The unique key is (candidate, source,
-- origin), so a single actor reposting the same message into two crawled
-- chats yields one signal, not two.
CREATE TABLE IF NOT EXISTS candidate_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES job_candidates(id) ON DELETE CASCADE,
    source TEXT NOT NULL
        CHECK(source IN (
            'owner_seed', 'hashtag_search', 'fulltext_search', 'contacts_search',
            'global_search', 'recommendations', 'message_link', 'message_mention', 'forward'
        )),
    origin_key TEXT NOT NULL,
    source_chat_id INTEGER,
    source_message_id INTEGER,
    snippet TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_id, source, origin_key)
);

CREATE TABLE IF NOT EXISTS recon_frontier (
    candidate_id INTEGER PRIMARY KEY REFERENCES job_candidates(id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES recon_jobs(id) ON DELETE CASCADE,
    priority REAL NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK(state IN ('pending', 'claimed', 'done', 'skipped')),
    attempts INTEGER NOT NULL DEFAULT 0,
    not_before TIMESTAMP,
    claimed_until TIMESTAMP,
    lease_owner TEXT,
    last_error TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- The action ledger is also the budget: rolling windows are counted from it,
-- so there is no separate counter that can drift from what actually happened.
-- duration_ms is recorded because Telethon sleeps through short FloodWaits
-- inside the call, making an unexpectedly slow call the only visible trace.
CREATE TABLE IF NOT EXISTS telegram_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    kind TEXT NOT NULL
        CHECK(kind IN (
            'join', 'hashtag_search', 'fulltext_search', 'contacts_search', 'global_search',
            'recommendations', 'resolve_username', 'invite_check', 'history_page',
            'media_download_live', 'media_download_backfill'
        )),
    idempotency_key TEXT NOT NULL UNIQUE,
    job_id TEXT REFERENCES recon_jobs(id) ON DELETE SET NULL,
    candidate_id INTEGER REFERENCES job_candidates(id) ON DELETE SET NULL,
    outcome TEXT NOT NULL DEFAULT 'reserved'
        CHECK(outcome IN ('reserved', 'succeeded', 'failed', 'flood_wait', 'ambiguous')),
    flood_wait_seconds INTEGER,
    error_code TEXT,
    duration_ms REAL,
    reserved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS account_cooldowns (
    account_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    blocked_until TIMESTAMP NOT NULL,
    reason TEXT NOT NULL,
    manual_resume_required INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (account_id, scope)
);

-- Messages captured from chats under reconnaissance, both live and
-- backfilled. Kept apart from the monitoring `messages` table: this is bulk
-- crawl material that must never queue behind live alert delivery, and it
-- carries no alerts, no pipeline runs, and no LLM cost.
CREATE TABLE IF NOT EXISTS scout_messages (
    chat_id INTEGER NOT NULL,
    telegram_msg_id INTEGER NOT NULL,
    sender_id INTEGER,
    sender_name TEXT,
    text TEXT,
    date TIMESTAMP NOT NULL,
    entities_json TEXT,
    forward_chat_id INTEGER,
    forward_message_id INTEGER,
    reply_to_message_id INTEGER,
    content_hash TEXT,
    source TEXT NOT NULL DEFAULT 'live' CHECK(source IN ('live', 'backfill')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, telegram_msg_id)
);

-- Chats whose history is being walked backwards in the background. A target
-- is a standing intent measured in years, worked one page at a time whenever
-- the daemon has nothing better to do, so a restart resumes from the cursor
-- rather than starting the archive again.
CREATE TABLE IF NOT EXISTS backfill_targets (
    chat_id INTEGER PRIMARY KEY,
    label TEXT,
    target_days INTEGER NOT NULL DEFAULT 730,
    oldest_message_id INTEGER,
    oldest_message_date TIMESTAMP,
    messages_stored INTEGER NOT NULL DEFAULT 0,
    pages_done INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK(state IN ('pending', 'complete', 'exhausted', 'failed')),
    last_error TEXT,
    not_before TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Chats queued to be joined, worked one at a time. Joining is the only
-- reconnaissance action other people can see, and the account's standing is
-- spent per attempt, so the queue exists to spread attempts over hours
-- instead of letting a batch land at once.
CREATE TABLE IF NOT EXISTS join_queue (
    chat_ref TEXT PRIMARY KEY,
    label TEXT,
    watcher_name TEXT,
    target_days INTEGER NOT NULL DEFAULT 730,
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK(state IN ('pending', 'joined', 'requested', 'failed', 'skipped')),
    attempts INTEGER NOT NULL DEFAULT 0,
    not_before TIMESTAMP,
    last_error TEXT,
    joined_chat_id INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_actions_budget
    ON telegram_actions(account_id, kind, reserved_at);
CREATE INDEX IF NOT EXISTS idx_backfill_due ON backfill_targets(state, not_before, pages_done);
CREATE INDEX IF NOT EXISTS idx_join_queue_due ON join_queue(state, not_before);
CREATE INDEX IF NOT EXISTS idx_scout_messages_chat_date ON scout_messages(chat_id, date);
CREATE INDEX IF NOT EXISTS idx_scout_messages_forward
    ON scout_messages(forward_chat_id, forward_message_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON recon_jobs(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_candidates_job_state ON job_candidates(job_id, state);
CREATE INDEX IF NOT EXISTS idx_frontier_claimable
    ON recon_frontier(job_id, state, priority DESC, not_before);
CREATE INDEX IF NOT EXISTS idx_aliases_chat ON chat_aliases(chat_uuid);
