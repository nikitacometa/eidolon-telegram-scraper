-- Eidolon database schema

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_msg_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    chat_title TEXT,
    sender_id INTEGER,
    sender_name TEXT,
    text TEXT,
    date TIMESTAMP NOT NULL,
    raw_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id, telegram_msg_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watcher_name TEXT NOT NULL,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    filter_level INTEGER NOT NULL,  -- 1=rule, 2=embedding, 3=llm
    score REAL,
    llm_response TEXT,
    matched_keyword TEXT,
    delivery_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(delivery_status IN ('pending', 'sent', 'failed')),
    delivery_attempts INTEGER NOT NULL DEFAULT 0
        CHECK(delivery_attempts >= 0),
    next_attempt_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_error TEXT,
    claimed_until TIMESTAMP,
    lease_owner TEXT,
    sent_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(watcher_name, message_id)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES messages(id),
    watcher_name TEXT NOT NULL,
    watcher_config_fingerprint TEXT NOT NULL DEFAULT 'legacy-unknown',
    processing_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(processing_status IN ('pending', 'completed', 'failed')),
    rule_passed INTEGER NOT NULL DEFAULT 0,
    embedding_status TEXT NOT NULL DEFAULT 'skipped',
    embedding_passed INTEGER NOT NULL DEFAULT 0,
    embedding_score REAL,
    embedding_negative_score REAL,
    embedding_threshold REAL,
    embedding_model TEXT,
    embedding_latency_ms REAL,
    embedding_input_tokens INTEGER NOT NULL DEFAULT 0,
    llm_status TEXT NOT NULL DEFAULT 'skipped',
    llm_relevant INTEGER NOT NULL DEFAULT 0,
    llm_passed INTEGER NOT NULL DEFAULT 0,
    llm_verdict TEXT,
    llm_confidence REAL,
    llm_model TEXT,
    llm_prompt_version TEXT,
    llm_latency_ms REAL,
    llm_input_tokens INTEGER NOT NULL DEFAULT 0,
    llm_output_tokens INTEGER NOT NULL DEFAULT 0,
    accepted INTEGER NOT NULL DEFAULT 0,
    alert_created INTEGER NOT NULL DEFAULT 0,
    alert_sent INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    UNIQUE(message_id, watcher_name)
);

CREATE TABLE IF NOT EXISTS chats (
    chat_id INTEGER PRIMARY KEY,
    title TEXT,
    type TEXT,  -- group, supergroup, channel
    joined_at TIMESTAMP,
    last_message_at TIMESTAMP,
    message_count INTEGER DEFAULT 0
);

-- Which chats the daemon observes, and how. Chat membership is runtime state
-- that reconnaissance changes on its own, so it lives here rather than in the
-- git-tracked policy file: promoting a discovered chat is a row, not a rewrite
-- of a configuration file that the process cannot even write to under
-- ProtectSystem=strict.
CREATE TABLE IF NOT EXISTS observed_chats (
    chat_id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'monitor'
        CHECK(mode IN ('monitor', 'recon', 'paused')),
    title TEXT,
    source TEXT NOT NULL DEFAULT 'config'
        CHECK(source IN ('config', 'recon', 'manual')),
    job_id TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Which policies apply to an observed chat. A reconnaissance chat has none
-- until it is promoted; a monitored chat may have several.
CREATE TABLE IF NOT EXISTS chat_policy_bindings (
    chat_id INTEGER NOT NULL REFERENCES observed_chats(chat_id) ON DELETE CASCADE,
    watcher_name TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'config'
        CHECK(source IN ('config', 'recon', 'manual')),
    job_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, watcher_name)
);

CREATE TABLE IF NOT EXISTS filter_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watcher_name TEXT NOT NULL,
    date DATE NOT NULL,
    messages_total INTEGER DEFAULT 0,
    passed_level1 INTEGER DEFAULT 0,
    passed_level2 INTEGER DEFAULT 0,
    passed_level3 INTEGER DEFAULT 0,
    accepted INTEGER DEFAULT 0,
    alerts_sent INTEGER DEFAULT 0,
    UNIQUE(watcher_name, date)
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_date ON messages(chat_id, date);
CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date);
CREATE INDEX IF NOT EXISTS idx_alerts_watcher ON alerts(watcher_name, created_at);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_watcher
    ON pipeline_runs(watcher_name, created_at);
CREATE INDEX IF NOT EXISTS idx_bindings_watcher ON chat_policy_bindings(watcher_name);

-- Watchers authored by the assistant rather than by a person editing YAML.
--
-- Deliberately a separate table rather than an overlay on the git-tracked
-- policy file. Policy under review by a human and policy created from a chat
-- message are different things with different trust, and merging them field by
-- field was the complexity the original design walked away from. Here the merge
-- is a union over disjoint name spaces: agent names carry an `agent-` prefix,
-- a collision with a config name is refused, and config always wins.
CREATE TABLE IF NOT EXISTS agent_watchers (
    name TEXT PRIMARY KEY,
    -- Watcher.model_dump_json(mode='json'), with chats forced empty: which
    -- chats a policy watches is runtime state that lives in observed_chats.
    definition TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'revoked')),
    -- Who asked for it. Julia's bridge is read-only and must never be given the
    -- tool that writes here; this column is what makes it visible if it ever is.
    created_by TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- A single counter bumped in the same transaction as every write above, so the
-- daemon can notice a change with one scalar read per tick instead of scanning
-- the table. A timestamp would not do: two edits inside the same second are
-- indistinguishable, and the poller would miss the second one.
CREATE TABLE IF NOT EXISTS agent_watchers_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    generation INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO agent_watchers_meta (id, generation) VALUES (1, 0);
