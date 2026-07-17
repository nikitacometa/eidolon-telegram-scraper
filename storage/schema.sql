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
