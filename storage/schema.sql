-- Eidolon database schema

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_msg_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
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
    message_id INTEGER REFERENCES messages(id),
    filter_level INTEGER NOT NULL,  -- 1=rule, 2=embedding, 3=llm
    score REAL,
    llm_response TEXT,
    sent_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    alerts_sent INTEGER DEFAULT 0,
    UNIQUE(watcher_name, date)
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_date ON messages(chat_id, date);
CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date);
CREATE INDEX IF NOT EXISTS idx_alerts_watcher ON alerts(watcher_name, created_at);
