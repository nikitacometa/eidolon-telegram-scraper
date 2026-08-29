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
    reply_to_message_id INTEGER,
    reply_backfill_checked INTEGER NOT NULL DEFAULT 0
        CHECK(reply_backfill_checked IN (0, 1)),
    -- Album membership: several Telegram messages carrying one advertisement.
    grouped_id INTEGER,
    has_media INTEGER NOT NULL DEFAULT 0 CHECK(has_media IN (0, 1)),
    telegram_photo_id INTEGER,
    -- 0 means "ingested before media pointers existed", not "no media": see
    -- the migration in storage/db.py for why the distinction is load-bearing.
    media_scan_version INTEGER NOT NULL DEFAULT 0,
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
-- idx_messages_grouped is created by the migration in storage/db.py, not here:
-- this script runs before the ALTER TABLEs, so on a database that predates the
-- media columns an index over grouped_id would fail on a column that does not
-- exist yet and take the whole connect() down with it.
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

-- ---------------------------------------------------------------------------
-- Housing: the rental-search subsystem.
--
-- Deliberately its own tables rather than an overlay on `alerts`/`pipeline_runs`.
-- A housing listing is not one message with one verdict: it is a content unit
-- (an album is several Telegram messages carrying one advertisement), its facts
-- arrive on two timelines (text now, photographs later), and every fact has a
-- third state -- unknown -- that must never collapse into "does not match".
-- `claim_due_alerts` joins alerts to a single message row, which a unit-level
-- alert has no single one of, so housing carries its own outbox as well.
-- ---------------------------------------------------------------------------

-- One advertisement, as opposed to one Telegram message. Album members share a
-- `grouped_id` and arrive as separate events milliseconds apart, so a unit is
-- held open for a short quiet window and finalized by a sweep that reads this
-- table -- not by an in-process timer, which a restart would lose.
CREATE TABLE IF NOT EXISTS housing_live_units (
    -- 'g:<chat_id>:<grouped_id>' for an album, 'm:<chat_id>:<message_id>' otherwise.
    unit_key TEXT PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    grouped_id INTEGER,
    -- Bumped when an edit changes the text, so a later extraction cannot be
    -- written against a version of the advertisement that no longer exists.
    unit_version INTEGER NOT NULL DEFAULT 1,
    representative_message_id INTEGER NOT NULL,
    assembled_text TEXT,
    media_count INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'assembling'
        CHECK(state IN ('assembling', 'ready', 'extracting', 'extracted',
                        'matching', 'done', 'error')),
    settle_after TIMESTAMP NOT NULL,
    last_error TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_housing_units_pending
    ON housing_live_units(state, settle_after);

CREATE TABLE IF NOT EXISTS housing_live_unit_messages (
    unit_key TEXT NOT NULL REFERENCES housing_live_units(unit_key) ON DELETE CASCADE,
    message_id INTEGER NOT NULL,
    telegram_msg_id INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    has_media INTEGER NOT NULL DEFAULT 0 CHECK(has_media IN (0, 1)),
    telegram_photo_id INTEGER,
    has_text INTEGER NOT NULL DEFAULT 0 CHECK(has_text IN (0, 1)),
    PRIMARY KEY (unit_key, message_id)
);

-- What the extractors believe about a unit. Every field carries where it came
-- from, because "the text said two bathrooms" and "a photograph showed two
-- bathrooms" are claims of different strength, and "nobody said" is a third
-- answer that has to survive all the way to the alert.
CREATE TABLE IF NOT EXISTS housing_live_facts (
    unit_key TEXT PRIMARY KEY REFERENCES housing_live_units(unit_key) ON DELETE CASCADE,
    unit_version INTEGER NOT NULL,
    is_rental_offer INTEGER,
    is_vehicle_ad INTEGER,
    bedrooms INTEGER,
    bedrooms_source TEXT CHECK(bedrooms_source IN ('text', 'vision', 'unknown')),
    bathrooms INTEGER,
    bathrooms_source TEXT CHECK(bathrooms_source IN ('text', 'vision', 'unknown')),
    monthly_price_thb INTEGER,
    price_source TEXT CHECK(price_source IN ('text', 'unknown')),
    tv_present INTEGER,
    tv_size_class TEXT CHECK(tv_size_class IN ('none', 'small', 'medium', 'large', 'unclear')),
    tv_source TEXT CHECK(tv_source IN ('text', 'vision', 'unknown')),
    area_raw TEXT,
    evidence_quote TEXT,
    vision_status TEXT NOT NULL DEFAULT 'not_attempted'
        CHECK(vision_status IN ('not_attempted', 'pending', 'done', 'unavailable', 'error')),
    extractor_version TEXT NOT NULL,
    extracted_at TIMESTAMP
);

-- What the owner is looking for. Append-only revisions plus one active pointer:
-- an edit is a new row, never an overwrite, so an alert can always name the
-- requirements it was judged against even after they change.
CREATE TABLE IF NOT EXISTS housing_requirements_revisions (
    revision INTEGER PRIMARY KEY,
    definition_json TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS housing_requirements_active (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    active_revision INTEGER NOT NULL
        REFERENCES housing_requirements_revisions(revision),
    -- Same counter trick as agent_watchers_meta: the daemon notices an edit
    -- with one scalar read per tick, and two edits in the same second are
    -- still two distinct generations.
    generation INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS housing_matches (
    unit_key TEXT NOT NULL REFERENCES housing_live_units(unit_key) ON DELETE CASCADE,
    requirements_revision INTEGER NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('confirmed', 'possible', 'hard_miss')),
    field_verdicts_json TEXT NOT NULL,
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (unit_key, requirements_revision)
);

CREATE TABLE IF NOT EXISTS housing_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_key TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    chat_title TEXT,
    telegram_msg_id INTEGER NOT NULL,
    requirements_revision INTEGER NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('confirmed', 'possible')),
    kind TEXT NOT NULL DEFAULT 'live' CHECK(kind IN ('live', 'update', 'digest')),
    body_html TEXT NOT NULL,
    photo_paths_json TEXT,
    delivery_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(delivery_status IN ('pending', 'delivered', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    claimed_until TIMESTAMP,
    lease_owner TEXT,
    last_error TEXT,
    next_attempt_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMP
);

-- One verdict per unit is delivered once. An upgrade from possible to
-- confirmed is a different verdict, so it is a new row rather than a
-- suppressed duplicate.
CREATE UNIQUE INDEX IF NOT EXISTS idx_housing_alerts_dedup
    ON housing_alerts(unit_key, verdict, kind);
CREATE INDEX IF NOT EXISTS idx_housing_alerts_due
    ON housing_alerts(delivery_status, next_attempt_at);

-- Photographs worth fetching, and what became of the attempt.
--
-- This table lives beside the rest of housing rather than in the scout
-- archive because it is part of one advertisement's life, and it deliberately
-- outlives the 30-day sweep over `messages`: the file on disk and the vision
-- answer derived from it stay useful long after the message row is gone.
CREATE TABLE IF NOT EXISTS housing_media (
    unit_key TEXT NOT NULL REFERENCES housing_live_units(unit_key) ON DELETE CASCADE,
    chat_id INTEGER NOT NULL,
    telegram_msg_id INTEGER NOT NULL,
    telegram_photo_id INTEGER NOT NULL,
    priority TEXT NOT NULL DEFAULT 'live' CHECK(priority IN ('live', 'backfill')),
    download_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(download_status IN ('pending', 'downloaded', 'failed_gone', 'failed')),
    local_path TEXT,
    byte_size INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    not_before TIMESTAMP,
    last_error TEXT,
    requested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    downloaded_at TIMESTAMP,
    PRIMARY KEY (unit_key, telegram_msg_id)
);

CREATE INDEX IF NOT EXISTS idx_housing_media_pending
    ON housing_media(download_status, priority, requested_at);
-- Telegram's photo id is stable across reposts and chats, so the same
-- photograph crossposted into three chats is fetched once. The index is
-- partial because only a completed download has a file to reuse.
CREATE INDEX IF NOT EXISTS idx_housing_media_photo
    ON housing_media(telegram_photo_id) WHERE download_status = 'downloaded';

-- What kind of chat a housing-bound chat is.
--
-- A dedicated rentals board and the island's general talk chat need different
-- treatment: in the first, every message is a candidate and is read in full;
-- in the second, listings are one message in five and a cheap lexical gate
-- keeps the model bill proportional to listings rather than to conversation.
-- Absence from this table means general, which is the cautious default only
-- in the sense that it spends less — a chat wrongly marked general still sees
-- every message that mentions housing or a price.
CREATE TABLE IF NOT EXISTS housing_chat_kinds (
    chat_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'general_island'
        CHECK(kind IN ('dedicated_housing', 'general_island')),
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
