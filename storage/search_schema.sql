-- Derived search index over both message stores.
--
-- This file is a projection, never a source of truth: every row here is
-- rebuildable from data/eidolon.db and data/eidolon_scout.db, and deleting
-- the whole file costs nothing but the time to re-sync and re-embed.
--
-- It is a third file rather than a table inside either source database for a
-- mechanical reason: SQLite refuses to let an external-content FTS5 table or a
-- persisted view reference a table in a different attached schema, so a single
-- searchable union of two files cannot be indexed in place. Writing the
-- projection here is the only arrangement SQLite permits.
--
-- The second reason is safety. A trigger that throws inside the daemon's write
-- path rolls back the INSERT that fired it, and the ingest handler treats a
-- failed store as fatal, so an FTS5 sync trigger placed on `messages` or
-- `scout_messages` would drop the message AND stop live monitoring. Every
-- trigger below fires on `corpus_messages`, which no daemon writes to.

CREATE TABLE IF NOT EXISTS corpus_messages (
    corpus_id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Which store the row was first seen in. A message present in both is
    -- stored once; `source` records where the sync job saw it first.
    source TEXT NOT NULL CHECK(source IN ('live', 'scout')),
    chat_id INTEGER NOT NULL,
    telegram_msg_id INTEGER NOT NULL,
    chat_title TEXT,
    sender_id INTEGER,
    sender_name TEXT,
    text TEXT NOT NULL,
    date TIMESTAMP NOT NULL,
    -- Normalized-text digest. 25% of the corpus is the same announcement
    -- crossposted across chats; results roll up on this so one event does not
    -- fill a whole answer with copies of itself.
    content_hash TEXT NOT NULL,
    indexed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id, telegram_msg_id)
);

CREATE INDEX IF NOT EXISTS idx_corpus_chat_date ON corpus_messages(chat_id, date);
CREATE INDEX IF NOT EXISTS idx_corpus_date ON corpus_messages(date);
CREATE INDEX IF NOT EXISTS idx_corpus_hash ON corpus_messages(content_hash);

-- Chat-level metadata, denormalized so a search result can name its chat
-- without attaching either source database.
CREATE TABLE IF NOT EXISTS corpus_chats (
    chat_id INTEGER PRIMARY KEY,
    title TEXT,
    username TEXT,
    -- 'monitor' | 'recon' | 'paused' | 'history-only' (backfilled, not observed)
    mode TEXT,
    city_area TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    first_message_at TIMESTAMP,
    last_message_at TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Lexical lane.
--
-- unicode61 + a prefix index rather than trigram: measured on this corpus,
-- prefix matching reaches the same recall on inflected Russian (концерт 95/95,
-- кафе 373/376) at 0.84x the index size and 3-6x the query speed. Bare
-- unicode61 without the prefix index is the real failure mode -- 14.6% recall
-- on "бар" -- because Russian suffixation never matches an exact-token index.
CREATE VIRTUAL TABLE IF NOT EXISTS corpus_fts USING fts5(
    text,
    content='corpus_messages',
    content_rowid='corpus_id',
    tokenize='unicode61 remove_diacritics 2',
    prefix='2 3 4'
);

CREATE TRIGGER IF NOT EXISTS corpus_fts_ai AFTER INSERT ON corpus_messages BEGIN
    INSERT INTO corpus_fts(rowid, text) VALUES (new.corpus_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS corpus_fts_ad AFTER DELETE ON corpus_messages BEGIN
    INSERT INTO corpus_fts(corpus_fts, rowid, text) VALUES ('delete', old.corpus_id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS corpus_fts_au AFTER UPDATE ON corpus_messages BEGIN
    INSERT INTO corpus_fts(corpus_fts, rowid, text) VALUES ('delete', old.corpus_id, old.text);
    INSERT INTO corpus_fts(rowid, text) VALUES (new.corpus_id, new.text);
END;

-- Semantic lane.
--
-- Plain float32 blobs scored by numpy rather than a vector extension: numpy is
-- already an installed dependency, brute force over tens of thousands of
-- vectors is a single matmul, and this loads on any Python build. An extension
-- would buy ANN scaling this corpus is three orders of magnitude away from
-- needing, at the cost of a load path that is disabled on some interpreters.
CREATE TABLE IF NOT EXISTS corpus_embeddings (
    corpus_id INTEGER PRIMARY KEY REFERENCES corpus_messages(corpus_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    embedded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Extracted entities.
--
-- The question "which venues could host a concert" is an entity question, and
-- re-deriving venue names from raw text on every query is both slow and
-- inconsistent between runs. Extraction happens once per message; queries
-- become a GROUP BY.
CREATE TABLE IF NOT EXISTS places (
    place_id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Canonical lowercase key used for dedup. Stylized branding is folded to
    -- ASCII here (SYNCHØUSE -> synchouse) because a user types the plain form
    -- and neither tokenizer folds a distinct Unicode letter for them.
    canonical TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    city_area TEXT,
    place_type TEXT,
    first_seen_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    mention_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_places_city ON places(city_area);
CREATE INDEX IF NOT EXISTS idx_places_type ON places(place_type);

CREATE TABLE IF NOT EXISTS place_mentions (
    mention_id INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id INTEGER NOT NULL REFERENCES places(place_id) ON DELETE CASCADE,
    corpus_id INTEGER NOT NULL REFERENCES corpus_messages(corpus_id) ON DELETE CASCADE,
    event_types TEXT NOT NULL DEFAULT '[]',
    -- Verbatim fragment, same discipline as the L3 classifier's evidence
    -- field: a claim the reader can check against the source message.
    evidence_quote TEXT NOT NULL,
    confidence REAL,
    extracted_by TEXT NOT NULL,
    extracted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(place_id, corpus_id)
);

CREATE INDEX IF NOT EXISTS idx_mentions_corpus ON place_mentions(corpus_id);

-- Venue-name lookup. Trigram earns its cost on a few hundred proper nouns,
-- where substring and typo tolerance matter and index size does not.
CREATE VIRTUAL TABLE IF NOT EXISTS place_fts USING fts5(
    name, aliases,
    content='places',
    content_rowid='place_id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS place_fts_ai AFTER INSERT ON places BEGIN
    INSERT INTO place_fts(rowid, name, aliases) VALUES (new.place_id, new.name, new.aliases);
END;

CREATE TRIGGER IF NOT EXISTS place_fts_ad AFTER DELETE ON places BEGIN
    INSERT INTO place_fts(place_fts, rowid, name, aliases)
        VALUES ('delete', old.place_id, old.name, old.aliases);
END;

CREATE TRIGGER IF NOT EXISTS place_fts_au AFTER UPDATE ON places BEGIN
    INSERT INTO place_fts(place_fts, rowid, name, aliases)
        VALUES ('delete', old.place_id, old.name, old.aliases);
    INSERT INTO place_fts(rowid, name, aliases) VALUES (new.place_id, new.name, new.aliases);
END;

-- Chat references mined out of message text: the t.me links already sitting in
-- the corpus, which are the cheapest discovery source available and cost no
-- Telegram action budget to collect.
CREATE TABLE IF NOT EXISTS chat_references (
    ref TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('username', 'invite', 'unknown')),
    mention_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    -- Populated from join_queue/observed_chats at sync time so the agent can
    -- tell "already ours" from "worth joining" without a second lookup.
    known_state TEXT,
    sample_corpus_id INTEGER
);

-- Per-message extraction bookkeeping. Same resumable-worker shape the pipeline
-- already uses for pipeline_runs: only 'pending' rows are work.
CREATE TABLE IF NOT EXISTS extraction_state (
    corpus_id INTEGER PRIMARY KEY REFERENCES corpus_messages(corpus_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'extracted', 'no_venue', 'skipped', 'error')),
    attempted_at TIMESTAMP,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_extraction_pending ON extraction_state(status, corpus_id);

-- Incremental cursors, one row per source stream.
CREATE TABLE IF NOT EXISTS sync_state (
    source TEXT PRIMARY KEY,
    last_id INTEGER NOT NULL DEFAULT 0,
    rows_synced INTEGER NOT NULL DEFAULT 0,
    -- Ticks since the last full rescan. A cursor over a table whose rowid can
    -- be reused cannot be made exact by counting: deleting one row and
    -- inserting another leaves the count identical while the new row sits
    -- below the cursor, invisible forever. Rescanning on a schedule is the
    -- only cheap thing that is actually correct, and at this corpus size a
    -- full pass costs a few seconds.
    ticks INTEGER NOT NULL DEFAULT 0,
    last_run_at TIMESTAMP
);
