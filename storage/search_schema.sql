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
    reply_to_message_id INTEGER,
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
-- Backfilled messages arrive without their sender's name and get one from any
-- other message by the same id; without this the fill scans the whole table
-- once per sender.
CREATE INDEX IF NOT EXISTS idx_corpus_sender ON corpus_messages(sender_id);

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
    -- Structural facets stay intentionally small. Subject categories live in
    -- descriptor/open offering text and never require a schema change.
    entity_kind TEXT NOT NULL DEFAULT 'place'
        CHECK(entity_kind IN ('place', 'person', 'organization')),
    access_modes TEXT NOT NULL DEFAULT '["visit"]' CHECK(json_valid(access_modes)),
    primary_descriptor TEXT,
    descriptor_text TEXT NOT NULL DEFAULT '',
    offering_text TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_places_city ON places(city_area);
CREATE INDEX IF NOT EXISTS idx_places_type ON places(place_type);
CREATE TABLE IF NOT EXISTS descriptors (
    descriptor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    normalized TEXT NOT NULL UNIQUE,
    display_text TEXT NOT NULL,
    language TEXT,
    mention_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS place_descriptors (
    place_id INTEGER NOT NULL REFERENCES places(place_id) ON DELETE CASCADE,
    descriptor_id INTEGER NOT NULL REFERENCES descriptors(descriptor_id) ON DELETE CASCADE,
    mention_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    PRIMARY KEY (place_id, descriptor_id)
);

CREATE INDEX IF NOT EXISTS idx_place_descriptors_descriptor
    ON place_descriptors(descriptor_id, place_id);

CREATE TABLE IF NOT EXISTS descriptor_embeddings (
    descriptor_id INTEGER NOT NULL REFERENCES descriptors(descriptor_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'ready', 'error')),
    dim INTEGER,
    vec BLOB,
    attempts INTEGER NOT NULL DEFAULT 0,
    attempted_at TIMESTAMP,
    error TEXT,
    embedded_at TIMESTAMP,
    PRIMARY KEY (descriptor_id, model),
    CHECK((status = 'ready' AND dim IS NOT NULL AND vec IS NOT NULL)
        OR (status <> 'ready' AND vec IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_descriptor_embeddings_work
    ON descriptor_embeddings(model, status, descriptor_id);

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
    descriptor_id INTEGER REFERENCES descriptors(descriptor_id),
    descriptor_raw TEXT,
    offerings_raw TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(offerings_raw)),
    entity_kind_raw TEXT,
    access_modes_raw TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(access_modes_raw)),
    extractor_version TEXT NOT NULL DEFAULT 'places-v2',
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
        CHECK(status IN ('pending', 'extracted', 'no_venue', 'skipped', 'error', 'duplicate')),
    -- A provider timeout is not a verdict about the message. Without a retry
    -- the row stays 'error' forever and becomes a permanent hole in the venue
    -- index -- invisible, because an empty answer looks the same as an honest
    -- one. The counter is what stops a message that fails deterministically
    -- from being retried for the rest of the corpus's life.
    attempts INTEGER NOT NULL DEFAULT 0,
    attempted_at TIMESTAMP,
    error TEXT,
    active_prompt_version TEXT NOT NULL DEFAULT 'places-v2'
);

CREATE INDEX IF NOT EXISTS idx_extraction_pending ON extraction_state(status, corpus_id);

-- A prompt version is a durable unit of work. Failed replacement jobs leave
-- the previous active extraction untouched and can be retried without
-- creating another row or erasing paid output.
CREATE TABLE IF NOT EXISTS extraction_jobs (
    corpus_id INTEGER NOT NULL REFERENCES corpus_messages(corpus_id) ON DELETE CASCADE,
    prompt_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'running', 'succeeded', 'error')),
    attempts INTEGER NOT NULL DEFAULT 0,
    not_before TIMESTAMP,
    attempted_at TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT,
    PRIMARY KEY (corpus_id, prompt_version)
);

CREATE INDEX IF NOT EXISTS idx_extraction_jobs_work
    ON extraction_jobs(prompt_version, status, not_before, corpus_id);

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

-- Contact handles mined out of message text. Deterministic, not extracted by a
-- model: a handle is a lexical pattern, and paying a model to find one would
-- buy nothing but a chance to hallucinate it. Kept per-message rather than per
-- place so the same rows answer both "how do I reach whoever runs this venue"
-- (join through place_mentions) and "who is this person" (join through sender).
CREATE TABLE IF NOT EXISTS message_contacts (
    contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus_id INTEGER NOT NULL REFERENCES corpus_messages(corpus_id) ON DELETE CASCADE,
    kind TEXT NOT NULL
        CHECK(kind IN ('telegram', 'phone', 'instagram', 'whatsapp', 'zalo', 'facebook', 'maps')),
    -- Comparable form: handles lowercased and stripped of their sigil, phone
    -- numbers reduced to digits. Two spellings of one contact must collapse,
    -- otherwise a venue's organiser is counted three times and ranked as three.
    value TEXT NOT NULL,
    -- How it actually appeared, for quoting back at the reader.
    display TEXT NOT NULL,
    UNIQUE(corpus_id, kind, value)
);

CREATE INDEX IF NOT EXISTS idx_message_contacts_value ON message_contacts(kind, value);
CREATE INDEX IF NOT EXISTS idx_message_contacts_corpus ON message_contacts(corpus_id);

-- What each extraction pass actually cost. The token counts come back in every
-- API response and were being thrown away, so every statement about this
-- system's bill was an estimate derived from prompt length. One row per run.
CREATE TABLE IF NOT EXISTS extraction_cost (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    messages INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    -- Read back from cache at a tenth of the input rate. Zero until the stable
    -- prefix crosses the provider's minimum, which is worth being able to see.
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- Housing listings: the structured record of what was on offer, and when.
--
-- Derived like everything else in this file — every row is rebuildable from
-- the corpus by re-running the extractor — and written only by index_cli.py,
-- never by the daemon.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS housing_listings (
    listing_id INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus_id INTEGER NOT NULL UNIQUE REFERENCES corpus_messages(corpus_id) ON DELETE CASCADE,
    chat_id INTEGER NOT NULL,
    -- The same advertisement crossposted into three chats is three rows with
    -- one hash. Trend aggregation counts a hash once; keeping the rows is what
    -- lets a later question ask where a listing appeared.
    content_hash TEXT,
    posted_at TIMESTAMP NOT NULL,
    is_rental_offer INTEGER NOT NULL,
    is_vehicle_ad INTEGER NOT NULL,
    bedrooms INTEGER,
    bathrooms INTEGER,
    monthly_price_thb INTEGER,
    price_note TEXT,
    tv_present INTEGER,
    tv_size_class TEXT,
    area_raw TEXT,
    evidence_quote TEXT,
    extractor_version TEXT NOT NULL,
    extracted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_housing_listings_month
    ON housing_listings(posted_at) WHERE is_rental_offer = 1;
CREATE INDEX IF NOT EXISTS idx_housing_listings_hash ON housing_listings(content_hash);

CREATE TABLE IF NOT EXISTS housing_listing_state (
    corpus_id INTEGER PRIMARY KEY REFERENCES corpus_messages(corpus_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'extracted', 'not_housing', 'gated', 'error')),
    attempts INTEGER NOT NULL DEFAULT 0,
    attempted_at TIMESTAMP,
    error TEXT,
    extractor_version TEXT NOT NULL DEFAULT 'housing-text-v1'
);

CREATE INDEX IF NOT EXISTS idx_housing_state_pending
    ON housing_listing_state(status, corpus_id);

-- Append-only. A bucket recomputed after more history arrives is a new row,
-- not an edit: "why is March different from what I saw yesterday" has to have
-- an answer, and n_chats_included is usually it.
CREATE TABLE IF NOT EXISTS housing_price_trend (
    -- A counter rather than the timestamp: two runs inside the same second
    -- are two runs, and keying on CURRENT_TIMESTAMP made the second one fail
    -- an integrity check instead of appending.
    run_id INTEGER NOT NULL,
    period_kind TEXT NOT NULL CHECK(period_kind IN ('month', 'quarter')),
    period TEXT NOT NULL,
    bedrooms_bucket TEXT NOT NULL,
    n_listings INTEGER NOT NULL,
    n_priced INTEGER NOT NULL,
    median_thb REAL,
    p25_thb REAL,
    p75_thb REAL,
    n_chats_included INTEGER NOT NULL,
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (period_kind, period, bedrooms_bucket, run_id)
);
