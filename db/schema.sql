-- reddit-intel schema (MVP)
-- 6 tables: posts, comments, pending_painpoints, painpoints, painpoint_sources, categories

-- === RAW STORAGE ===

CREATE TABLE IF NOT EXISTS posts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    reddit_id     TEXT UNIQUE NOT NULL,
    subreddit     TEXT NOT NULL,
    title         TEXT NOT NULL,
    selftext      TEXT,
    url           TEXT,
    author        TEXT,
    score         INTEGER DEFAULT 0,
    upvote_ratio  REAL,
    num_comments  INTEGER DEFAULT 0,
    permalink     TEXT NOT NULL,
    created_utc   REAL,
    is_self       INTEGER DEFAULT 0,
    link_flair    TEXT,
    stickied      INTEGER DEFAULT 0,
    extra         JSON,
    fetched_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_utc);

CREATE TABLE IF NOT EXISTS comments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    reddit_id        TEXT UNIQUE NOT NULL,
    post_id          INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    parent_reddit_id TEXT,
    author           TEXT,
    body             TEXT NOT NULL,
    score            INTEGER DEFAULT 0,
    controversiality INTEGER DEFAULT 0,
    permalink        TEXT,
    created_utc      REAL,
    depth            INTEGER DEFAULT 0,
    fetched_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);

-- === TAXONOMY ===

CREATE TABLE IF NOT EXISTS categories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    parent_id   INTEGER REFERENCES categories(id),
    description TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_id);

-- === PAINPOINT EXTRACTION ===

CREATE TABLE IF NOT EXISTS pending_painpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id         INTEGER NOT NULL REFERENCES posts(id),
    comment_id      INTEGER REFERENCES comments(id),
    category_id     INTEGER REFERENCES categories(id),
    title           TEXT NOT NULL,
    description     TEXT,
    quoted_text     TEXT,
    severity        INTEGER DEFAULT 5 CHECK(severity BETWEEN 1 AND 10),
    extracted_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pending_pp_post ON pending_painpoints(post_id);
CREATE INDEX IF NOT EXISTS idx_pending_pp_category ON pending_painpoints(category_id);

-- === MERGED PAINPOINTS ===

CREATE TABLE IF NOT EXISTS painpoints (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    description   TEXT,
    severity      INTEGER DEFAULT 5 CHECK(severity BETWEEN 1 AND 10),
    signal_count  INTEGER DEFAULT 1,
    category_id   INTEGER REFERENCES categories(id),
    first_seen    TEXT NOT NULL,
    last_updated  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_painpoints_category ON painpoints(category_id);

CREATE TABLE IF NOT EXISTS painpoint_sources (
    painpoint_id          INTEGER NOT NULL REFERENCES painpoints(id) ON DELETE CASCADE,
    pending_painpoint_id  INTEGER NOT NULL REFERENCES pending_painpoints(id) ON DELETE CASCADE,
    PRIMARY KEY (painpoint_id, pending_painpoint_id)
);

-- ============================================================================
-- Painpoint ingest pipeline — idempotent CREATEs only
-- (see docs/PAINPOINT_INGEST_PLAN.md). ALTER TABLE additions for
-- existing tables live in db/__init__.py:_apply_migrations() because
-- SQLite has no ALTER TABLE ... ADD COLUMN IF NOT EXISTS.
-- ============================================================================

-- §7.5: multi-source pending painpoints. The existing
-- pending_painpoints.(post_id, comment_id) hold the *primary* source;
-- additional sources for batched LLM extraction go in this junction.
CREATE TABLE IF NOT EXISTS pending_painpoint_sources (
    pending_painpoint_id INTEGER NOT NULL REFERENCES pending_painpoints(id) ON DELETE CASCADE,
    post_id              INTEGER NOT NULL REFERENCES posts(id),
    comment_id           INTEGER REFERENCES comments(id)
);
-- SQLite PRIMARY KEYs can't contain expressions; uniqueness with nullable
-- comment_id needs an expression index instead.
CREATE UNIQUE INDEX IF NOT EXISTS idx_pps_unique
    ON pending_painpoint_sources(pending_painpoint_id, post_id, COALESCE(comment_id, -1));
CREATE INDEX IF NOT EXISTS idx_pps_post ON pending_painpoint_sources(post_id);

-- View that unions primary + extra sources so callers don't have to
-- remember the two-place storage.
CREATE VIEW IF NOT EXISTS pending_painpoint_all_sources AS
    SELECT id AS pending_painpoint_id, post_id, comment_id
        FROM pending_painpoints
    UNION
    SELECT pending_painpoint_id, post_id, comment_id
        FROM pending_painpoint_sources;

-- §7.2: per-event audit trail for the category worker (see §5.3).
-- triggering_pp and target_category are plain integer references, NOT
-- FKs — the event itself can delete the row it points at (e.g.,
-- delete_category deletes its target), and we still want the audit
-- row to be valid after the fact.
CREATE TABLE IF NOT EXISTS category_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type       TEXT NOT NULL,
    proposed_at      TEXT NOT NULL,
    triggering_pp    INTEGER,
    target_category  INTEGER,
    payload_json     JSON NOT NULL,
    metric_name      TEXT NOT NULL,
    metric_value     REAL NOT NULL,
    threshold        REAL NOT NULL,
    accepted         INTEGER NOT NULL,
    reason           TEXT
);
CREATE INDEX IF NOT EXISTS idx_cat_events_proposed ON category_events(proposed_at);
CREATE INDEX IF NOT EXISTS idx_cat_events_type ON category_events(event_type);

-- (§7.7 signal_score removed in v3 — replaced by embedding-based pipeline.)
