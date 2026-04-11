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
