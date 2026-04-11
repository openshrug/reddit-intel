import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parents[1] / "trends.db"
SCHEMA_FILE = Path(__file__).parent / "schema.sql"


# Painpoint ingest pipeline migrations (see docs/PAINPOINT_INGEST_PLAN.md §7).
# Each entry is an idempotent ALTER TABLE; we run them via a try/except for
# "duplicate column name" because SQLite has no ADD COLUMN IF NOT EXISTS.
_MIGRATIONS = [
    # §7.1: cache relevance on the painpoint row
    "ALTER TABLE painpoints ADD COLUMN relevance REAL",
    "ALTER TABLE painpoints ADD COLUMN relevance_updated_at TEXT",
    # §7.3: cache the MinHash signature
    "ALTER TABLE painpoints ADD COLUMN minhash_blob BLOB",
    # §7.4: split-check trigger discipline
    "ALTER TABLE categories ADD COLUMN last_split_check_at TEXT",
    "ALTER TABLE categories ADD COLUMN painpoint_count_at_last_check INTEGER DEFAULT 0",
    # §7.7: post-level signal_score (compute logic from SIGNAL_SCORING_PLAN.md)
    "ALTER TABLE posts ADD COLUMN signal_score REAL",
    "ALTER TABLE posts ADD COLUMN signal_score_updated_at TEXT",
    "ALTER TABLE posts ADD COLUMN upvote_count INTEGER",
    "ALTER TABLE posts ADD COLUMN downvote_count INTEGER",
    "ALTER TABLE posts ADD COLUMN cluster_size INTEGER DEFAULT 1",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _apply_migrations(conn):
    """Apply ALTER TABLE migrations idempotently. SQLite has no ADD COLUMN IF
    NOT EXISTS, so we catch the duplicate-column-name error per statement."""
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "duplicate column name" not in msg:
                raise

    # Indexes that depend on columns added by the migrations above must be
    # created here, not in schema.sql (which can't reference columns that
    # don't exist yet at script-execution time).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_posts_signal_score "
        "ON posts(signal_score DESC)"
    )


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA_FILE.read_text())
    _apply_migrations(conn)
    conn.commit()
    conn.close()

    # Seed the YAML taxonomy first — seed_taxonomy() short-circuits if the
    # categories table already has any rows, so it must run before we add the
    # Uncategorized sentinel.
    from .seed import seed_taxonomy
    seed_taxonomy()

    # §7.6: Uncategorized sentinel — always exists, never deleted.
    # INSERT OR IGNORE so re-running init_db() is a no-op.
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO categories (name, parent_id, description, created_at) "
        "VALUES (?, NULL, ?, ?)",
        ("Uncategorized",
         "Sentinel bucket for painpoints awaiting category-worker processing.",
         _now()),
    )
    conn.commit()
    conn.close()
