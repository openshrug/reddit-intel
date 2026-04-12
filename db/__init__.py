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
    # §7.3: (removed — minhash_blob replaced by sqlite-vec embeddings)
    # §7.4: split-check trigger discipline
    "ALTER TABLE categories ADD COLUMN last_split_check_at TEXT",
    "ALTER TABLE categories ADD COLUMN painpoint_count_at_last_check INTEGER DEFAULT 0",
    # §7.7: (removed — signal_score and derived columns dropped in v3)
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    # Load sqlite-vec for vector similarity search
    from .embeddings import load_sqlite_vec
    load_sqlite_vec(conn)
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

    # (signal_score index removed in v3 — column no longer exists)


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA_FILE.read_text())
    _apply_migrations(conn)
    # Create sqlite-vec virtual tables for embedding similarity
    from .embeddings import init_vec_tables
    init_vec_tables(conn)
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
