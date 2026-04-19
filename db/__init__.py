import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parents[1] / "trends.db"
SCHEMA_FILE = Path(__file__).parent / "schema.sql"

# Canonical name + lookup for the Uncategorized sentinel category.
# Defined here to avoid drift across db/painpoints.py, db/embeddings.py,
# db/category_events.py — they all used to redefine these.
UNCATEGORIZED_NAME = "Uncategorized"


def uncategorized_id(conn):
    """Resolve the Uncategorized sentinel category id. Cached in
    sqlite3.Connection.row_factory's local memo would be nicer but
    sqlite3.Connection isn't subscriptable; the SELECT is one
    indexed lookup, fast enough."""
    row = conn.execute(
        "SELECT id FROM categories WHERE name = ?", (UNCATEGORIZED_NAME,)
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "Uncategorized sentinel category missing — db.init_db() not run?"
        )
    return row["id"]


# Painpoint ingest pipeline migrations.
# Each entry is an idempotent ALTER TABLE; we run them via a try/except for
# "duplicate column name" because SQLite has no ADD COLUMN IF NOT EXISTS.
# Kept as a list (not dropped) because existing DBs still have the legacy
# `relevance`, `relevance_updated_at`, `last_split_check_at` columns —
# leaving them in place is a no-op; we just don't read/write them anymore.
_MIGRATIONS = [
    "ALTER TABLE categories ADD COLUMN painpoint_count_at_last_check INTEGER DEFAULT 0",
    # Incremental-centroid state: we keep (sum_of_member_embeddings, count)
    # on the category row so update_category_embedding is O(1) instead of
    # JOIN-and-average-all-members on every mutation.
    "ALTER TABLE categories ADD COLUMN member_emb_sum_blob BLOB",
    "ALTER TABLE categories ADD COLUMN member_emb_count INTEGER NOT NULL DEFAULT 0",
    # When the category's MEMBER SET last changed (add / remove / move),
    # dedicated from painpoints.last_updated which also fires on
    # signal_count bumps — using MAX(last_updated) conflated "real
    # membership activity" with "duplicate-pending bumps".
    "ALTER TABLE categories ADD COLUMN member_set_last_changed_at TEXT",
    # When the category's centroid was last rewritten — used by the
    # reroute step to skip painpoints whose current category centroid
    # hasn't moved since they were last re-checked.
    "ALTER TABLE categories ADD COLUMN centroid_updated_at TEXT",
    # When this painpoint was last reroute-checked; skip re-checking if
    # nothing relevant has changed since.
    "ALTER TABLE painpoints ADD COLUMN reroute_checked_at TEXT",
    # Seed vs runtime-minted. Seed categories (from taxonomy.yaml + the
    # Uncategorized sentinel) have human-curated name/description anchors;
    # runtime-minted ones have LLM-synthesized descriptions of unknown
    # quality. update_category_embedding uses this flag to pick the
    # anchor-vs-member-mean blend weight — seeds trust the anchor heavily
    # (declared intent), runtime categories lean more on member evidence.
    "ALTER TABLE categories ADD COLUMN is_seed INTEGER NOT NULL DEFAULT 0",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def in_clause_placeholders(n):
    """Build the `?,?,...` placeholder string for a SQL `IN (...)` clause.
    Centralised so ad-hoc `','.join('?' * len(ids))` patterns don't drift;
    callers still bind the values themselves.
    """
    if n <= 0:
        raise ValueError(f"in_clause_placeholders needs n >= 1, got {n}")
    return ",".join("?" * n)


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
        "INSERT OR IGNORE INTO categories "
        "(name, parent_id, description, created_at, is_seed) "
        "VALUES (?, NULL, ?, ?, 1)",
        ("Uncategorized",
         "Sentinel bucket for painpoints awaiting category-worker processing.",
         _now()),
    )
    conn.commit()
    conn.close()

    # Backfill is_seed for DBs created before this column existed. Also
    # picks up cases where taxonomy.yaml grew a new entry whose name
    # matches an existing runtime-minted category — that category now
    # gets promoted to seed, which is the taxonomist's intent.
    from .seed import backfill_is_seed
    backfill_is_seed()

    # Bulk-populate category_fts for any category that predates the
    # virtual table (or was inserted on a code path that didn't sync).
    # Idempotent — already-indexed categories are skipped by anti-join.
    from .category_retrieval import init_category_fts
    conn = get_db()
    try:
        init_category_fts(conn)
        conn.commit()
    finally:
        conn.close()

    # Eagerly bootstrap category embeddings if an OPENAI_API_KEY is set.
    # Done OUTSIDE the merge_lock so the first promote_pending doesn't
    # eat a multi-second OpenAI HTTP round-trip while holding the lock.
    # Skips silently if no key (tests, fresh checkouts).
    import os
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from .embeddings import OpenAIEmbedder, bootstrap_category_embeddings
            conn = get_db()
            try:
                bootstrap_category_embeddings(conn, OpenAIEmbedder())
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            # Don't fail init_db if bootstrap has issues — the worker
            # will fall back to inline bootstrap on first promote.
            import logging
            logging.getLogger(__name__).warning(
                "init_db: eager category-embedding bootstrap failed (%s) — "
                "first promote will bootstrap inline instead", e,
            )
