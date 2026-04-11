import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

DB_PATH = Path(__file__).parent / "trends.db"


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")  # wait up to 30s for locks
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""

    -- === CORE ENTITIES ===

    CREATE TABLE IF NOT EXISTS runs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at  TEXT NOT NULL,
        finished_at TEXT,
        sources     TEXT,           -- JSON list
        signals_total   INTEGER DEFAULT 0,
        signals_kept    INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS signals (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      INTEGER NOT NULL REFERENCES runs(id),
        source      TEXT NOT NULL,
        title       TEXT NOT NULL,
        url         TEXT,
        score       INTEGER DEFAULT 0,
        percentile  REAL,           -- 0-100, normalized within source per run
        extra       JSON,
        scraped_at  TEXT NOT NULL,
        kept        INTEGER DEFAULT 0,
        importance  INTEGER,
        why_kept    TEXT
    );

    CREATE TABLE IF NOT EXISTS categories (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT UNIQUE NOT NULL,
        slug        TEXT UNIQUE NOT NULL,
        parent_id   INTEGER REFERENCES categories(id),  -- NULL = root
        description TEXT,
        created_by  TEXT DEFAULT 'system',
        created_at  TEXT NOT NULL
    );

    -- LLM can propose new categories; they only get promoted after N signals
    CREATE TABLE IF NOT EXISTS pending_categories (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT UNIQUE NOT NULL,
        parent_slug TEXT,  -- suggested parent
        signal_count INTEGER DEFAULT 1,
        first_seen  TEXT NOT NULL
    );

    -- === PRODUCTS ===

    CREATE TABLE IF NOT EXISTS products (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT UNIQUE NOT NULL,
        url             TEXT,
        description     TEXT,
        builder         TEXT,
        tech_complexity TEXT CHECK(tech_complexity IN ('LOW','MEDIUM','HIGH')),
        viral_trigger   TEXT,
        why_viral       TEXT,
        status          TEXT DEFAULT 'active' CHECK(status IN ('active','stalling','dead')),
        first_seen      TEXT NOT NULL,
        last_updated    TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS product_categories (
        product_id  INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
        category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
        PRIMARY KEY (product_id, category_id)
    );

    -- === PAINPOINTS ===

    CREATE TABLE IF NOT EXISTS painpoints (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        title           TEXT NOT NULL,
        description     TEXT,
        severity        INTEGER DEFAULT 5 CHECK(severity BETWEEN 1 AND 10),
        signal_count    INTEGER DEFAULT 1,
        first_seen      TEXT NOT NULL,
        last_updated    TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS painpoint_categories (
        painpoint_id INTEGER NOT NULL REFERENCES painpoints(id) ON DELETE CASCADE,
        category_id  INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
        PRIMARY KEY (painpoint_id, category_id)
    );

    CREATE TABLE IF NOT EXISTS quotes (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        painpoint_id    INTEGER REFERENCES painpoints(id) ON DELETE CASCADE,
        product_id      INTEGER REFERENCES products(id) ON DELETE SET NULL,
        text            TEXT NOT NULL,
        source          TEXT,
        source_url      TEXT,
        sentiment       TEXT DEFAULT 'negative' CHECK(sentiment IN ('negative','positive','neutral')),
        scraped_at      TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS product_painpoints (
        product_id      INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
        painpoint_id    INTEGER NOT NULL REFERENCES painpoints(id) ON DELETE CASCADE,
        relationship    TEXT DEFAULT 'addresses' CHECK(relationship IN ('addresses','fails_at','partial')),
        effectiveness   INTEGER CHECK(effectiveness BETWEEN 1 AND 10),  -- how well does it solve this pain?
        gap_description TEXT,   -- what's STILL missing despite this product existing
        gap_type        TEXT,   -- pricing, performance, ux, scope, reliability, integration
        notes           TEXT,
        PRIMARY KEY (product_id, painpoint_id)
    );

    -- === FUNDING ===

    CREATE TABLE IF NOT EXISTS investors (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        name    TEXT UNIQUE NOT NULL,
        type    TEXT,   -- vc, angel, corporate, retail
        url     TEXT
    );

    CREATE TABLE IF NOT EXISTS funding_rounds (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id      INTEGER REFERENCES products(id) ON DELETE SET NULL,
        amount          TEXT,
        amount_usd      INTEGER,    -- normalized to USD for sorting
        valuation       TEXT,
        round_type      TEXT,
        what_they_build TEXT,
        why_funded      TEXT,       -- investor thesis
        source_url      TEXT,
        announced_at    TEXT,
        scraped_at      TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS round_investors (
        round_id    INTEGER NOT NULL REFERENCES funding_rounds(id) ON DELETE CASCADE,
        investor_id INTEGER NOT NULL REFERENCES investors(id) ON DELETE CASCADE,
        lead        INTEGER DEFAULT 0,
        PRIMARY KEY (round_id, investor_id)
    );

    CREATE TABLE IF NOT EXISTS round_painpoints (
        round_id     INTEGER NOT NULL REFERENCES funding_rounds(id) ON DELETE CASCADE,
        painpoint_id INTEGER NOT NULL REFERENCES painpoints(id) ON DELETE CASCADE,
        PRIMARY KEY (round_id, painpoint_id)
    );

    CREATE TABLE IF NOT EXISTS funding_categories (
        round_id    INTEGER NOT NULL REFERENCES funding_rounds(id) ON DELETE CASCADE,
        category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
        PRIMARY KEY (round_id, category_id)
    );

    -- === SCRAPE CURSORS (cross-round pagination state) ===

    CREATE TABLE IF NOT EXISTS scrape_cursors (
        source      TEXT NOT NULL,  -- e.g. 'reddit'
        cursor_key  TEXT NOT NULL,  -- e.g. 'programming|hot|week'
        cursor      TEXT,           -- opaque cursor token (null = exhausted)
        updated_at  TEXT NOT NULL,
        PRIMARY KEY (source, cursor_key)
    );

    -- === INDEXES ===

    CREATE INDEX IF NOT EXISTS idx_signals_run ON signals(run_id);
    CREATE INDEX IF NOT EXISTS idx_signals_source ON signals(source);
    CREATE INDEX IF NOT EXISTS idx_signals_kept ON signals(kept);
    CREATE INDEX IF NOT EXISTS idx_products_name ON products(name);
    CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_id);
    CREATE INDEX IF NOT EXISTS idx_painpoint_categories_cat ON painpoint_categories(category_id);
    CREATE INDEX IF NOT EXISTS idx_quotes_painpoint ON quotes(painpoint_id);
    CREATE INDEX IF NOT EXISTS idx_funding_product ON funding_rounds(product_id);
    CREATE INDEX IF NOT EXISTS idx_funding_amount ON funding_rounds(amount_usd);
    """)
    conn.commit()
    conn.close()


# --- Run management ---

def start_run(sources):
    conn = get_db()
    now = _now()
    cur = conn.execute(
        "INSERT INTO runs (started_at, sources) VALUES (?, ?)",
        (now, json.dumps(sources))
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def finish_run(run_id, signals_total, signals_kept):
    conn = get_db()
    conn.execute(
        "UPDATE runs SET finished_at=?, signals_total=?, signals_kept=? WHERE id=?",
        (_now(), signals_total, signals_kept, run_id)
    )
    conn.commit()
    conn.close()


# --- Signals ---

def save_signals(run_id, source, items):
    conn = get_db()
    now = _now()
    for item in items:
        conn.execute(
            "INSERT INTO signals (run_id, source, title, url, score, extra, scraped_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, source,
             item.get("title", item.get("name", "")),
             item.get("url", ""),
             item.get("score", item.get("stars", 0)),
             json.dumps(item), now)
        )
    conn.commit()
    conn.close()


# --- Scrape cursors (cross-round pagination state) ---

def get_cursor(source, key):
    """Get the saved cursor for (source, key). Returns None if not set or exhausted."""
    conn = get_db()
    row = conn.execute(
        "SELECT cursor FROM scrape_cursors WHERE source=? AND cursor_key=?",
        (source, key)
    ).fetchone()
    conn.close()
    return row["cursor"] if row else None


def save_cursor(source, key, cursor):
    """Save or update a cursor. cursor=None means 'exhausted, start over next time'."""
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO scrape_cursors (source, cursor_key, cursor, updated_at) VALUES (?,?,?,?)",
        (source, key, cursor, _now())
    )
    conn.commit()
    conn.close()


def reset_cursor(source, key=None):
    """Reset a cursor (or all cursors for a source)."""
    conn = get_db()
    if key:
        conn.execute("DELETE FROM scrape_cursors WHERE source=? AND cursor_key=?", (source, key))
    else:
        conn.execute("DELETE FROM scrape_cursors WHERE source=?", (source,))
    conn.commit()
    conn.close()


def compute_percentiles(run_id):
    """Compute percentile rank (0-100) for each signal within its source for this run."""
    conn = get_db()

    # Get distinct sources in this run
    sources = conn.execute(
        "SELECT DISTINCT source FROM signals WHERE run_id=?", (run_id,)
    ).fetchall()

    for (source,) in sources:
        # Get all scores for this source, sorted ascending
        rows = conn.execute(
            "SELECT id, score FROM signals WHERE run_id=? AND source=? ORDER BY score ASC",
            (run_id, source)
        ).fetchall()

        n = len(rows)
        if n == 0:
            continue

        if n == 1:
            conn.execute("UPDATE signals SET percentile=50.0 WHERE id=?", (rows[0]["id"],))
            continue

        # Assign percentile: (rank / n) * 100
        for rank, row in enumerate(rows):
            pct = round((rank / (n - 1)) * 100, 1) if n > 1 else 50.0
            conn.execute("UPDATE signals SET percentile=? WHERE id=?", (pct, row["id"]))

    conn.commit()
    conn.close()


# --- Products ---

_PRODUCT_FIELDS = ("url", "description", "builder", "tech_complexity",
                   "viral_trigger", "why_viral", "status")
_VALID_TECH_COMPLEXITY = {"LOW", "MEDIUM", "HIGH"}


def _clean_product_kwargs(kwargs):
    """Filter product kwargs: drop None/empty, validate enums."""
    cleaned = {}
    for k, v in kwargs.items():
        if k not in _PRODUCT_FIELDS:
            continue
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        # Validate tech_complexity enum
        if k == "tech_complexity" and v.upper() not in _VALID_TECH_COMPLEXITY:
            continue
        cleaned[k] = v.upper() if k == "tech_complexity" else v
    return cleaned


def upsert_product(name, **kwargs):
    if not name or not name.strip():
        return None

    conn = get_db()
    now = _now()
    cleaned = _clean_product_kwargs(kwargs)
    existing = conn.execute("SELECT id FROM products WHERE name=?", (name,)).fetchone()

    if existing:
        sets = ["last_updated=?"] + [f"{k}=?" for k in cleaned]
        vals = [now] + list(cleaned.values()) + [existing["id"]]
        conn.execute(f"UPDATE products SET {', '.join(sets)} WHERE id=?", vals)
        pid = existing["id"]
    else:
        cols = ["name", "first_seen", "last_updated"] + list(cleaned.keys())
        vals = [name, now, now] + list(cleaned.values())
        ph = ", ".join(["?"] * len(cols))
        conn.execute(f"INSERT INTO products ({', '.join(cols)}) VALUES ({ph})", vals)
        pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.commit()
    conn.close()
    return pid


# --- Painpoints ---

def upsert_painpoint(title, **kwargs):
    if not title or not title.strip():
        return None

    # Clamp severity to valid range
    severity = kwargs.get("severity", 5)
    if not isinstance(severity, int) or severity < 1 or severity > 10:
        severity = max(1, min(10, int(severity) if severity else 5))

    conn = get_db()
    now = _now()
    existing = conn.execute("SELECT id FROM painpoints WHERE title=?", (title,)).fetchone()

    if existing:
        conn.execute(
            "UPDATE painpoints SET signal_count = signal_count + 1, last_updated=? WHERE id=?",
            (now, existing["id"])
        )
        pid = existing["id"]
    else:
        conn.execute(
            "INSERT INTO painpoints (title, description, severity, first_seen, last_updated) VALUES (?,?,?,?,?)",
            (title, kwargs.get("description", "") or "", severity, now, now)
        )
        pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.commit()
    conn.close()

    # Link to category if provided (by slug or name) — accepts "category" or "domain"
    category = kwargs.get("category") or kwargs.get("domain")
    if category:
        link_painpoint_category(pid, category)

    return pid


def _link_entity_to_category(junction_table, fk_column, entity_id, category_slug_or_name):
    """Generic helper to link any entity to a category by slug or name."""
    if not entity_id or not category_slug_or_name:
        return False
    conn = get_db()
    cat = conn.execute(
        "SELECT id FROM categories WHERE slug=? OR name=?",
        (category_slug_or_name, category_slug_or_name)
    ).fetchone()
    if not cat:
        conn.close()
        return False
    try:
        conn.execute(
            f"INSERT INTO {junction_table} ({fk_column}, category_id) VALUES (?,?)",
            (entity_id, cat["id"])
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()
    return True


def link_painpoint_category(painpoint_id, category_slug_or_name):
    return _link_entity_to_category("painpoint_categories", "painpoint_id",
                                     painpoint_id, category_slug_or_name)


def link_product_category(product_id, category_slug_or_name):
    return _link_entity_to_category("product_categories", "product_id",
                                     product_id, category_slug_or_name)


def link_funding_category(round_id, category_slug_or_name):
    return _link_entity_to_category("funding_categories", "round_id",
                                     round_id, category_slug_or_name)


def propose_category(name, parent_slug=None):
    """LLM-proposed category. Goes into pending_categories until promoted."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM pending_categories WHERE name=?", (name,)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE pending_categories SET signal_count = signal_count + 1 WHERE id=?",
            (existing["id"],)
        )
    else:
        conn.execute(
            "INSERT INTO pending_categories (name, parent_slug, first_seen) VALUES (?,?,?)",
            (name, parent_slug, _now())
        )
    conn.commit()
    conn.close()


def get_ready_pending_categories(min_signals=3):
    """Categories with enough signals to be reviewed for promotion."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM pending_categories WHERE signal_count >= ? ORDER BY signal_count DESC",
        (min_signals,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def promote_pending_category(pending_id, parent_slug, description=""):
    """Promote a pending category to the real taxonomy."""
    conn = get_db()
    pending = conn.execute(
        "SELECT * FROM pending_categories WHERE id=?", (pending_id,)
    ).fetchone()
    if not pending:
        conn.close()
        return None

    parent = conn.execute("SELECT id FROM categories WHERE slug=?", (parent_slug,)).fetchone()
    parent_id = parent["id"] if parent else None

    name = pending["name"]
    slug = _slug(name)

    # Check for slug collision
    if conn.execute("SELECT id FROM categories WHERE slug=?", (slug,)).fetchone():
        conn.execute("DELETE FROM pending_categories WHERE id=?", (pending_id,))
        conn.commit()
        conn.close()
        return None

    conn.execute(
        "INSERT INTO categories (name, slug, parent_id, description, created_by, created_at) VALUES (?,?,?,?,?,?)",
        (name, slug, parent_id, description, "llm_promoted", _now())
    )
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute("DELETE FROM pending_categories WHERE id=?", (pending_id,))
    conn.commit()
    conn.close()
    return new_id


def reject_pending_category(pending_id):
    conn = get_db()
    conn.execute("DELETE FROM pending_categories WHERE id=?", (pending_id,))
    conn.commit()
    conn.close()


def rename_category(category_id, new_name, new_description=None):
    """Rename a category. Refuses to rename seed categories."""
    if not new_name or not new_name.strip():
        return False
    conn = get_db()
    row = conn.execute("SELECT created_by FROM categories WHERE id=?", (category_id,)).fetchone()
    if not row or row["created_by"] == "seed":
        conn.close()
        return False

    sets = ["name=?", "slug=?"]
    vals = [new_name, _slug(new_name)]
    if new_description is not None:
        sets.append("description=?")
        vals.append(new_description)
    vals.append(category_id)
    try:
        conn.execute(f"UPDATE categories SET {', '.join(sets)} WHERE id=?", vals)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def merge_categories(keep_id, merge_ids):
    """Merge duplicate categories: move junction links to keep_id, delete others.
    Refuses to merge seed categories. Validates IDs exist."""
    conn = get_db()
    try:
        # Validate keep_id exists
        if not conn.execute("SELECT 1 FROM categories WHERE id=?", (keep_id,)).fetchone():
            return False

        # Filter: exists, not seed, not self
        safe_merge_ids = []
        for mid in merge_ids:
            if mid == keep_id:
                continue
            row = conn.execute(
                "SELECT created_by FROM categories WHERE id=?", (mid,)
            ).fetchone()
            if row and row["created_by"] != "seed":
                safe_merge_ids.append(mid)

        for mid in safe_merge_ids:
            for table in ("product_categories", "painpoint_categories", "funding_categories"):
                conn.execute(
                    f"UPDATE OR IGNORE {table} SET category_id=? WHERE category_id=?",
                    (keep_id, mid)
                )
                conn.execute(f"DELETE FROM {table} WHERE category_id=?", (mid,))

            conn.execute("UPDATE categories SET parent_id=? WHERE parent_id=?", (keep_id, mid))
            conn.execute("DELETE FROM categories WHERE id=?", (mid,))

        conn.commit()
        return True
    finally:
        conn.close()


def add_quote(painpoint_id, text, source, source_url="", sentiment="negative", product_id=None):
    conn = get_db()
    # Avoid duplicate quotes
    existing = conn.execute(
        "SELECT id FROM quotes WHERE painpoint_id=? AND text=?", (painpoint_id, text)
    ).fetchone()
    if existing:
        conn.close()
        return existing["id"]
    conn.execute(
        "INSERT INTO quotes (painpoint_id, product_id, text, source, source_url, sentiment, scraped_at) VALUES (?,?,?,?,?,?,?)",
        (painpoint_id, product_id, text, source, source_url, sentiment, _now())
    )
    qid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return qid


# --- Funding ---

def _ensure_investor_with_conn(conn, name, investor_type="vc"):
    """Ensure investor exists using a shared connection (no open/close)."""
    existing = conn.execute("SELECT id FROM investors WHERE name=?", (name,)).fetchone()
    if existing:
        return existing["id"]
    conn.execute("INSERT INTO investors (name, type) VALUES (?,?)", (name, investor_type))
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def ensure_investor(name, investor_type="vc"):
    """Public single-call version that opens its own connection."""
    conn = get_db()
    iid = _ensure_investor_with_conn(conn, name, investor_type)
    conn.commit()
    conn.close()
    return iid


def _parse_amount_usd(amount_str):
    """Parse '$15M' or '$1.5B' into integer USD."""
    if not amount_str:
        return None
    import re
    amount_str = str(amount_str)
    m = re.match(r'\$(\d+(?:\.\d+)?)\s*([BMK])?', amount_str.upper())
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2) or ''
    if unit == 'B':
        return int(num * 1_000_000_000)
    elif unit == 'M':
        return int(num * 1_000_000)
    elif unit == 'K':
        return int(num * 1_000)
    return int(num)


def save_funding_round(company, amount="", round_type="", investor_names=None,
                       source_url="", valuation="", what_they_build="",
                       painpoint_solved="", why_funded="", product_id=None):
    """
    Save a funding round. Links to product and painpoints via junction tables.
    Uses a single connection throughout to avoid nested-writer lock errors.
    """
    conn = get_db()
    try:
        now = _now()
        existing = conn.execute(
            "SELECT id FROM funding_rounds WHERE product_id=? AND amount=? AND product_id IS NOT NULL",
            (product_id, amount)
        ).fetchone()
        if not existing and company:
            existing = conn.execute(
                "SELECT id FROM funding_rounds WHERE what_they_build=? AND amount=?",
                (what_they_build, amount)
            ).fetchone()
        if existing:
            return existing["id"]

        amount_usd = _parse_amount_usd(amount)
        conn.execute(
            """INSERT INTO funding_rounds (product_id, amount, amount_usd, valuation, round_type,
               what_they_build, why_funded, source_url, scraped_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            (product_id, amount, amount_usd, valuation, round_type,
             what_they_build, why_funded, source_url, now)
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Link investors (shared connection, no nesting)
        for inv_name in (investor_names or []):
            if not inv_name:
                continue
            inv_id = _ensure_investor_with_conn(conn, inv_name)
            try:
                conn.execute(
                    "INSERT INTO round_investors (round_id, investor_id) VALUES (?,?)",
                    (rid, inv_id)
                )
            except sqlite3.IntegrityError:
                pass

        # Link painpoint (upsert inline)
        if painpoint_solved:
            pp_title = painpoint_solved[:200]
            pp_row = conn.execute(
                "SELECT id FROM painpoints WHERE title=?", (pp_title,)
            ).fetchone()
            if pp_row:
                pp_id = pp_row["id"]
                conn.execute(
                    "UPDATE painpoints SET signal_count = signal_count + 1, last_updated=? WHERE id=?",
                    (now, pp_id)
                )
            else:
                conn.execute(
                    "INSERT INTO painpoints (title, description, severity, first_seen, last_updated) VALUES (?,?,?,?,?)",
                    (pp_title, painpoint_solved, 6, now, now)
                )
                pp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            try:
                conn.execute(
                    "INSERT INTO round_painpoints (round_id, painpoint_id) VALUES (?,?)",
                    (rid, pp_id)
                )
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        return rid
    finally:
        conn.close()


# --- Categories ---

# --- Deletion ---

def delete_painpoints(ids):
    """Delete painpoints by ID. CASCADE removes linked quotes and junctions."""
    conn = get_db()
    for pid in ids:
        conn.execute("DELETE FROM painpoints WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return len(ids)


def delete_products(ids):
    conn = get_db()
    for pid in ids:
        conn.execute("DELETE FROM products WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return len(ids)


def delete_categories(ids):
    """Delete categories. CASCADE handles junction tables."""
    conn = get_db()
    for cid in ids:
        # Only delete non-seed categories — don't let LLM delete the taxonomy
        row = conn.execute("SELECT created_by FROM categories WHERE id=?", (cid,)).fetchone()
        if row and row["created_by"] != "seed":
            conn.execute("DELETE FROM categories WHERE id=?", (cid,))
    conn.commit()
    conn.close()
    return len(ids)


def get_all_for_cleanup(table, limit=100):
    """Get all entries from a table for LLM review."""
    conn = get_db()
    if table == "painpoints":
        rows = conn.execute("""
            SELECT p.id, p.title, p.description, p.severity, p.signal_count,
                   GROUP_CONCAT(c.name) as categories, COUNT(DISTINCT q.id) as quote_count
            FROM painpoints p
            LEFT JOIN painpoint_categories pc ON pc.painpoint_id = p.id
            LEFT JOIN categories c ON pc.category_id = c.id
            LEFT JOIN quotes q ON q.painpoint_id = p.id
            GROUP BY p.id ORDER BY p.signal_count ASC LIMIT ?""", (limit,)).fetchall()
    elif table == "products":
        rows = conn.execute(
            "SELECT id, name, description, status, tech_complexity FROM products ORDER BY last_updated ASC LIMIT ?",
            (limit,)).fetchall()
    elif table == "categories":
        rows = conn.execute("""
            SELECT c.id, c.name, c.description, c.created_by,
                   COUNT(DISTINCT pc.painpoint_id) as painpoint_count
            FROM categories c
            LEFT JOIN painpoint_categories pc ON pc.category_id = c.id
            WHERE c.created_by != 'seed'
            GROUP BY c.id ORDER BY painpoint_count ASC LIMIT ?""", (limit,)).fetchall()
    else:
        conn.close()
        return []
    conn.close()
    return [dict(r) for r in rows]


def merge_painpoints(keep_id, merge_ids):
    """Merge duplicate painpoints: move quotes, junctions, bump signal_count, delete dupes.
    Validates all IDs exist before touching anything."""
    conn = get_db()
    try:
        # Validate keep_id exists
        if not conn.execute("SELECT 1 FROM painpoints WHERE id=?", (keep_id,)).fetchone():
            return False

        # Filter merge_ids to only valid, non-self IDs
        valid_merge = []
        for mid in merge_ids:
            if mid == keep_id:
                continue
            if conn.execute("SELECT 1 FROM painpoints WHERE id=?", (mid,)).fetchone():
                valid_merge.append(mid)

        if not valid_merge:
            return False

        for mid in valid_merge:
            # Move quotes
            conn.execute(
                "UPDATE quotes SET painpoint_id=? WHERE painpoint_id=?",
                (keep_id, mid)
            )
            # Move junction links (painpoint_categories, product_painpoints, round_painpoints)
            for table in ("painpoint_categories", "product_painpoints", "round_painpoints"):
                conn.execute(
                    f"UPDATE OR IGNORE {table} SET painpoint_id=? WHERE painpoint_id=?",
                    (keep_id, mid)
                )
                conn.execute(f"DELETE FROM {table} WHERE painpoint_id=?", (mid,))
            # Bump signal count
            conn.execute(
                "UPDATE painpoints SET signal_count = signal_count + "
                "(SELECT signal_count FROM painpoints WHERE id=?) WHERE id=?",
                (mid, keep_id)
            )
            conn.execute("DELETE FROM painpoints WHERE id=?", (mid,))

        conn.commit()
        return True
    finally:
        conn.close()


# --- Queries ---

def get_top_painpoints(limit=10):
    conn = get_db()
    rows = conn.execute("""
        SELECT p.*, GROUP_CONCAT(DISTINCT c.name) as categories,
               COUNT(DISTINCT q.id) as quote_count
        FROM painpoints p
        LEFT JOIN painpoint_categories pc ON pc.painpoint_id = p.id
        LEFT JOIN categories c ON pc.category_id = c.id
        LEFT JOIN quotes q ON q.painpoint_id = p.id
        GROUP BY p.id
        ORDER BY p.signal_count DESC, p.severity DESC
        LIMIT ?""", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_category_list_flat():
    """Return categories as 'Parent > Child' strings for LLM prompts."""
    conn = get_db()
    rows = conn.execute("""
        SELECT c.name as child, c.slug, p.name as parent, c.description
        FROM categories c
        LEFT JOIN categories p ON c.parent_id = p.id
        WHERE c.parent_id IS NOT NULL
        ORDER BY p.name, c.name
    """).fetchall()
    conn.close()
    return [
        {"path": f"{r['parent']} > {r['child']}", "slug": r["slug"], "description": r["description"]}
        for r in rows
    ]


def get_facts_by_category(slug, include_descendants=True):
    """Get products, painpoints, funding in a category (and descendants).
    Painpoints include a `categories` field (comma-joined) for consistency with
    get_top_painpoints()."""
    conn = get_db()

    if include_descendants:
        cat_ids = conn.execute("""
            WITH RECURSIVE tree(id) AS (
                SELECT id FROM categories WHERE slug=?
                UNION SELECT c.id FROM categories c JOIN tree t ON c.parent_id=t.id
            )
            SELECT id FROM tree
        """, (slug,)).fetchall()
    else:
        cat_ids = conn.execute("SELECT id FROM categories WHERE slug=?", (slug,)).fetchall()

    ids = [r["id"] for r in cat_ids]
    if not ids:
        conn.close()
        return {"products": [], "painpoints": [], "funding": []}

    placeholders = ",".join("?" * len(ids))

    products = conn.execute(f"""
        SELECT DISTINCT p.* FROM products p
        JOIN product_categories pc ON pc.product_id=p.id
        WHERE pc.category_id IN ({placeholders})
    """, ids).fetchall()

    painpoints = conn.execute(f"""
        SELECT p.*, GROUP_CONCAT(DISTINCT cat.name) as categories,
               COUNT(DISTINCT q.id) as quote_count
        FROM painpoints p
        JOIN painpoint_categories pc ON pc.painpoint_id=p.id
        LEFT JOIN categories cat ON cat.id=pc.category_id
        LEFT JOIN quotes q ON q.painpoint_id=p.id
        WHERE pc.category_id IN ({placeholders})
        GROUP BY p.id
        ORDER BY p.signal_count DESC
    """, ids).fetchall()

    funding = conn.execute(f"""
        SELECT DISTINCT f.*, p.name as product_name FROM funding_rounds f
        LEFT JOIN products p ON f.product_id=p.id
        JOIN funding_categories fc ON fc.round_id=f.id
        WHERE fc.category_id IN ({placeholders})
        ORDER BY f.amount_usd DESC NULLS LAST
    """, ids).fetchall()

    conn.close()
    return {
        "products": [dict(r) for r in products],
        "painpoints": [dict(r) for r in painpoints],
        "funding": [dict(r) for r in funding],
    }


def get_quotes_for_painpoints(painpoint_ids, per_painpoint=1):
    """Batch-fetch quotes for multiple painpoints. Returns dict of pp_id -> [quote_texts]."""
    if not painpoint_ids:
        return {}
    conn = get_db()
    placeholders = ",".join("?" * len(painpoint_ids))
    rows = conn.execute(
        f"SELECT painpoint_id, text FROM quotes WHERE painpoint_id IN ({placeholders}) ORDER BY id",
        painpoint_ids
    ).fetchall()
    conn.close()

    result = {}
    for r in rows:
        pid = r["painpoint_id"]
        if pid not in result:
            result[pid] = []
        if len(result[pid]) < per_painpoint:
            result[pid].append(r["text"])
    return result


def get_recent_funding(limit=20):
    conn = get_db()
    rows = conn.execute("""
        SELECT fr.*, p.name as product_name
        FROM funding_rounds fr
        LEFT JOIN products p ON fr.product_id = p.id
        ORDER BY fr.amount_usd DESC NULLS LAST, fr.scraped_at DESC
        LIMIT ?""", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_db()
    stats = {}
    for table in ["runs", "signals", "products", "painpoints", "funding_rounds",
                   "quotes", "categories", "investors", "pending_categories"]:
        stats[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    stats["signals_kept"] = conn.execute("SELECT COUNT(*) FROM signals WHERE kept=1").fetchone()[0]
    conn.close()
    return stats


# --- Market views ---

def get_market_gaps(limit=20):
    """Categories with lots of pain but few products = underserved markets."""
    conn = get_db()
    rows = conn.execute("""
        SELECT
            c.name AS category,
            c.slug,
            COUNT(DISTINCT pp.id) AS painpoint_count,
            COUNT(DISTINCT p.id) AS product_count,
            ROUND(AVG(pp.severity), 1) AS avg_severity,
            SUM(pp.signal_count) AS total_signals,
            COALESCE(SUM(fr.amount_usd), 0) AS funding_usd,
            CASE
                WHEN COUNT(DISTINCT p.id) = 0 THEN 999
                ELSE ROUND(CAST(COUNT(DISTINCT pp.id) AS REAL) / COUNT(DISTINCT p.id), 1)
            END AS pain_to_product_ratio
        FROM categories c
        LEFT JOIN painpoint_categories pc ON pc.category_id = c.id
        LEFT JOIN painpoints pp ON pp.id = pc.painpoint_id
        LEFT JOIN product_categories prc ON prc.category_id = c.id
        LEFT JOIN products p ON p.id = prc.product_id
        LEFT JOIN funding_categories fc ON fc.category_id = c.id
        LEFT JOIN funding_rounds fr ON fr.id = fc.round_id
        WHERE c.parent_id IS NOT NULL
        GROUP BY c.id
        HAVING painpoint_count > 0
        ORDER BY pain_to_product_ratio DESC, total_signals DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_product_gaps(limit=20):
    """Products with effectiveness scores and gap descriptions.
    Returns products that fail_at or partially solve painpoints, ranked by gap severity."""
    conn = get_db()
    rows = conn.execute("""
        SELECT
            p.name AS product,
            pp.title AS painpoint,
            pp.severity AS painpoint_severity,
            ppp.relationship,
            ppp.effectiveness,
            ppp.gap_description,
            ppp.gap_type,
            ppp.notes,
            COUNT(DISTINCT q.id) AS negative_quotes
        FROM product_painpoints ppp
        JOIN products p ON p.id = ppp.product_id
        JOIN painpoints pp ON pp.id = ppp.painpoint_id
        LEFT JOIN quotes q ON q.painpoint_id = pp.id AND q.sentiment = 'negative'
        WHERE ppp.relationship IN ('fails_at', 'partial')
        GROUP BY ppp.product_id, ppp.painpoint_id
        ORDER BY ppp.effectiveness ASC NULLS FIRST, pp.severity DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_hot_categories(limit=10):
    """Categories with the most recent activity (signals + funding + new products)."""
    conn = get_db()
    rows = conn.execute("""
        SELECT
            c.name AS category,
            c.slug,
            COUNT(DISTINCT pp.id) AS painpoints,
            COUNT(DISTINCT p.id) AS products,
            SUM(pp.signal_count) AS total_signals,
            COALESCE(SUM(fr.amount_usd), 0) AS total_funding_usd,
            COUNT(DISTINCT fr.id) AS funding_rounds
        FROM categories c
        LEFT JOIN painpoint_categories pc ON pc.category_id = c.id
        LEFT JOIN painpoints pp ON pp.id = pc.painpoint_id
        LEFT JOIN product_categories prc ON prc.category_id = c.id
        LEFT JOIN products p ON p.id = prc.product_id
        LEFT JOIN funding_categories fc ON fc.category_id = c.id
        LEFT JOIN funding_rounds fr ON fr.id = fc.round_id
        WHERE c.parent_id IS NOT NULL
        GROUP BY c.id
        HAVING total_signals > 0 OR total_funding_usd > 0
        ORDER BY total_signals DESC, total_funding_usd DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def link_product_painpoint(product_name, painpoint_title, relationship="addresses",
                           effectiveness=None, gap_description="", gap_type="", notes=""):
    """Link a product to a painpoint with effectiveness score and gap analysis."""
    conn = get_db()
    prod = conn.execute("SELECT id FROM products WHERE name=?", (product_name,)).fetchone()
    pp = conn.execute("SELECT id FROM painpoints WHERE title=?", (painpoint_title,)).fetchone()
    if not prod or not pp:
        conn.close()
        return False

    # Clamp effectiveness
    if effectiveness is not None:
        effectiveness = max(1, min(10, int(effectiveness)))

    try:
        conn.execute(
            """INSERT INTO product_painpoints
               (product_id, painpoint_id, relationship, effectiveness, gap_description, gap_type, notes)
               VALUES (?,?,?,?,?,?,?)""",
            (prod["id"], pp["id"], relationship, effectiveness, gap_description, gap_type, notes)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.execute(
            """UPDATE product_painpoints
               SET relationship=?, effectiveness=?, gap_description=?, gap_type=?, notes=?
               WHERE product_id=? AND painpoint_id=?""",
            (relationship, effectiveness, gap_description, gap_type, notes, prod["id"], pp["id"])
        )
        conn.commit()
    conn.close()
    return True


def run_sql(query, params=None):
    """Execute arbitrary read-only SQL. For LLM research queries.
    Only allows SELECT statements."""
    query = query.strip()
    if not query.upper().startswith("SELECT"):
        return {"error": "Only SELECT queries allowed"}
    conn = get_db()
    try:
        rows = conn.execute(query, params or ()).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


# === TAXONOMY ===

TAXONOMY_FILE = Path(__file__).parent / "taxonomy.yaml"


def _slug(name):
    return name.lower().replace(" ", "-").replace("&", "and").replace("/", "-")


def seed_taxonomy():
    """Populate the categories table from taxonomy.yaml if empty."""
    conn = get_db()
    if conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0] > 0:
        conn.close()
        return

    if not TAXONOMY_FILE.exists():
        conn.close()
        return

    taxonomy = yaml.safe_load(TAXONOMY_FILE.read_text())
    now = _now()

    for parent_name, parent_data in taxonomy.items():
        conn.execute(
            "INSERT INTO categories (name, slug, parent_id, description, created_by, created_at) VALUES (?,?,?,?,?,?)",
            (parent_name, _slug(parent_name), None, parent_data.get("desc", ""), "seed", now)
        )
        parent_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for child_name, child_desc in parent_data.get("children", {}).items():
            conn.execute(
                "INSERT INTO categories (name, slug, parent_id, description, created_by, created_at) VALUES (?,?,?,?,?,?)",
                (child_name, _slug(child_name), parent_id, child_desc, "seed", now)
            )

    conn.commit()
    conn.close()


# Initialize on import
init_db()
seed_taxonomy()


if __name__ == "__main__":
    stats = get_stats()
    print("Database stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
