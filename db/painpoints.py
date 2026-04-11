import sqlite3

from . import get_db, _now
from .categories import get_category_id_by_name


# ---------------------------------------------------------------------------
# Pending painpoints (immutable, append-only)
# ---------------------------------------------------------------------------

def save_pending_painpoint(post_id, title, *, comment_id=None,
                           category_name=None, description=None,
                           quoted_text=None, severity=5):
    """Append a single LLM-extracted painpoint observation.

    The category_name is resolved to a category_id at insert time.
    Returns the new pending_painpoints.id.
    """
    severity = max(1, min(10, int(severity or 5)))
    category_id = get_category_id_by_name(category_name)

    conn = get_db()
    conn.execute(
        """INSERT INTO pending_painpoints
           (post_id, comment_id, category_id, title, description,
            quoted_text, severity, extracted_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (post_id, comment_id, category_id, title, description,
         quoted_text, severity, _now()),
    )
    pp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return pp_id


def save_pending_painpoints_batch(items):
    """Batch-insert multiple pending painpoints in one transaction.

    Args:
        items: list of dicts, each with keys matching save_pending_painpoint
               params (post_id, title, comment_id, category_name, etc.).

    Returns:
        List of new pending_painpoints ids.
    """
    conn = get_db()
    now = _now()
    ids = []

    for item in items:
        severity = max(1, min(10, int(item.get("severity", 5) or 5)))
        category_id = get_category_id_by_name(item.get("category_name"))

        conn.execute(
            """INSERT INTO pending_painpoints
               (post_id, comment_id, category_id, title, description,
                quoted_text, severity, extracted_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                item["post_id"],
                item.get("comment_id"),
                category_id,
                item["title"],
                item.get("description"),
                item.get("quoted_text"),
                severity,
                now,
            ),
        )
        ids.append(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    conn.commit()
    conn.close()
    return ids


def get_unmerged_pending():
    """Return pending painpoints not yet linked to any merged painpoint."""
    conn = get_db()
    rows = conn.execute("""
        SELECT pp.*, p.subreddit, p.permalink AS post_permalink
        FROM pending_painpoints pp
        JOIN posts p ON p.id = pp.post_id
        LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id
        WHERE ps.painpoint_id IS NULL
        ORDER BY pp.extracted_at
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Merged painpoints
# ---------------------------------------------------------------------------

def upsert_painpoint(title, *, description=None, severity=5, category_id=None):
    """Create or update a merged painpoint.

    If a painpoint with this exact title already exists, bumps signal_count
    and updates last_updated. Otherwise creates a new row.

    Returns the painpoint id.
    """
    severity = max(1, min(10, int(severity or 5)))
    conn = get_db()
    now = _now()

    existing = conn.execute(
        "SELECT id FROM painpoints WHERE title = ?", (title,)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE painpoints SET signal_count = signal_count + 1, last_updated = ? WHERE id = ?",
            (now, existing["id"]),
        )
        pid = existing["id"]
    else:
        conn.execute(
            "INSERT INTO painpoints (title, description, severity, category_id, first_seen, last_updated) "
            "VALUES (?,?,?,?,?,?)",
            (title, description or "", severity, category_id, now, now),
        )
        pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.commit()
    conn.close()
    return pid


def link_painpoint_source(painpoint_id, pending_painpoint_id):
    """Link a merged painpoint to a pending (raw) observation."""
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) VALUES (?,?)",
            (painpoint_id, pending_painpoint_id),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()


def merge_pending_into_painpoint(pending_ids, title, *, description=None,
                                 severity=5, category_id=None):
    """Full merge operation: create/update merged painpoint and link sources.

    Args:
        pending_ids: list of pending_painpoints.id to fold into one painpoint.
        title: merged painpoint title.
        description: merged description.
        severity: merged severity (1-10).
        category_id: single category for this merged painpoint.

    Returns:
        The merged painpoint id.
    """
    conn = get_db()
    now = _now()
    severity = max(1, min(10, int(severity or 5)))

    existing = conn.execute(
        "SELECT id, signal_count FROM painpoints WHERE title = ?", (title,)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE painpoints SET signal_count = signal_count + ?, last_updated = ? WHERE id = ?",
            (len(pending_ids), now, existing["id"]),
        )
        painpoint_id = existing["id"]
    else:
        conn.execute(
            "INSERT INTO painpoints (title, description, severity, signal_count, category_id, "
            "first_seen, last_updated) VALUES (?,?,?,?,?,?,?)",
            (title, description or "", severity, len(pending_ids), category_id, now, now),
        )
        painpoint_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    for pid in pending_ids:
        try:
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) VALUES (?,?)",
                (painpoint_id, pid),
            )
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    return painpoint_id
