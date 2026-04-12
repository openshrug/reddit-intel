import json
import logging
import sqlite3

from . import get_db, _now
from .categories import get_category_id_by_name

log = logging.getLogger(__name__)

UNCATEGORIZED_NAME = "Uncategorized"


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


# ===========================================================================
# Painpoint ingest pipeline (see docs/PAINPOINT_INGEST_PLAN.md)
# ===========================================================================


def add_pending_source(pending_id, post_id, comment_id=None, *, conn=None):
    """Append an *additional* source to a pending painpoint (§7.5).

    The first source of a pending pp is stored in the legacy
    pending_painpoints.(post_id, comment_id) columns; additional sources
    go into the pending_painpoint_sources junction. Single-source pendings
    have zero rows here. Multi-source pendings have N-1.
    """
    own_conn = conn is None
    conn = conn or get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO pending_painpoint_sources "
            "(pending_painpoint_id, post_id, comment_id) VALUES (?,?,?)",
            (pending_id, post_id, comment_id),
        )
        if own_conn:
            conn.commit()
    finally:
        if own_conn:
            conn.close()


def get_uncategorized_id(conn=None):
    """Resolve the sentinel Uncategorized category id (§7.6)."""
    own_conn = conn is None
    conn = conn or get_db()
    try:
        row = conn.execute(
            "SELECT id FROM categories WHERE name = ?", (UNCATEGORIZED_NAME,)
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "Uncategorized sentinel category missing — db.init_db() not run?"
            )
        return row["id"]
    finally:
        if own_conn:
            conn.close()


def merge_painpoints(conn, survivor_id, loser_id):
    """Merge two merged painpoints into one.

    Mechanical contract:
      1. Repoint loser's painpoint_sources rows at the survivor.
      2. survivor.signal_count += loser.signal_count
      3. survivor.first_seen = min(survivor.first_seen, loser.first_seen)
      4. survivor.last_updated = now()
      5. DELETE the loser row (cascade removes loser's embedding from
         painpoint_vec via the row delete).
      7. survivor.category_id is **unchanged** — it keeps whatever category
         it was in. Deliberate-but-arbitrary.
      8. Logged via `logging.info` (NOT to category_events).

    Caller is responsible for the merge_lock; this function does not
    open/close its own transaction.
    """
    survivor = conn.execute(
        "SELECT id, signal_count, first_seen FROM painpoints WHERE id = ?",
        (survivor_id,),
    ).fetchone()
    loser = conn.execute(
        "SELECT id, signal_count, first_seen FROM painpoints WHERE id = ?",
        (loser_id,),
    ).fetchone()
    if survivor is None or loser is None:
        raise ValueError(
            f"merge_painpoints: missing row (survivor={survivor_id}, loser={loser_id})"
        )
    if survivor_id == loser_id:
        return survivor_id

    # 1. Repoint sources. INSERT OR IGNORE because the survivor may already
    # cite some of the same pending pps (idempotent re-merge).
    pending_ids = [
        r["pending_painpoint_id"]
        for r in conn.execute(
            "SELECT pending_painpoint_id FROM painpoint_sources WHERE painpoint_id = ?",
            (loser_id,),
        ).fetchall()
    ]
    for pid in pending_ids:
        try:
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                "VALUES (?, ?)",
                (survivor_id, pid),
            )
        except sqlite3.IntegrityError:
            pass
    conn.execute(
        "DELETE FROM painpoint_sources WHERE painpoint_id = ?", (loser_id,)
    )

    # 2-4. Update survivor's stats.
    new_signal = (survivor["signal_count"] or 0) + (loser["signal_count"] or 0)
    new_first_seen = min(survivor["first_seen"], loser["first_seen"])
    conn.execute(
        "UPDATE painpoints SET signal_count = ?, first_seen = ?, last_updated = ? "
        "WHERE id = ?",
        (new_signal, new_first_seen, _now(), survivor_id),
    )

    # 5. Delete the loser (and its embedding from painpoint_vec).
    conn.execute("DELETE FROM painpoint_vec WHERE rowid = ?", (loser_id,))
    conn.execute("DELETE FROM painpoints WHERE id = ?", (loser_id,))

    # 8. Audit log to file (not to category_events).
    log.info(
        "merge_painpoints survivor=%s loser=%s pre_signal_survivor=%s "
        "pre_signal_loser=%s post_signal=%s",
        survivor_id, loser_id,
        survivor["signal_count"], loser["signal_count"], new_signal,
    )

    return survivor_id



def _create_painpoint_from_pending(conn, pending_id, embedding=None, embedder=None):
    """Create a new merged painpoint from a pending painpoint.

    Uses the LLM-proposed category from extraction time if it resolved
    to a real category in the taxonomy; if embedding-based category
    assignment is available, uses find_best_category instead. Falls back
    to the Uncategorized sentinel if no good match.
    """
    from .embeddings import (
        find_best_category,
        store_painpoint_embedding,
        update_category_embedding,
    )

    pending = conn.execute(
        "SELECT id, title, description, severity, category_id "
        "FROM pending_painpoints WHERE id = ?",
        (pending_id,),
    ).fetchone()
    if pending is None:
        raise ValueError(f"pending_painpoint {pending_id} not found")

    # Category assignment: prefer embedding-based if we have an embedding,
    # else fall back to the LLM-proposed category from extraction time.
    if embedding is not None:
        category_id = find_best_category(conn, embedding, embedder=embedder)
    else:
        category_id = pending["category_id"]
        if category_id is None:
            category_id = get_uncategorized_id(conn=conn)
            log.debug(
                "pending_painpoint %s has no resolved category -- using Uncategorized",
                pending_id,
            )

    now = _now()
    conn.execute(
        "INSERT INTO painpoints "
        "(title, description, severity, signal_count, category_id, first_seen, last_updated) "
        "VALUES (?, ?, ?, 1, ?, ?, ?)",
        (pending["title"], pending["description"] or "", pending["severity"],
         category_id, now, now),
    )
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Link this single pending pp as the painpoint's first source row.
    conn.execute(
        "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) VALUES (?, ?)",
        (new_id, pending_id),
    )

    # Store embedding so subsequent painpoints can match against it.
    if embedding is not None:
        store_painpoint_embedding(conn, new_id, embedding)
        update_category_embedding(conn, category_id)

    return new_id


def _link_pending_to_painpoint(conn, painpoint_id, pending_id):
    """Link a pending pp into an existing merged painpoint and bump signal_count."""
    try:
        conn.execute(
            "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) VALUES (?, ?)",
            (painpoint_id, pending_id),
        )
        conn.execute(
            "UPDATE painpoints SET signal_count = signal_count + 1, last_updated = ? "
            "WHERE id = ?",
            (_now(), painpoint_id),
        )
    except sqlite3.IntegrityError:
        # Already linked — idempotent.
        pass


def promote_pending(pending_id, *, embedder=None, now=None):
    """End-to-end promotion of one pending painpoint into the merged table.

    Steps:
      1. Compute relevance. If below threshold, hard-delete and return None.
      2. Compute embedding (calls OpenAI API, outside the lock).
      3. Acquire the merge lock.
      4. Inside the lock: cosine similarity search against all existing
         painpoint embeddings. If above MERGE_COSINE_THRESHOLD → link.
         Otherwise → create new painpoint in the best-matching category.

    Returns the painpoints.id the pending pp was attached to, or None if
    the pending row was dropped for low relevance.
    """
    from .relevance import compute_pending_relevance, MIN_RELEVANCE_TO_PROMOTE
    from .embeddings import (
        MERGE_COSINE_THRESHOLD,
        find_most_similar_painpoint,
    )
    from .locks import merge_lock

    if embedder is None:
        from .embeddings import OpenAIEmbedder
        embedder = OpenAIEmbedder()

    # Step 1 — relevance check + drop (outside the lock).
    conn = get_db()
    try:
        relevance = compute_pending_relevance(pending_id, conn=conn, now=now)
        if relevance < MIN_RELEVANCE_TO_PROMOTE:
            conn.execute(
                "DELETE FROM pending_painpoints WHERE id = ?", (pending_id,)
            )
            conn.commit()
            log.info(
                "promote_pending dropped pending_id=%s relevance=%.4f < %.4f",
                pending_id, relevance, MIN_RELEVANCE_TO_PROMOTE,
            )
            return None
    finally:
        conn.close()

    # Step 2 — compute embedding (outside the lock, may call OpenAI).
    conn = get_db()
    try:
        pending = conn.execute(
            "SELECT title, description FROM pending_painpoints WHERE id = ?",
            (pending_id,),
        ).fetchone()
    finally:
        conn.close()

    text = f"{pending['title']} {pending['description'] or ''}".strip()
    embedding = embedder.embed(text)

    # Steps 3-4 — under the lock: cosine similarity → link or create.
    conn = get_db()
    try:
        with merge_lock(conn, timeout=30):
            result = find_most_similar_painpoint(conn, embedding)
            if result is not None:
                best_id, cosine_sim = result
                if cosine_sim >= MERGE_COSINE_THRESHOLD:
                    _link_pending_to_painpoint(conn, best_id, pending_id)
                    return best_id

            # No match — create new painpoint in the best category
            new_id = _create_painpoint_from_pending(
                conn, pending_id, embedding=embedding, embedder=embedder,
            )
            return new_id
    finally:
        conn.close()
