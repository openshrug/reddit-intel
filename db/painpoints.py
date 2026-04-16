import logging
import sqlite3

from . import _now, get_db, in_clause_placeholders, uncategorized_id
from .categories import get_category_id_by_name
from .embeddings import (
    MERGE_COSINE_THRESHOLD,
    OpenAIEmbedder,
    add_member_to_centroid,
    find_best_category,
    find_most_similar_painpoint,
    store_painpoint_embedding,
    update_category_embedding,
)
from .locks import merge_lock

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pending painpoints (immutable, append-only)
# ---------------------------------------------------------------------------

def save_pending_painpoint(post_id, title, *, comment_id=None,
                           category_name=None, description=None,
                           quoted_text=None, severity):
    """Append a single LLM-extracted painpoint observation.

    The category_name is resolved to a category_id at insert time.
    Returns the new pending_painpoints.id, or None if FK validation
    fails (post_id doesn't exist) — silent skip with warning, matching
    save_pending_painpoints_batch's behavior on hallucinated IDs.
    """
    if severity is None:
        raise ValueError("severity is required (1-10) — no silent default")
    severity = max(1, min(10, int(severity)))
    category_id = get_category_id_by_name(category_name)

    conn = get_db()
    try:
        # FK validation parity with the batch path — skip silently with
        # a warning instead of raising IntegrityError when the LLM
        # hallucinates a post_id or comment_id not in the input.
        if not conn.execute(
            "SELECT 1 FROM posts WHERE id = ?", (post_id,)
        ).fetchone():
            log.warning(
                "save_pending: post_id %s not found — skipping this painpoint "
                "(LLM hallucinated an ID?)", post_id,
            )
            return None

        if comment_id is not None and not conn.execute(
            "SELECT 1 FROM comments WHERE id = ?", (comment_id,)
        ).fetchone():
            log.warning(
                "save_pending: comment_id %s not found, setting to NULL",
                comment_id,
            )
            comment_id = None

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
        return pp_id
    finally:
        conn.close()


def save_pending_painpoints_batch(items):
    """Batch-insert multiple pending painpoints in one transaction.

    Validates post_id and comment_id existence in TWO bulk SELECTs
    instead of 2N (was a per-item round-trip). The LLM can hallucinate
    IDs not in the input batch — those rows are skipped (post) or have
    comment_id NULLed.

    Args:
        items: list of dicts, each with keys matching save_pending_painpoint
               params (post_id, title, comment_id, category_name, etc.).

    Returns:
        List of new pending_painpoints ids.
    """
    if not items:
        return []

    conn = get_db()
    now = _now()
    ids = []

    try:
        # Bulk-validate post_ids in one query.
        wanted_post_ids = {item["post_id"] for item in items}
        existing_post_ids = {
            r[0]
            for r in conn.execute(
                f"SELECT id FROM posts WHERE id IN "
                f"({in_clause_placeholders(len(wanted_post_ids))})",
                list(wanted_post_ids),
            ).fetchall()
        }
        # Bulk-validate comment_ids in one query.
        wanted_comment_ids = {
            item["comment_id"] for item in items
            if item.get("comment_id") is not None
        }
        existing_comment_ids = set()
        if wanted_comment_ids:
            existing_comment_ids = {
                r[0]
                for r in conn.execute(
                    f"SELECT id FROM comments WHERE id IN "
                    f"({in_clause_placeholders(len(wanted_comment_ids))})",
                    list(wanted_comment_ids),
                ).fetchall()
            }

        for item in items:
            post_id = item["post_id"]
            if post_id not in existing_post_ids:
                log.warning(
                    "save_pending: post_id %s not found — skipping this painpoint "
                    "(LLM hallucinated an ID?)", post_id,
                )
                continue

            sev = item.get("severity")
            if sev is None:
                raise ValueError(
                    f"severity missing from item for post_id={post_id} "
                    "— extraction must emit 1-10"
                )
            severity = max(1, min(10, int(sev)))
            category_id = get_category_id_by_name(item.get("category_name"))

            comment_id = item.get("comment_id")
            if comment_id is not None and comment_id not in existing_comment_ids:
                log.warning("save_pending: comment_id %s not found, setting to NULL",
                            comment_id)
                comment_id = None

            conn.execute(
                """INSERT INTO pending_painpoints
                   (post_id, comment_id, category_id, title, description,
                    quoted_text, severity, extracted_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    post_id,
                    comment_id,
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
    except Exception:
        conn.rollback()
        raise
    finally:
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


# ===========================================================================
# Painpoint ingest pipeline
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
    """Backward-compat shim — delegates to db.uncategorized_id().
    Kept because tests / external callers may import this name."""
    own_conn = conn is None
    conn = conn or get_db()
    try:
        return uncategorized_id(conn)
    finally:
        if own_conn:
            conn.close()


def _create_painpoint_from_pending(conn, pending_id, embedding, embedder=None):
    """Create a new merged painpoint from a pending painpoint.

    `embedding` is required (no default). The previous "no embedding"
    path created painpoints that never got stored in painpoint_vec —
    invisible to find_most_similar_painpoint forever, so every future
    promote of the same pain would create a duplicate. Production
    always supplied an embedding, so this branch was dead code with a
    silent-corruption footgun. Now made explicit.

    Uses find_best_category(conn, embedding) to assign the painpoint to
    the best-matching category by cosine similarity; falls back to
    Uncategorized internally if nothing scores above
    CATEGORY_COSINE_THRESHOLD.
    """
    if embedding is None:
        raise ValueError(
            "_create_painpoint_from_pending requires an embedding — "
            "without one the new painpoint would be invisible to "
            "find_most_similar_painpoint and every future promote of "
            "the same pain would create a duplicate"
        )

    pending = conn.execute(
        "SELECT id, title, description, severity FROM pending_painpoints WHERE id = ?",
        (pending_id,),
    ).fetchone()
    if pending is None:
        raise ValueError(f"pending_painpoint {pending_id} not found")

    category_id = find_best_category(conn, embedding, embedder=embedder)

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

    # Store embedding + increment the category's cached member sum so
    # the next update_category_embedding is O(1). Then refresh the
    # blended category_vec so find_best_category sees the new state.
    store_painpoint_embedding(conn, new_id, embedding)
    add_member_to_centroid(conn, category_id, embedding)
    update_category_embedding(conn, category_id)

    return new_id


def _link_pending_to_painpoint(conn, painpoint_id, pending_id):
    """Link a pending pp into an existing merged painpoint and bump
    signal_count.

    NOTE: does NOT call update_category_embedding. Linking adds a new
    PENDING source to an existing painpoint — the painpoint's category
    membership and its own embedding don't change, so the category
    centroid (mean of member painpoint embeddings) is unchanged.
    Re-computing it after every link was O(N²) wasted work in batch
    promote runs. The centroid only shifts when:
      - a painpoint is created (handled in _create_painpoint_from_pending)
      - a painpoint moves between categories (handled in apply_*_event)
      - merge_painpoints retires a painpoint (handled there)
    """
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
        return


def promote_pending(pending_id, *, embedder=None, embedding=None, now=None):
    """End-to-end promotion of one pending painpoint into the merged table.

    Steps:
      1. Compute embedding (calls OpenAI API, outside the lock) — unless
         a pre-computed `embedding` was passed in (batch path from
         promoter.run_once).
      2. Acquire the merge lock.
      3. Inside the lock: cosine similarity search against all existing
         painpoint embeddings. If above MERGE_COSINE_THRESHOLD → link.
         Otherwise → create new painpoint in the best-matching category.

    Returns the painpoints.id the pending pp was attached to.
    """
    del now  # accepted for API compatibility
    if embedder is None:
        embedder = OpenAIEmbedder()

    if embedding is None:
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

    # Under the lock: cosine similarity → link or create.
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
