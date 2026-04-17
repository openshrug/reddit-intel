import logging
import sqlite3

from . import _now, get_db, in_clause_placeholders, uncategorized_id
from .categories import get_category_id_by_name
from .embeddings import (
    MERGE_COSINE_THRESHOLD,
    PENDING_MERGE_THRESHOLD,
    OpenAIEmbedder,
    add_member_to_centroid,
    find_best_category,
    find_most_similar_painpoint,
    find_most_similar_pending,
    store_painpoint_embedding,
    store_pending_painpoint_embedding,
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


def save_pending_painpoints_batch(items, *, embedder=None):
    """Batch-insert multiple pending painpoints in one transaction.

    Validates post_id and comment_id existence in TWO bulk SELECTs
    instead of 2N (was a per-item round-trip). The LLM can hallucinate
    IDs not in the input batch — those rows are skipped (post) or have
    comment_id NULLed.

    When `embedder` is provided, each item's `title + description` is
    embedded (one batched API call for the whole list) and checked
    against `pending_painpoint_vec`. Near-duplicates (cosine ≥
    PENDING_MERGE_THRESHOLD) are collapsed onto the existing pending via
    `pending_painpoint_sources` instead of creating another near-copy
    row. Duplicates are also detected within the current batch (earlier
    items in the loop populate the vec table, so later items see them).

    When `embedder` is None (legacy/test callers), the dedup path is
    skipped — preserves exact prior behaviour.

    Args:
        items: list of dicts, each with keys matching save_pending_painpoint
               params (post_id, title, comment_id, category_name, etc.).
        embedder: optional `OpenAIEmbedder` / `FakeEmbedder`. Pass one to
                  enable pending-stage dedup.

    Returns:
        List of pending_painpoint ids, one per input item. Duplicates
        return the id of the pending they were merged INTO (so callers
        that track "which pending this row became" still get a valid id).
        Skipped items (FK hallucination) contribute no entry.
    """
    if not items:
        return []

    conn = get_db()
    now = _now()
    ids = []

    # Embed up front (outside the write transaction) so one API call
    # covers the whole batch; the per-item loop then just does vec
    # lookups, which are fast.
    embeddings = None
    if embedder is not None:
        texts = [
            f"{(item.get('title') or '').strip()} "
            f"{(item.get('description') or '').strip()}".strip()
            for item in items
        ]
        embeddings = embedder.embed_batch(texts)
        if len(embeddings) != len(items):
            raise RuntimeError(
                f"embedder returned {len(embeddings)} vectors for "
                f"{len(items)} items — would silently mis-pair rows"
            )

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

        for idx, item in enumerate(items):
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

            # Dedup path: if this observation is a near-duplicate of an
            # existing pending (either from a prior batch or earlier in
            # this loop), attach the new source to that pending and
            # skip the INSERT.
            if embeddings is not None:
                emb = embeddings[idx]
                match = find_most_similar_pending(
                    conn, emb, threshold=PENDING_MERGE_THRESHOLD,
                )
                if match is not None:
                    existing_id, cos = match
                    log.info(
                        "pending dedup: new observation cos=%.3f ≥ %.2f — "
                        "linking post=%s/comment=%s to existing pending %d",
                        cos, PENDING_MERGE_THRESHOLD, post_id, comment_id,
                        existing_id,
                    )
                    # Attribution: the new post/comment becomes an extra
                    # source of the existing pending.
                    cursor = conn.execute(
                        "INSERT OR IGNORE INTO pending_painpoint_sources "
                        "(pending_painpoint_id, post_id, comment_id) "
                        "VALUES (?, ?, ?)",
                        (existing_id, post_id, comment_id),
                    )
                    # If the target pending has already been promoted,
                    # bump the linked painpoint's signal_count so the new
                    # observation contributes to ranking. Skip if the
                    # source already existed (INSERT OR IGNORE: rowcount
                    # == 0) to keep the op idempotent.
                    if cursor.rowcount:
                        promoted = conn.execute(
                            "SELECT painpoint_id FROM painpoint_sources "
                            "WHERE pending_painpoint_id = ?",
                            (existing_id,),
                        ).fetchone()
                        if promoted is not None:
                            conn.execute(
                                "UPDATE painpoints SET "
                                "signal_count = signal_count + 1, "
                                "last_updated = ? WHERE id = ?",
                                (now, promoted["painpoint_id"]),
                            )
                    ids.append(existing_id)
                    continue

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
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            ids.append(new_id)

            # Store the embedding so both (a) later items in this batch
            # can dedup against it, and (b) the promoter can reuse it
            # instead of re-embedding.
            if embeddings is not None:
                store_pending_painpoint_embedding(conn, new_id, embeddings[idx])

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

    # A pending that's been deduped against at save time carries extras
    # in pending_painpoint_sources. signal_count must initialise to the
    # full observation count (primary + extras), not hardcoded to 1.
    source_count = conn.execute(
        "SELECT COUNT(*) FROM pending_painpoint_all_sources "
        "WHERE pending_painpoint_id = ?",
        (pending_id,),
    ).fetchone()[0] or 1

    now = _now()
    conn.execute(
        "INSERT INTO painpoints "
        "(title, description, severity, signal_count, category_id, first_seen, last_updated) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (pending["title"], pending["description"] or "", pending["severity"],
         source_count, category_id, now, now),
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
    signal_count by the pending's total source count (primary + extras).

    Pending-stage dedup can attach multiple observations to one pending
    via `pending_painpoint_sources`, so a single pending may represent
    N real observations. signal_count must reflect observations, not
    pending rows, otherwise deduped topics underreport their signal.

    NOTE: does NOT call update_category_embedding. Linking adds a new
    PENDING source to an existing painpoint — the painpoint's category
    membership and its own embedding don't change, so the category
    centroid (mean of member painpoint embeddings) is unchanged.
    """
    source_count = conn.execute(
        "SELECT COUNT(*) FROM pending_painpoint_all_sources "
        "WHERE pending_painpoint_id = ?",
        (pending_id,),
    ).fetchone()[0] or 1   # at minimum the primary source
    try:
        conn.execute(
            "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) VALUES (?, ?)",
            (painpoint_id, pending_id),
        )
        conn.execute(
            "UPDATE painpoints SET signal_count = signal_count + ?, last_updated = ? "
            "WHERE id = ?",
            (source_count, _now(), painpoint_id),
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
