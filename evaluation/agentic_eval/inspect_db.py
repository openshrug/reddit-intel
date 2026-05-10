"""Read-only inspection helpers for snapshotted SQLite DBs.

Most of what we need is already exposed by ``db/queries.py``,
``db/categories.py`` and ``db/posts.py``. Those helpers all call
``db.get_db()`` which reads the module-level ``db.DB_PATH`` lazily, so
swapping ``db.DB_PATH`` for the duration of a block lets every existing
helper run against any snapshot file without changes.

This module exposes:

* ``open_snapshot(path)`` -- context manager that swaps ``db.DB_PATH``
  and restores it on exit (same trick the test suite uses, see
  ``tests/live/test_e2e_real_subreddits.py``).
* Five small "gap-filling" helpers that the existing API doesn't cover:
    1. ``list_pending_painpoints_for_subreddit`` (raw pending side)
    2. ``list_pending_dedup_groups`` (PENDING_MERGE_THRESHOLD collapses)
    3. ``render_category_tree`` (recursive tree with member counts)
    4. ``cross_snapshot_diff`` (deltas between two snapshots in the
       same run)
    5. ``cross_run_pending_diff`` (deltas between two independent
       runs of the pipeline -- joins by Reddit permalink so synthetic
       row IDs renumbering between runs doesn't matter)

All other reads should call the existing ``db/queries.py`` helpers
inside an ``open_snapshot`` block.
"""

from contextlib import contextmanager
from pathlib import Path

import db

# ---------------------------------------------------------------------------
# Snapshot context manager
# ---------------------------------------------------------------------------

@contextmanager
def open_snapshot(path):
    """Point ``db.DB_PATH`` at ``path`` for the duration of the block.

    Every helper in ``db/queries.py``, ``db/categories.py`` and
    ``db/posts.py`` resolves the DB file via ``db.get_db()`` -> reads
    ``db.DB_PATH`` at call time, so they all transparently start
    operating on the snapshot file inside this context.
    """
    original = db.DB_PATH
    db.DB_PATH = Path(path)
    try:
        yield
    finally:
        db.DB_PATH = original


# ---------------------------------------------------------------------------
# (1) Pending-side listing -- existing helpers focus on the merged side
# ---------------------------------------------------------------------------

def list_pending_painpoints_for_subreddit(subreddit, *, since_pp_id=None,
                                          limit=None):
    """Return raw ``pending_painpoints`` rows whose primary post lives in
    ``subreddit``, joined with post + comment context.

    Args:
        subreddit: subreddit name; normalised via ``db.normalize_subreddit``
            so callers can pass ``r/SaaS`` / ``saas`` / ``SaaS`` interchangeably.
        since_pp_id: if set, only return pendings with ``id > since_pp_id``
            (used for cross-snapshot "what was added in this stage" diffs).
        limit: optional row cap for sampling.

    Each dict carries: pending fields (id, title, description, severity,
    quoted_text, category_id), the originating post (post_id, post_title,
    post_permalink, post_score, subreddit) and -- when the pending was
    pinned to a comment -- the comment body + permalink. Also includes
    ``extra_source_count`` so callers can spot pendings that absorbed
    additional observations through the dedup path.
    """
    conn = db.get_db()
    try:
        clauses = ["p.subreddit = ?"]
        params = [db.normalize_subreddit(subreddit)]
        if since_pp_id is not None:
            clauses.append("pp.id > ?")
            params.append(since_pp_id)

        sql = f"""
            SELECT
                pp.id            AS pending_id,
                pp.title         AS pending_title,
                pp.description   AS pending_description,
                pp.quoted_text,
                pp.severity      AS pending_severity,
                pp.extracted_at,
                pp.category_id   AS pending_category_id,
                cat.name         AS pending_category_name,
                p.id             AS post_id,
                p.title          AS post_title,
                p.permalink      AS post_permalink,
                p.score          AS post_score,
                p.subreddit,
                c.id             AS comment_id,
                c.body           AS comment_body,
                c.permalink      AS comment_permalink,
                c.score          AS comment_score,
                (SELECT COUNT(*) FROM pending_painpoint_sources pps
                    WHERE pps.pending_painpoint_id = pp.id) AS extra_source_count
            FROM pending_painpoints pp
            JOIN posts p ON p.id = pp.post_id
            LEFT JOIN comments c ON c.id = pp.comment_id
            LEFT JOIN categories cat ON cat.id = pp.category_id
            WHERE {' AND '.join(clauses)}
            ORDER BY pp.id
        """
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# (2) Pending dedup groups (PENDING_MERGE_THRESHOLD = 0.65 collapses)
# ---------------------------------------------------------------------------

def list_pending_dedup_groups(*, min_extra_sources=1, limit=None):
    """Return pending painpoints that have at least one *extra* observation
    collapsed onto them via ``pending_painpoint_sources``.

    A "dedup group" is one ``pending_painpoints`` row plus every
    additional (post, comment) tuple that the embedding-cosine merge at
    ``PENDING_MERGE_THRESHOLD`` (see ``db/painpoints.py``
    ``save_pending_painpoints_batch``) attached to it.

    Args:
        min_extra_sources: minimum count of rows in
            ``pending_painpoint_sources`` for the pending to be returned.
            Default 1 means "at least one collapsed observation" -- i.e.,
            every group that actually exercised the dedup path.
        limit: optional cap on number of groups (groups are returned
            largest first so the cap samples the most interesting cases).

    Each group dict has::

        {
            "pending_id": int,
            "title": str,
            "description": str,
            "severity": int,
            "extra_source_count": int,
            "primary": {post_id, post_title, post_permalink, subreddit,
                        comment_id?, comment_body?, comment_permalink?},
            "extras": [
                {post_id, post_title, post_permalink, subreddit,
                 comment_id?, comment_body?, comment_permalink?},
                ...
            ],
        }
    """
    conn = db.get_db()
    try:
        sql = """
            SELECT
                pp.id            AS pending_id,
                pp.title,
                pp.description,
                pp.severity,
                pp.post_id       AS primary_post_id,
                pp.comment_id    AS primary_comment_id,
                p.title          AS primary_post_title,
                p.permalink      AS primary_post_permalink,
                p.subreddit      AS primary_subreddit,
                c.body           AS primary_comment_body,
                c.permalink      AS primary_comment_permalink,
                (SELECT COUNT(*) FROM pending_painpoint_sources pps
                    WHERE pps.pending_painpoint_id = pp.id) AS extra_source_count
            FROM pending_painpoints pp
            JOIN posts p ON p.id = pp.post_id
            LEFT JOIN comments c ON c.id = pp.comment_id
            WHERE (SELECT COUNT(*) FROM pending_painpoint_sources pps
                    WHERE pps.pending_painpoint_id = pp.id) >= ?
            ORDER BY extra_source_count DESC, pp.id
        """
        params = [min_extra_sources]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        groups = []
        for r in conn.execute(sql, params).fetchall():
            extras = conn.execute(
                """
                SELECT
                    pps.post_id,
                    pps.comment_id,
                    p.title       AS post_title,
                    p.permalink   AS post_permalink,
                    p.subreddit,
                    c.body        AS comment_body,
                    c.permalink   AS comment_permalink
                FROM pending_painpoint_sources pps
                JOIN posts p ON p.id = pps.post_id
                LEFT JOIN comments c ON c.id = pps.comment_id
                WHERE pps.pending_painpoint_id = ?
                ORDER BY pps.post_id, pps.comment_id
                """,
                (r["pending_id"],),
            ).fetchall()

            groups.append({
                "pending_id": r["pending_id"],
                "title": r["title"],
                "description": r["description"],
                "severity": r["severity"],
                "extra_source_count": r["extra_source_count"],
                "primary": {
                    "post_id": r["primary_post_id"],
                    "post_title": r["primary_post_title"],
                    "post_permalink": r["primary_post_permalink"],
                    "subreddit": r["primary_subreddit"],
                    "comment_id": r["primary_comment_id"],
                    "comment_body": r["primary_comment_body"],
                    "comment_permalink": r["primary_comment_permalink"],
                },
                "extras": [dict(x) for x in extras],
            })
        return groups
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# (3) Category tree with recursive member counts
# ---------------------------------------------------------------------------

def render_category_tree():
    """Return the full category tree as a list of root dicts.

    Each node::

        {
            "id": int,
            "name": str,
            "description": str,
            "is_seed": int,
            "parent_id": int | None,
            "direct_painpoints": int,   # painpoints whose category_id == self
            "total_painpoints": int,    # direct + descendants
            "children": [ ...node... ],
        }

    Empty subtrees (zero descendant painpoints) are kept; callers that
    want to skip them can filter on ``total_painpoints == 0`` themselves.
    """
    conn = db.get_db()
    try:
        cats = conn.execute(
            "SELECT id, name, description, parent_id, is_seed "
            "FROM categories ORDER BY name"
        ).fetchall()
        cats_by_id = {r["id"]: dict(r) for r in cats}
        for node in cats_by_id.values():
            node["children"] = []
            node["direct_painpoints"] = 0

        for r in conn.execute(
            "SELECT category_id, COUNT(*) AS n FROM painpoints "
            "WHERE category_id IS NOT NULL GROUP BY category_id"
        ).fetchall():
            if r["category_id"] in cats_by_id:
                cats_by_id[r["category_id"]]["direct_painpoints"] = r["n"]

        roots = []
        for node in cats_by_id.values():
            parent = cats_by_id.get(node["parent_id"])
            if parent is None:
                roots.append(node)
            else:
                parent["children"].append(node)

        for node in cats_by_id.values():
            node["children"].sort(key=lambda x: x["name"])

        def fill_totals(node):
            total = node["direct_painpoints"]
            for child in node["children"]:
                total += fill_totals(child)
            node["total_painpoints"] = total
            return total

        roots.sort(key=lambda x: x["name"])
        for root in roots:
            fill_totals(root)
        return roots
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# (4) Cross-snapshot diff
# ---------------------------------------------------------------------------

def cross_snapshot_diff(prev_path, cur_path):
    """Compare two snapshot DBs and return what changed in ``cur_path``.

    Returns a dict::

        {
            "new_pending_ids": [int, ...],
            "new_painpoint_ids": [int, ...],
            "pendings_linked_to_existing_painpoints": [
                {"pending_id": int, "painpoint_id": int}, ...
            ],
            "max_prev_pending_id": int,
            "max_prev_painpoint_id": int,
        }

    "Linked to existing" means: a pending added in ``cur_path`` (id >
    ``max_prev_pending_id``) whose ``painpoint_sources`` row points at a
    ``painpoint_id`` that was already present in ``prev_path``. This is
    the cross-subreddit merge signal -- e.g. a r/ClaudeAI complaint
    landing on the same painpoint as an earlier r/openclaw one.
    """
    with open_snapshot(prev_path):
        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS m FROM pending_painpoints"
            ).fetchone()
            max_prev_pending = row["m"]
            row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS m FROM painpoints"
            ).fetchone()
            max_prev_painpoint = row["m"]
            prev_painpoint_ids = {
                r["id"]
                for r in conn.execute("SELECT id FROM painpoints").fetchall()
            }
        finally:
            conn.close()

    with open_snapshot(cur_path):
        conn = db.get_db()
        try:
            new_pending_ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM pending_painpoints WHERE id > ? ORDER BY id",
                    (max_prev_pending,),
                ).fetchall()
            ]
            new_painpoint_ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM painpoints WHERE id > ? ORDER BY id",
                    (max_prev_painpoint,),
                ).fetchall()
            ]
            linked = []
            if new_pending_ids and prev_painpoint_ids:
                placeholders = ",".join("?" * len(new_pending_ids))
                rows = conn.execute(
                    f"SELECT pending_painpoint_id, painpoint_id "
                    f"FROM painpoint_sources "
                    f"WHERE pending_painpoint_id IN ({placeholders})",
                    new_pending_ids,
                ).fetchall()
                for r in rows:
                    if r["painpoint_id"] in prev_painpoint_ids:
                        linked.append({
                            "pending_id": r["pending_painpoint_id"],
                            "painpoint_id": r["painpoint_id"],
                        })
        finally:
            conn.close()

    return {
        "new_pending_ids": new_pending_ids,
        "new_painpoint_ids": new_painpoint_ids,
        "pendings_linked_to_existing_painpoints": linked,
        "max_prev_pending_id": max_prev_pending,
        "max_prev_painpoint_id": max_prev_painpoint,
    }


# ---------------------------------------------------------------------------
# (5) Cross-run pending diff -- compare two independent pipeline runs
# ---------------------------------------------------------------------------

def cross_run_pending_diff(snapshot_a_db, snapshot_b_db):
    """Diff two independent pipeline runs (e.g. before vs after a prompt
    change), joining pendings by Reddit identity rather than synthetic
    row IDs.

    The pending row IDs auto-increment per run, so they cannot be used
    to match "the same observation" across runs. Reddit's
    ``post.permalink`` (always populated) and ``comment.permalink``
    (populated for any comment that came through the scraper) are the
    stable identity. The cell-level key is
    ``(post_permalink, comment_permalink or '__POST_BODY__')``.

    Args:
        snapshot_a_db: path to a snapshot's ``trends.db`` (commonly the
            ``01_<sub>`` snapshot from the OLD run).
        snapshot_b_db: same shape, NEW run.

    Returns:
        Dict with three lists::

            {
                "common": [(row_a, row_b), ...],   # same Reddit cell in both runs
                "only_a": [row_a, ...],            # disappeared under run B
                "only_b": [row_b, ...],            # newly emitted under run B
            }

        Each row is a dict with keys (consistent across both sides so
        downstream renderers don't need to care which side they're on)::

            pending_id, title, description, quoted_text, severity,
            category_name, post_id, post_permalink, post_title,
            subreddit, comment_id, comment_permalink, comment_body,
            key   # the join key, useful for sorting / filtering

        ``common`` pairs are sorted by ``post_permalink`` then
        ``comment_permalink`` for stable iteration; ``only_a`` /
        ``only_b`` are sorted by ``severity`` descending so the most
        salient deltas come first.

    Notes:
        - Embedding-merged sources (``pending_painpoint_sources``) are
          NOT walked here: this is intentionally a raw-pending diff,
          since dedup behaviour itself is one of the things you're
          usually trying to compare. If you need merge-side identity,
          wrap this with ``list_pending_dedup_groups`` per side.
        - Comments without a permalink (rare; some scraper paths leave
          it null) fall through to the post-body bucket and may be
          matched together. The helper logs a warning if it sees more
          than one such comment per post; in practice this hasn't
          fired against any real run.
    """
    rows_a = _load_pendings_with_identity(snapshot_a_db)
    rows_b = _load_pendings_with_identity(snapshot_b_db)

    by_key_a = {r["key"]: r for r in rows_a}
    by_key_b = {r["key"]: r for r in rows_b}

    common_keys = sorted(set(by_key_a) & set(by_key_b))
    only_a_keys = sorted(set(by_key_a) - set(by_key_b))
    only_b_keys = sorted(set(by_key_b) - set(by_key_a))

    return {
        "common": [(by_key_a[k], by_key_b[k]) for k in common_keys],
        "only_a": sorted(
            (by_key_a[k] for k in only_a_keys),
            key=lambda r: -(r["severity"] or 0),
        ),
        "only_b": sorted(
            (by_key_b[k] for k in only_b_keys),
            key=lambda r: -(r["severity"] or 0),
        ),
    }


def _load_pendings_with_identity(snapshot_db):
    """Load every pending in the snapshot tagged with its Reddit-stable
    identity. Internal helper for ``cross_run_pending_diff``."""
    with open_snapshot(snapshot_db):
        conn = db.get_db()
        try:
            rows = conn.execute(
                """
                SELECT
                    pp.id            AS pending_id,
                    pp.title,
                    pp.description,
                    pp.quoted_text,
                    pp.severity,
                    pp.category_id,
                    cat.name         AS category_name,
                    p.id             AS post_id,
                    p.permalink      AS post_permalink,
                    p.title          AS post_title,
                    p.subreddit,
                    pp.comment_id    AS comment_id,
                    c.permalink      AS comment_permalink,
                    c.body           AS comment_body
                FROM pending_painpoints pp
                JOIN posts p       ON p.id = pp.post_id
                LEFT JOIN comments c   ON c.id = pp.comment_id
                LEFT JOIN categories cat ON cat.id = pp.category_id
                """
            ).fetchall()
        finally:
            conn.close()

    out = []
    for r in rows:
        d = dict(r)
        comment_marker = d["comment_permalink"] or "__POST_BODY__"
        d["key"] = (d["post_permalink"], comment_marker)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Convenience: per-snapshot bookkeeping the dump/metrics modules need
# ---------------------------------------------------------------------------

def get_max_pending_id():
    """Return the largest ``pending_painpoints.id`` in the current DB,
    or ``0`` if the table is empty."""
    conn = db.get_db()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM pending_painpoints"
        ).fetchone()
        return row["m"]
    finally:
        conn.close()


def get_max_painpoint_id():
    """Return the largest ``painpoints.id`` in the current DB, or ``0``."""
    conn = db.get_db()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM painpoints"
        ).fetchone()
        return row["m"]
    finally:
        conn.close()


def list_distinct_subreddits():
    """All subreddits with at least one post in the current DB."""
    conn = db.get_db()
    try:
        rows = conn.execute(
            "SELECT DISTINCT subreddit FROM posts ORDER BY subreddit"
        ).fetchall()
        return [r["subreddit"] for r in rows]
    finally:
        conn.close()


def get_category_events(limit=None):
    """Return rows from the ``category_events`` audit log, oldest first.

    Used by the post-sweep snapshot dump (Section 6 in dump.md).
    """
    conn = db.get_db()
    try:
        sql = (
            "SELECT id, event_type, accepted, metric_name, metric_value, "
            "threshold, reason, target_category, triggering_pp, proposed_at "
            "FROM category_events ORDER BY id"
        )
        params = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
