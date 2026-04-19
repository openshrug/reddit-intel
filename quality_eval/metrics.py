"""Per-snapshot programmatic metrics + cross-snapshot deltas.

Pure aggregation -- no LLM calls. Every metric is computed by the
existing read-only helpers in ``db/queries.py`` plus the gap-filling
helpers in ``inspect_db.py``. The output JSON is the input to a human
sanity-check and to the evaluator agent (it pairs with ``dump.md`` to
ground qualitative judgements).
"""

import json

import db
from db.queries import get_stats

from . import inspect_db


def compute_metrics(snapshot_path, *, previous_path=None):
    """Compute the metrics dict for ``snapshot_path``.

    If ``previous_path`` is given, also fills the ``delta_vs_previous``
    block (counts of new pendings/painpoints, cross-snapshot links).
    """
    with inspect_db.open_snapshot(snapshot_path):
        totals = _totals()
        pending_dedup = _pending_dedup_metrics()
        painpoint_merge = _painpoint_merge_metrics()
        categorization = _categorization_metrics()
        per_subreddit = _per_subreddit_metrics()
        category_events = _category_events_metrics()

    delta = None
    if previous_path is not None:
        diff = inspect_db.cross_snapshot_diff(previous_path, snapshot_path)
        delta = {
            "new_pendings": len(diff["new_pending_ids"]),
            "new_painpoints": len(diff["new_painpoint_ids"]),
            "linked_to_existing_painpoints": len(
                diff["pendings_linked_to_existing_painpoints"]
            ),
            "max_prev_pending_id": diff["max_prev_pending_id"],
            "max_prev_painpoint_id": diff["max_prev_painpoint_id"],
        }

    return {
        "totals": totals,
        "pending_dedup": pending_dedup,
        "painpoint_merge": painpoint_merge,
        "categorization": categorization,
        "per_subreddit": per_subreddit,
        "category_events": category_events,
        "delta_vs_previous": delta,
    }


def write_metrics(snapshot_path, out_path, *, previous_path=None):
    """Compute metrics and write them to ``out_path`` as pretty JSON."""
    payload = compute_metrics(snapshot_path, previous_path=previous_path)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


# ---------------------------------------------------------------------------
# Section helpers (each runs inside an already-opened snapshot context)
# ---------------------------------------------------------------------------

def _totals():
    """Global counters. Reuses ``db.queries.get_stats`` and adds a few
    that ``get_stats`` doesn't surface (root vs leaf categories,
    multi-source pending count, etc.)."""
    stats = get_stats()
    conn = db.get_db()
    try:
        roots = conn.execute(
            "SELECT COUNT(*) AS n FROM categories WHERE parent_id IS NULL"
        ).fetchone()["n"]
        leaves = conn.execute(
            "SELECT COUNT(*) AS n FROM categories c WHERE NOT EXISTS ("
            "SELECT 1 FROM categories c2 WHERE c2.parent_id = c.id)"
        ).fetchone()["n"]
        seed = conn.execute(
            "SELECT COUNT(*) AS n FROM categories WHERE is_seed = 1"
        ).fetchone()["n"]
        runtime = conn.execute(
            "SELECT COUNT(*) AS n FROM categories WHERE is_seed = 0"
        ).fetchone()["n"]
    finally:
        conn.close()

    return {
        "posts": stats["posts"],
        "comments": stats["comments"],
        "pending_painpoints": stats["pending_painpoints"],
        "painpoints": stats["painpoints"],
        "categories_total": stats["categories"],
        "categories_root": roots,
        "categories_leaf": leaves,
        "categories_seed": seed,
        "categories_runtime": runtime,
        "subreddits_with_posts": stats["subreddits"],
        "unmerged_pending": stats["unmerged_pending"],
    }


def _pending_dedup_metrics():
    """How aggressively did the PENDING_MERGE_THRESHOLD path collapse
    observations onto existing pendings?

    ``rate`` = extra_observations / total_observations. 0.0 means every
    LLM-emitted observation became its own pending row; higher means
    the dedup path collapsed near-duplicates.
    """
    conn = db.get_db()
    try:
        total_pending = conn.execute(
            "SELECT COUNT(*) AS n FROM pending_painpoints"
        ).fetchone()["n"]
        groups = conn.execute(
            "SELECT COUNT(*) AS n FROM ("
            " SELECT pending_painpoint_id FROM pending_painpoint_sources"
            " GROUP BY pending_painpoint_id"
            ")"
        ).fetchone()["n"]
        extra_observations = conn.execute(
            "SELECT COUNT(*) AS n FROM pending_painpoint_sources"
        ).fetchone()["n"]
        max_extras_row = conn.execute(
            "SELECT pending_painpoint_id, COUNT(*) AS n "
            "FROM pending_painpoint_sources "
            "GROUP BY pending_painpoint_id ORDER BY n DESC LIMIT 1"
        ).fetchone()
        max_extras = max_extras_row["n"] if max_extras_row else 0
    finally:
        conn.close()

    total_observations = total_pending + extra_observations
    rate = (extra_observations / total_observations) if total_observations else 0.0
    return {
        "groups_with_extras": groups,
        "extra_observations": extra_observations,
        "total_observations": total_observations,
        "max_extras_in_one_group": max_extras,
        "rate": round(rate, 4),
    }


def _painpoint_merge_metrics():
    """How aggressively did the promoter (and post-sweep painpoint_merge
    step) collapse pendings onto the same painpoint?

    ``rate`` = (pendings_linked - distinct_painpoints) / pendings_linked.
    0.0 means every linked pending got its own painpoint; higher means
    pendings collapsed onto fewer painpoints.
    """
    conn = db.get_db()
    try:
        pendings_linked = conn.execute(
            "SELECT COUNT(*) AS n FROM painpoint_sources"
        ).fetchone()["n"]
        distinct_painpoints = conn.execute(
            "SELECT COUNT(DISTINCT painpoint_id) AS n FROM painpoint_sources"
        ).fetchone()["n"]
        multi = conn.execute(
            "SELECT COUNT(*) AS n FROM ("
            " SELECT painpoint_id FROM painpoint_sources"
            " GROUP BY painpoint_id HAVING COUNT(*) > 1"
            ")"
        ).fetchone()["n"]
        max_signal_row = conn.execute(
            "SELECT MAX(signal_count) AS m FROM painpoints"
        ).fetchone()
        max_signal = max_signal_row["m"] or 0
    finally:
        conn.close()

    rate = (
        (pendings_linked - distinct_painpoints) / pendings_linked
        if pendings_linked else 0.0
    )
    return {
        "pendings_linked": pendings_linked,
        "distinct_painpoints": distinct_painpoints,
        "painpoints_with_multi_sources": multi,
        "max_signal_count": max_signal,
        "rate": round(rate, 4),
    }


def _categorization_metrics():
    """What share of painpoints landed in Uncategorized vs a real
    category? Snapshots 1-3 reflect promote-time placement only;
    snapshot 4 reflects post-sweep placement."""
    conn = db.get_db()
    try:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM painpoints"
        ).fetchone()["n"]
        uncat_row = conn.execute(
            "SELECT COUNT(*) AS n FROM painpoints p "
            "JOIN categories c ON c.id = p.category_id "
            "WHERE c.name = 'Uncategorized'"
        ).fetchone()
        uncategorized = uncat_row["n"]
        no_category = conn.execute(
            "SELECT COUNT(*) AS n FROM painpoints WHERE category_id IS NULL"
        ).fetchone()["n"]
        runtime = conn.execute(
            "SELECT COUNT(*) AS n FROM painpoints p "
            "JOIN categories c ON c.id = p.category_id "
            "WHERE c.is_seed = 0 AND c.name != 'Uncategorized'"
        ).fetchone()["n"]
        seed = conn.execute(
            "SELECT COUNT(*) AS n FROM painpoints p "
            "JOIN categories c ON c.id = p.category_id "
            "WHERE c.is_seed = 1 AND c.name != 'Uncategorized'"
        ).fetchone()["n"]
    finally:
        conn.close()

    pct_uncat = (uncategorized / total) if total else 0.0
    return {
        "painpoints_total": total,
        "in_uncategorized": uncategorized,
        "in_seed_category": seed,
        "in_runtime_category": runtime,
        "no_category_id": no_category,
        "pct_uncategorized": round(pct_uncat, 4),
    }


def _per_subreddit_metrics():
    """For every subreddit with posts, how many posts/comments/pendings
    /painpoints are attributable to it?"""
    out = {}
    for sr in inspect_db.list_distinct_subreddits():
        conn = db.get_db()
        try:
            posts = conn.execute(
                "SELECT COUNT(*) AS n FROM posts WHERE subreddit = ?", (sr,),
            ).fetchone()["n"]
            comments = conn.execute(
                "SELECT COUNT(*) AS n FROM comments cm "
                "JOIN posts p ON p.id = cm.post_id "
                "WHERE p.subreddit = ?", (sr,),
            ).fetchone()["n"]
            pendings = conn.execute(
                "SELECT COUNT(DISTINCT pp.id) AS n FROM pending_painpoints pp "
                "JOIN posts p ON p.id = pp.post_id "
                "WHERE p.subreddit = ?", (sr,),
            ).fetchone()["n"]
            painpoints = conn.execute(
                "SELECT COUNT(DISTINCT ps.painpoint_id) AS n "
                "FROM painpoint_sources ps "
                "JOIN pending_painpoints pp ON pp.id = ps.pending_painpoint_id "
                "JOIN posts p ON p.id = pp.post_id "
                "WHERE p.subreddit = ?", (sr,),
            ).fetchone()["n"]
        finally:
            conn.close()

        out[sr] = {
            "posts": posts,
            "comments": comments,
            "pendings": pendings,
            "painpoints_with_evidence": painpoints,
        }
    return out


def _category_events_metrics():
    """Counts per (event_type, accepted) -- empty until the post-sweep
    snapshot."""
    conn = db.get_db()
    try:
        rows = conn.execute(
            "SELECT event_type, accepted, COUNT(*) AS n "
            "FROM category_events GROUP BY event_type, accepted"
        ).fetchall()
    finally:
        conn.close()

    out = {}
    for r in rows:
        bucket = out.setdefault(r["event_type"], {"proposed": 0, "accepted": 0})
        bucket["proposed"] += r["n"]
        if r["accepted"]:
            bucket["accepted"] += r["n"]
    return out
