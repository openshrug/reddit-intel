"""Category worker event types, proposers, and acceptance tests
(§5 of docs/PAINPOINT_INGEST_PLAN.md).

The worker runs four passes per sweep, each calling one of the
`propose_*_events` generators below. Each yielded event is then run
through `apply_with_test`, which evaluates the event's domain-specific
test FIRST and only enters a savepoint to apply the mutation if the test
passed (per §5.4 — savepoints protect against mid-apply failures, NOT
against test rejection, because every test is pre-mutation).
"""

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, List, Optional

from . import _now
from .category_clustering import (
    cluster_painpoints,
    category_member_titles,
    inter_category_similarity,
)
from .llm_naming import LLMNamer
from .relevance import (
    MIN_RELEVANCE_TO_PROMOTE,
    cache_painpoint_relevance,
    get_or_compute_painpoint_relevance,
)

# Tunables — see §10 of the plan.
MIN_SUB_CLUSTER_SIZE = 5
SPLIT_RECHECK_DELTA = 10
MERGE_CATEGORY_THRESHOLD = 0.50
MIN_CATEGORY_RELEVANCE = 1.0
UNCATEGORIZED_NAME = "Uncategorized"

# Sweep clustering threshold is *lower* than the promoter's SIM_THRESHOLD
# (0.55) on purpose. Layer B at promote time wants HIGH precision (don't
# link painpoints that aren't clearly the same). Sweep clustering at sweep
# time wants HIGH recall (find anything that might be related so we can
# group them under a category). With the same threshold, anything that
# would cluster in Uncategorized would already have linked at Layer B, so
# the sweep would never find a clusterable group. Different operational
# requirements, different thresholds.
SWEEP_CLUSTER_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


@dataclass
class CategoryEvent:
    event_type: str        # add_category_new | add_category_split | delete_category | merge_categories
    payload: dict          # event-specific data
    target_category: Optional[int] = None
    triggering_pp: Optional[int] = None
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0


# ---------------------------------------------------------------------------
# Step 1 — process Uncategorized
# ---------------------------------------------------------------------------


def _uncategorized_id(conn):
    row = conn.execute(
        "SELECT id FROM categories WHERE name = ?", (UNCATEGORIZED_NAME,)
    ).fetchone()
    return row["id"] if row else None


def propose_uncategorized_events(conn):
    """Step 1: cluster the Uncategorized bucket via §3.2 text MinHash, yield
    one `add_category_new` event per cluster of size ≥ MIN_SUB_CLUSTER_SIZE.

    Singletons stay in Uncategorized waiting for siblings to arrive.
    """
    uncat_id = _uncategorized_id(conn)
    if uncat_id is None:
        return
    members = [
        dict(r)
        for r in conn.execute(
            "SELECT id, title, description FROM painpoints WHERE category_id = ?",
            (uncat_id,),
        ).fetchall()
    ]
    if not members:
        return

    clusters = cluster_painpoints(members, threshold=SWEEP_CLUSTER_THRESHOLD)
    for cluster in clusters:
        if len(cluster) < MIN_SUB_CLUSTER_SIZE:
            continue
        yield CategoryEvent(
            event_type="add_category_new",
            payload={
                "painpoint_ids": [p["id"] for p in cluster],
                "sample_titles": [p["title"] for p in cluster],
                "sample_descriptions": [p["description"] or "" for p in cluster],
            },
            target_category=uncat_id,
            metric_name="cluster_size",
            metric_value=float(len(cluster)),
            threshold=float(MIN_SUB_CLUSTER_SIZE),
        )


# ---------------------------------------------------------------------------
# Step 2 — split crowded categories
# ---------------------------------------------------------------------------


def propose_split_events(conn):
    """Step 2: for each non-Uncategorized category whose member count has
    grown by ≥ SPLIT_RECHECK_DELTA since the last check, propose
    `add_category_split` if intra-bucket clustering finds ≥2 sub-clusters
    of size ≥ MIN_SUB_CLUSTER_SIZE.
    """
    uncat_id = _uncategorized_id(conn)
    rows = conn.execute(
        "SELECT id, name, parent_id, painpoint_count_at_last_check FROM categories "
        "WHERE name != ?",
        (UNCATEGORIZED_NAME,),
    ).fetchall()

    for cat in rows:
        cat_id = cat["id"]
        current_count = conn.execute(
            "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat_id,)
        ).fetchone()[0]
        last_count = cat["painpoint_count_at_last_check"] or 0

        if current_count - last_count < SPLIT_RECHECK_DELTA:
            continue

        members = category_member_titles(conn, cat_id)
        clusters = cluster_painpoints(members, threshold=SWEEP_CLUSTER_THRESHOLD)
        valid = [c for c in clusters if len(c) >= MIN_SUB_CLUSTER_SIZE]

        # Always update the trigger snapshot — even if the split fails, we
        # don't want to re-cluster on every sweep until the bucket grows
        # by another SPLIT_RECHECK_DELTA.
        conn.execute(
            "UPDATE categories SET painpoint_count_at_last_check = ?, "
            "last_split_check_at = ? WHERE id = ?",
            (current_count, _now(), cat_id),
        )

        if len(valid) < 2:
            # Test would reject; skip the event entirely (audit trail still
            # gets the rejected entry via the worker if you want it — see
            # the trigger-discipline test in §9 for why we don't propose
            # rejected events for stable buckets).
            continue

        yield CategoryEvent(
            event_type="add_category_split",
            payload={
                "parent_category_id": cat["parent_id"],
                "source_category_id": cat_id,
                "source_category_name": cat["name"],
                "clusters": [[p["id"] for p in c] for c in valid],
                "cluster_titles": [[p["title"] for p in c] for c in valid],
            },
            target_category=cat_id,
            metric_name="sub_cluster_count",
            metric_value=float(len(valid)),
            threshold=2.0,
        )


# ---------------------------------------------------------------------------
# Step 3 — delete dead categories
# ---------------------------------------------------------------------------


def category_relevance_mass(conn, category_id):
    """Sum of cached relevance over members of a category. Members with NULL
    cached relevance get computed on the fly."""
    rows = conn.execute(
        "SELECT id, relevance FROM painpoints WHERE category_id = ?", (category_id,)
    ).fetchall()
    total = 0.0
    for r in rows:
        rel = r["relevance"]
        if rel is None:
            rel = cache_painpoint_relevance(r["id"], conn=conn)
        total += rel or 0.0
    return total


def propose_delete_events(conn):
    """Step 3: for each non-Uncategorized category whose total relevance
    mass < MIN_CATEGORY_RELEVANCE, propose `delete_category`. The
    acceptance test (no member with relevance > MIN_RELEVANCE_TO_PROMOTE)
    runs in apply_with_test.

    A category is only considered "dead" if it has at least one member
    today. An empty (unborn) category — e.g., a freshly seeded one that
    no painpoint has landed in yet — is not dead, it's unused. Different
    concept; we leave those alone.
    """
    rows = conn.execute(
        "SELECT id, name, parent_id FROM categories WHERE name != ?",
        (UNCATEGORIZED_NAME,),
    ).fetchall()
    for cat in rows:
        member_count = conn.execute(
            "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat["id"],)
        ).fetchone()[0]
        if member_count == 0:
            continue   # unborn, not dead

        mass = category_relevance_mass(conn, cat["id"])
        if mass >= MIN_CATEGORY_RELEVANCE:
            continue
        yield CategoryEvent(
            event_type="delete_category",
            payload={
                "category_id": cat["id"],
                "category_name": cat["name"],
                "parent_id": cat["parent_id"],
            },
            target_category=cat["id"],
            metric_name="category_mass",
            metric_value=mass,
            threshold=MIN_CATEGORY_RELEVANCE,
        )


# ---------------------------------------------------------------------------
# Step 4 — merge duplicate sibling categories
# ---------------------------------------------------------------------------


def propose_merge_events(conn):
    """Step 4: for each pair of sibling categories under the same parent,
    compute inter-category text similarity (max pairwise member-title
    MinHash). If above MERGE_CATEGORY_THRESHOLD, propose merge_categories.
    """
    parents = conn.execute(
        "SELECT DISTINCT parent_id FROM categories WHERE parent_id IS NOT NULL"
    ).fetchall()
    for p in parents:
        siblings = conn.execute(
            "SELECT id, name FROM categories WHERE parent_id = ? AND name != ? "
            "ORDER BY id",
            (p["parent_id"], UNCATEGORIZED_NAME),
        ).fetchall()
        ids = [s["id"] for s in siblings]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                sim = inter_category_similarity(conn, a, b)
                if sim < MERGE_CATEGORY_THRESHOLD:
                    continue
                yield CategoryEvent(
                    event_type="merge_categories",
                    payload={
                        "survivor_id": a,    # arbitrary tie-break: lower id wins
                        "loser_id": b,
                    },
                    target_category=a,
                    metric_name="merge_text_sim",
                    metric_value=sim,
                    threshold=MERGE_CATEGORY_THRESHOLD,
                )


# ---------------------------------------------------------------------------
# Acceptance tests (pre-mutation, see §5.2)
# ---------------------------------------------------------------------------


def _test_add_category_new(conn, event):
    """Cluster size ≥ MIN_SUB_CLUSTER_SIZE — already enforced at proposal
    time, so always passes here. The collision check happens *during*
    apply, after the LLM has named it."""
    return True, "cluster_size >= threshold"


def _test_add_category_split(conn, event):
    """≥2 sub-clusters of size ≥ MIN_SUB_CLUSTER_SIZE — already enforced
    at proposal time."""
    return True, "valid sub-clusters >= 2"


def _test_delete_category(conn, event):
    """Safety check: no member painpoint has relevance > MIN_RELEVANCE_TO_PROMOTE."""
    cat_id = event.payload["category_id"]
    members = conn.execute(
        "SELECT id FROM painpoints WHERE category_id = ?", (cat_id,)
    ).fetchall()
    for r in members:
        rel = get_or_compute_painpoint_relevance(r["id"], conn=conn)
        if rel is not None and rel > MIN_RELEVANCE_TO_PROMOTE:
            return False, f"member painpoint {r['id']} has live relevance {rel:.3f}"
    return True, "no live members"


def _test_merge_categories(conn, event):
    """Already gated by MERGE_CATEGORY_THRESHOLD at proposal time."""
    return True, "similarity > threshold"


_TESTS = {
    "add_category_new": _test_add_category_new,
    "add_category_split": _test_add_category_split,
    "delete_category": _test_delete_category,
    "merge_categories": _test_merge_categories,
}


def run_acceptance_test(conn, event):
    fn = _TESTS.get(event.event_type)
    if fn is None:
        return False, "unknown event type"
    return fn(conn, event)


# ---------------------------------------------------------------------------
# Apply (under savepoint)
# ---------------------------------------------------------------------------


def _apply_add_category_new(conn, event, namer):
    name_resp = namer.name_new_category(
        event.payload["sample_titles"], event.payload["sample_descriptions"]
    )
    name = name_resp.get("name", "").strip()
    description = name_resp.get("description", "").strip()
    if not name:
        raise RuntimeError("LLM returned empty category name")

    # Name collision check — promote merge if exact-name match exists.
    existing = conn.execute(
        "SELECT id FROM categories WHERE name = ?", (name,)
    ).fetchone()
    if existing is not None:
        target_id = existing["id"]
    else:
        conn.execute(
            "INSERT INTO categories (name, parent_id, description, created_at) "
            "VALUES (?, NULL, ?, ?)",
            (name, description, _now()),
        )
        target_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    for pp_id in event.payload["painpoint_ids"]:
        conn.execute(
            "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE id = ?",
            (target_id, _now(), pp_id),
        )

    return target_id


def _apply_add_category_split(conn, event, namer):
    parent_id = event.payload["parent_category_id"]
    source_id = event.payload["source_category_id"]
    parent_name = event.payload["source_category_name"]
    cluster_titles = event.payload["cluster_titles"]
    clusters = event.payload["clusters"]

    sub_resp = namer.name_split_subcategories(parent_name, cluster_titles)
    if len(sub_resp) < len(clusters):
        raise RuntimeError(
            f"LLM returned {len(sub_resp)} sub-cats for {len(clusters)} clusters"
        )

    new_cat_ids = []
    for cluster, sub in zip(clusters, sub_resp):
        name = (sub.get("name") or "").strip()
        description = (sub.get("description") or "").strip()
        if not name:
            raise RuntimeError("LLM returned empty sub-category name")
        # Resolve / insert
        existing = conn.execute(
            "SELECT id FROM categories WHERE name = ?", (name,)
        ).fetchone()
        if existing is not None:
            sub_id = existing["id"]
        else:
            conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES (?, ?, ?, ?)",
                (name, parent_id, description, _now()),
            )
            sub_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        new_cat_ids.append(sub_id)

        for pp_id in cluster:
            conn.execute(
                "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE id = ?",
                (sub_id, _now(), pp_id),
            )

    # Retire the source category if it's now empty.
    remaining = conn.execute(
        "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (source_id,)
    ).fetchone()[0]
    if remaining == 0 and parent_id is not None:
        conn.execute("DELETE FROM categories WHERE id = ?", (source_id,))

    return new_cat_ids


def _apply_delete_category(conn, event, namer):
    cat_id = event.payload["category_id"]
    parent_id = event.payload["parent_id"]

    # Relink any surviving members to the parent (or Uncategorized if no parent).
    fallback_id = parent_id
    if fallback_id is None:
        fallback_id = _uncategorized_id(conn)

    conn.execute(
        "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE category_id = ?",
        (fallback_id, _now(), cat_id),
    )
    conn.execute("DELETE FROM categories WHERE id = ?", (cat_id,))
    return cat_id


def _apply_merge_categories(conn, event, namer):
    survivor_id = event.payload["survivor_id"]
    loser_id = event.payload["loser_id"]
    conn.execute(
        "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE category_id = ?",
        (survivor_id, _now(), loser_id),
    )
    conn.execute("DELETE FROM categories WHERE id = ?", (loser_id,))
    return survivor_id


_APPLIERS = {
    "add_category_new": _apply_add_category_new,
    "add_category_split": _apply_add_category_split,
    "delete_category": _apply_delete_category,
    "merge_categories": _apply_merge_categories,
}


def apply_event(conn, event, namer):
    fn = _APPLIERS.get(event.event_type)
    if fn is None:
        raise ValueError(f"unknown event type {event.event_type}")
    return fn(conn, event, namer)


# ---------------------------------------------------------------------------
# log + apply runner
# ---------------------------------------------------------------------------


def log_event(conn, event, accepted, reason=""):
    conn.execute(
        "INSERT INTO category_events "
        "(event_type, proposed_at, triggering_pp, target_category, payload_json, "
        " metric_name, metric_value, threshold, accepted, reason) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            event.event_type,
            _now(),
            event.triggering_pp,
            event.target_category,
            json.dumps(event.payload, default=str),
            event.metric_name,
            event.metric_value,
            event.threshold,
            1 if accepted else 0,
            reason,
        ),
    )


def apply_with_test(conn, event, namer):
    """Test first, apply only on accept; savepoint protects against
    mid-apply failures, not against test rejection (per §5.4)."""
    ok, reason = run_acceptance_test(conn, event)
    if not ok:
        log_event(conn, event, accepted=False, reason=reason)
        return False

    conn.execute("SAVEPOINT cat_event")
    try:
        apply_event(conn, event, namer)
    except Exception as e:
        conn.execute("ROLLBACK TO SAVEPOINT cat_event")
        log_event(conn, event, accepted=False, reason=f"apply error: {e}")
        return False

    conn.execute("RELEASE SAVEPOINT cat_event")
    log_event(conn, event, accepted=True, reason="applied")
    return True
