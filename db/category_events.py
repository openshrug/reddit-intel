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
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

from datetime import datetime, timedelta, timezone

from . import _now
from .category_clustering import (
    cluster_painpoints,
    category_member_titles,
    inter_category_similarity,
)
from . import UNCATEGORIZED_NAME, uncategorized_id
from .embeddings import MERGE_COSINE_THRESHOLD, update_category_embedding
from .relevance import _parse_iso

MIN_SUB_CLUSTER_SIZE = 5
SPLIT_RECHECK_DELTA = 10
MERGE_CATEGORY_THRESHOLD = 0.80
# A category with no member added/updated in this many days is stale —
# propose delete. Replaces the old mass-based threshold (which was a size
# test dressed up as a quality test).
CATEGORY_STALE_DAYS = 30

# Step 5 (reroute). A painpoint is moved to a non-current category only
# when the alternative's cosine sim exceeds the current sim by at least
# REROUTE_MARGIN. Conservative so we don't thrash near-ties. Fixes
# residual singleton hijacking that the split step can't cluster.
REROUTE_MARGIN = 0.08


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
    # Pre-fetched LLM response, populated by prefetch_llm_for_event.
    # If present, the _apply_* function uses this instead of making a
    # fresh LLM call. Lets us parallelize LLM calls across events
    # without releasing the merge_lock between them.
    llm_result: Optional[dict] = None


# ---------------------------------------------------------------------------
# Step 1 — process Uncategorized
# ---------------------------------------------------------------------------


def _uncategorized_id(conn):
    """Local wrapper around db.uncategorized_id that returns None
    instead of raising if the sentinel is missing — keeps the
    proposers tolerant of an uninitialised DB (used in tests)."""
    try:
        return uncategorized_id(conn)
    except RuntimeError:
        return None


def propose_uncategorized_events(conn, embedder=None):
    """Step 1: cluster the Uncategorized bucket via embedding cosine sim,
    yield one `add_category_new` event per cluster of size >=
    MIN_SUB_CLUSTER_SIZE.

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

    clusters = cluster_painpoints(members, threshold=MERGE_COSINE_THRESHOLD, embedder=embedder, conn=conn)
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


def propose_split_events(conn, embedder=None, namer=None):
    """Step 2: for each non-Uncategorized category that has grown by ≥
    SPLIT_RECHECK_DELTA since the last check, ask the LLM whether to
    split it.

    Fixed-size-floor heuristics (MIN_SUB_CLUSTER_SIZE) kept producing
    either "one giant blob" or "all tiny fragments" depending on the
    clustering threshold — neither a reliable split signal. Instead we
    now cluster the members, send the LLM a summary (category name,
    description, total members, top cluster samples), and let the LLM
    decide whether splitting makes semantic sense.

    The LLM returns a SplitDecision with either "keep" or "split" plus
    proposed sub-categories grouped by cluster indices.
    """
    if namer is None:
        from .llm_naming import LLMNamer
        namer = LLMNamer()

    # Minimum category size to even consider splitting. Below this, the
    # category can't plausibly have distinct sub-topics.
    MIN_CATEGORY_SIZE_FOR_SPLIT = 10
    # Top N clusters (by size) to show the LLM. Keeps the prompt bounded.
    MAX_CLUSTERS_SHOWN = 10

    rows = conn.execute(
        "SELECT id, name, description, parent_id, painpoint_count_at_last_check "
        "FROM categories WHERE name != ?",
        (UNCATEGORIZED_NAME,),
    ).fetchall()

    # Phase 1 (sequential, under merge_lock): collect candidates, cluster
    # their members, snapshot the trigger counter. The LLM call stays for
    # phase 2 so we can fan it out.
    candidates = []  # list of dicts: cat, current_count, top_clusters, cluster_payload
    for cat in rows:
        cat_id = cat["id"]
        current_count = conn.execute(
            "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat_id,)
        ).fetchone()[0]
        last_count = cat["painpoint_count_at_last_check"] or 0

        if current_count < MIN_CATEGORY_SIZE_FOR_SPLIT:
            continue
        if current_count - last_count < SPLIT_RECHECK_DELTA:
            continue

        members = category_member_titles(conn, cat_id)
        clusters = cluster_painpoints(members, threshold=MERGE_COSINE_THRESHOLD, embedder=embedder, conn=conn)

        # Rank clusters by size and take the top N for the LLM prompt.
        clusters_sorted = sorted(clusters, key=len, reverse=True)
        top_clusters = clusters_sorted[:MAX_CLUSTERS_SHOWN]
        cluster_payload = [
            {"size": len(c), "sample_titles": [p["title"] for p in c[:3]]}
            for c in top_clusters
        ]

        log.info("split-check '%s' (%d members): top clusters sizes=%s",
                 cat["name"], current_count,
                 [c["size"] for c in cluster_payload[:5]])

        candidates.append({
            "cat": cat,
            "current_count": current_count,
            "top_clusters": top_clusters,
            "cluster_payload": cluster_payload,
        })

    if not candidates:
        return

    # Phase 2 (parallel, no DB): fan out decide_split LLM calls across
    # all candidates. Cap concurrency at 5 — the global semaphore in
    # llm.py provides additional process-wide ceiling.
    decisions = parallel_namer_calls(
        (
            cand["cat"]["id"],
            (lambda c=cand: namer.decide_split(
                c["cat"]["name"], c["cat"]["description"] or "",
                c["current_count"], c["cluster_payload"],
            )),
        )
        for cand in candidates
    )

    # Phase 3 (sequential, under merge_lock): yield events using the
    # decisions. Update the snapshot ONLY when the LLM returned a
    # decision — a failed LLM call leaves the candidate eligible for
    # the next sweep instead of silently suppressing it.
    for cand in candidates:
        cat = cand["cat"]
        cat_id = cat["id"]
        top_clusters = cand["top_clusters"]
        decision = decisions.get(cat_id)
        if decision is None:
            log.warning(
                "split-check LLM error or no decision for '%s' — leaving "
                "snapshot unchanged so we retry next sweep", cat["name"],
            )
            continue

        # LLM returned a decision (keep or split) — bump the snapshot so
        # we don't re-ask the LLM until the category grows by another
        # SPLIT_RECHECK_DELTA. This applies to BOTH "keep" and "split"
        # outcomes: the LLM's "keep" is meaningful and shouldn't be
        # overridden until more evidence accumulates.
        conn.execute(
            "UPDATE categories SET painpoint_count_at_last_check = ? WHERE id = ?",
            (cand["current_count"], cat_id),
        )

        log.info("split-check LLM decision for '%s': %s (%d subs) — %s",
                 cat["name"], decision.decision, len(decision.subcategories),
                 decision.reason[:80])

        if decision.decision != "split":
            continue

        # Build the subcategory payload, resolving cluster_indices back to
        # painpoint ids. The LLM may group multiple clusters into one
        # sub-category.
        subcat_payload = []
        covered_cluster_ids = set()
        for sub in decision.subcategories:
            indices = [i for i in sub.cluster_indices if 0 <= i < len(top_clusters)]
            if not indices:
                continue
            covered_cluster_ids.update(indices)
            pp_ids = []
            for i in indices:
                pp_ids.extend(p["id"] for p in top_clusters[i])
            if not pp_ids:
                continue
            subcat_payload.append({
                "name": sub.name,
                "description": sub.description,
                "painpoint_ids": pp_ids,
            })

        if len(subcat_payload) < 2:
            # LLM said split but didn't provide ≥2 usable sub-groups. Skip.
            continue

        yield CategoryEvent(
            event_type="add_category_split",
            payload={
                "parent_category_id": cat["parent_id"],
                "source_category_id": cat["id"],
                "source_category_name": cat["name"],
                "subcategories": subcat_payload,
                "llm_reason": decision.reason,
            },
            target_category=cat["id"],
            metric_name="llm_split_decision",
            metric_value=float(len(subcat_payload)),
            threshold=2.0,
        )


# ---------------------------------------------------------------------------
# Step 3 — delete dead categories
# ---------------------------------------------------------------------------


def _category_last_activity(conn, category_id):
    """Max `painpoints.last_updated` across a category's current members.

    Returns a timezone-aware UTC datetime, or None if the category has no
    members. `last_updated` fires whenever a painpoint is created, moved,
    or linked to a new pending source — it's the right proxy for "when
    did this category last see a membership event."
    """
    row = conn.execute(
        "SELECT MAX(last_updated) AS ts FROM painpoints WHERE category_id = ?",
        (category_id,),
    ).fetchone()
    if row is None or row["ts"] is None:
        return None
    return _parse_iso(row["ts"])


def propose_delete_events(conn, now=None):
    """Step 3: propose `delete_category` for any non-Uncategorized category
    whose most recently updated member is older than CATEGORY_STALE_DAYS.

    "Stale" replaces the old mass-based threshold, which was a size test
    dressed up as a quality test (tiny-but-valid categories were getting
    culled). Staleness tracks membership activity directly.

    Unborn categories (no members yet) are left alone — they're unused,
    not dead.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=CATEGORY_STALE_DAYS)

    rows = conn.execute(
        "SELECT id, name, parent_id FROM categories WHERE name != ?",
        (UNCATEGORIZED_NAME,),
    ).fetchall()
    for cat in rows:
        last_activity = _category_last_activity(conn, cat["id"])
        if last_activity is None:
            continue   # unborn
        if last_activity >= cutoff:
            continue   # fresh
        age_days = (now - last_activity).total_seconds() / 86400.0
        yield CategoryEvent(
            event_type="delete_category",
            payload={
                "category_id": cat["id"],
                "category_name": cat["name"],
                "parent_id": cat["parent_id"],
            },
            target_category=cat["id"],
            metric_name="category_age_days",
            metric_value=age_days,
            threshold=float(CATEGORY_STALE_DAYS),
        )


# ---------------------------------------------------------------------------
# Step 4 — merge duplicate sibling categories
# ---------------------------------------------------------------------------


def propose_merge_events(conn, embedder=None):
    """Step 4: for each pair of sibling categories under the same parent,
    compute inter-category embedding similarity. If above
    MERGE_CATEGORY_THRESHOLD, propose merge_categories.
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
                sim = inter_category_similarity(conn, a, b, embedder=embedder)
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
    """Re-check staleness under the lock: a member added between propose
    and apply keeps the category alive.
    """
    cat_id = event.payload["category_id"]
    last_activity = _category_last_activity(conn, cat_id)
    if last_activity is None:
        return True, "no members"
    cutoff = datetime.now(timezone.utc) - timedelta(days=CATEGORY_STALE_DAYS)
    if last_activity >= cutoff:
        return False, f"member activity at {last_activity.isoformat()} is fresh"
    return True, "still stale"


# ---------------------------------------------------------------------------
# Step 5 — per-painpoint reroute (handles singleton mis-routings)
# ---------------------------------------------------------------------------


def propose_reroute_events(conn, embedder=None):
    """Step 5: for every categorised painpoint, find its best-matching
    category by embedding cosine. If that category differs from the current
    one by at least REROUTE_MARGIN, propose a reroute.

    Uses a LEAVE-ONE-OUT centroid to estimate the current-category fit —
    otherwise the painpoint's own embedding dominates the normalized
    centroid it's being compared against (especially in small categories),
    artificially inflating current_sim and blocking every reroute.
    Singleton categories (1 member) have no leave-one-out centroid, so
    we treat current_sim = 0 (any plausible alternative wins).

    Scope:
    - Skips painpoints in Uncategorized (step 1 handles those).
    - Skips reroutes *to* Uncategorized (don't demote).
    """
    from .embeddings import (
        find_best_category_ranked,
        iter_category_member_embeddings,
        leave_one_out_centroid_sim,
    )

    uncat_id = _uncategorized_id(conn)

    cat_rows = conn.execute(
        """
        SELECT DISTINCT p.category_id AS cat_id
        FROM painpoints p
        JOIN painpoint_vec v ON v.rowid = p.id
        WHERE p.category_id IS NOT NULL
          AND p.category_id != COALESCE(?, -1)
        """,
        (uncat_id,),
    ).fetchall()

    for cat_row in cat_rows:
        current_cat = cat_row["cat_id"]
        # Batch-load all member embeddings once — reused across every
        # painpoint in this category for leave-one-out.
        members = list(iter_category_member_embeddings(conn, current_cat))
        if not members:
            continue

        for pp_id, emb in members:
            loo_sim = leave_one_out_centroid_sim(members, pp_id, emb)
            current_sim = 0.0 if loo_sim is None else loo_sim

            ranked = find_best_category_ranked(conn, emb, limit=50)
            best_other_id = None
            best_other_sim = -1.0
            for cat_id, sim in ranked:
                if cat_id == current_cat or cat_id == uncat_id:
                    continue
                if sim > best_other_sim:
                    best_other_id = cat_id
                    best_other_sim = sim

            if best_other_id is None:
                continue
            margin = best_other_sim - current_sim
            if margin < REROUTE_MARGIN:
                continue

            yield CategoryEvent(
                event_type="reroute_painpoint",
                payload={
                    "painpoint_id": pp_id,
                    "from_category_id": current_cat,
                    "to_category_id": best_other_id,
                    "current_sim": current_sim,
                    "best_sim": best_other_sim,
                },
                target_category=best_other_id,
                triggering_pp=pp_id,
                metric_name="reroute_margin",
                metric_value=margin,
                threshold=REROUTE_MARGIN,
            )


def _test_reroute_painpoint(conn, event):
    """Cascade-safe: confirm the painpoint is still in from_category and the
    target category still exists. An earlier reroute in the same sweep may
    have moved the painpoint already, or a delete may have removed the
    target — reject cleanly in those cases."""
    pp_id = event.payload["painpoint_id"]
    from_id = event.payload["from_category_id"]
    to_id = event.payload["to_category_id"]
    pp_row = conn.execute(
        "SELECT category_id FROM painpoints WHERE id = ?", (pp_id,)
    ).fetchone()
    if pp_row is None:
        return False, "cascade: painpoint deleted"
    if pp_row["category_id"] != from_id:
        return False, (
            f"cascade: pp {pp_id} now in cat {pp_row['category_id']}, "
            f"expected {from_id}"
        )
    target = conn.execute(
        "SELECT id FROM categories WHERE id = ?", (to_id,)
    ).fetchone()
    if target is None:
        return False, f"cascade: target category {to_id} gone"
    return True, f"margin {event.metric_value:.3f} >= {event.threshold:.3f}"


def _apply_reroute_painpoint(conn, event, namer):
    pp_id = event.payload["painpoint_id"]
    from_id = event.payload["from_category_id"]
    to_id = event.payload["to_category_id"]

    conn.execute(
        "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE id = ?",
        (to_id, _now(), pp_id),
    )
    # Both centroids shift: from_id lost a member, to_id gained one.
    update_category_embedding(conn, from_id)
    update_category_embedding(conn, to_id)
    return pp_id


def _test_merge_categories(conn, event):
    """Already gated by MERGE_CATEGORY_THRESHOLD at proposal time.

    Pre-apply check: the merge proposer yields all candidate pairs at
    once, but earlier merges in the same sweep may have deleted one of
    the categories in this pair (cascade). Reject cleanly here rather
    than crashing in apply.
    """
    survivor_id = event.payload["survivor_id"]
    loser_id = event.payload["loser_id"]
    exists = {
        r["id"]
        for r in conn.execute(
            "SELECT id FROM categories WHERE id IN (?, ?)", (survivor_id, loser_id),
        ).fetchall()
    }
    if survivor_id not in exists or loser_id not in exists:
        return False, (
            f"cascade: survivor {survivor_id} in={survivor_id in exists}, "
            f"loser {loser_id} in={loser_id in exists}"
        )
    return True, "similarity > threshold"


_TESTS = {
    "add_category_new": _test_add_category_new,
    "add_category_split": _test_add_category_split,
    "delete_category": _test_delete_category,
    "merge_categories": _test_merge_categories,
    "reroute_painpoint": _test_reroute_painpoint,
}


def run_acceptance_test(conn, event):
    fn = _TESTS.get(event.event_type)
    if fn is None:
        return False, "unknown event type"
    return fn(conn, event)


# ---------------------------------------------------------------------------
# Apply (under savepoint)
# ---------------------------------------------------------------------------


def _get_taxonomy_for_llm(conn):
    """Build the taxonomy tree for the LLM prompt.

    Returns two things:
    - roots: list of {"name": str, "description": str} for root categories
    - flat: list of {"path": "Root > Child", ...} for the full tree

    The LLM is shown both so it can pick a root to place the new
    category under.
    """
    roots = conn.execute(
        "SELECT name, description FROM categories "
        "WHERE parent_id IS NULL AND name != ? ORDER BY name",
        (UNCATEGORIZED_NAME,),
    ).fetchall()
    children = conn.execute("""
        SELECT c.name AS child, p.name AS parent, c.description
        FROM categories c
        JOIN categories p ON c.parent_id = p.id
        WHERE c.name != ?
        ORDER BY p.name, c.name
    """, (UNCATEGORIZED_NAME,)).fetchall()

    root_list = [{"name": r["name"], "description": r["description"] or ""} for r in roots]
    flat_list = [
        {"path": f"{r['parent']} > {r['child']}", "name": r["child"],
         "description": r["description"] or ""}
        for r in children
    ]
    return root_list, flat_list


def _resolve_parent_id(conn, parent_name):
    """Resolve a parent category name returned by the LLM to a parent_id.

    The LLM sometimes returns the full path format 'Root > Child' instead
    of just the category name. We try both the raw string and the last
    segment after '>'. Matches ANY existing category (root or child).
    Returns None if the name is null/empty or doesn't match anything.
    """
    if not parent_name:
        return None
    candidates = [parent_name.strip()]
    if ">" in parent_name:
        # "Cloud & Infrastructure > Databases" → try "Databases" too
        candidates.append(parent_name.split(">")[-1].strip())
        # Also try the first segment: "Cloud & Infrastructure"
        candidates.append(parent_name.split(">")[0].strip())
    for name in candidates:
        row = conn.execute(
            "SELECT id FROM categories WHERE name = ?", (name,)
        ).fetchone()
        if row is not None:
            return row["id"]
    log.warning(
        "LLM proposed parent=%r but no category matched (tried %s) — "
        "falling back to root placement",
        parent_name, candidates,
    )
    return None


def _apply_add_category_new(conn, event, namer):
    # Use pre-fetched LLM response if available (from prefetch_llm_for_event
    # called concurrently before we entered the savepoint). Falls back to
    # an inline call if not pre-fetched.
    name_resp = event.llm_result
    if name_resp is None:
        _roots, flat = _get_taxonomy_for_llm(conn)
        name_resp = namer.name_new_category(
            event.payload["sample_titles"], event.payload["sample_descriptions"],
            existing_taxonomy=flat,
        )
    name = name_resp.get("name", "").strip()
    description = name_resp.get("description", "").strip()
    parent_name = name_resp.get("parent")
    if not name:
        raise RuntimeError("LLM returned empty category name")

    parent_id = _resolve_parent_id(conn, parent_name)

    # Name collision check — if a category with this name already exists,
    # reuse it instead of creating a duplicate.
    existing = conn.execute(
        "SELECT id FROM categories WHERE name = ?", (name,)
    ).fetchone()
    if existing is not None:
        target_id = existing["id"]
    else:
        conn.execute(
            "INSERT INTO categories (name, parent_id, description, created_at) "
            "VALUES (?, ?, ?, ?)",
            (name, parent_id, description, _now()),
        )
        target_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    for pp_id in event.payload["painpoint_ids"]:
        conn.execute(
            "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE id = ?",
            (target_id, _now(), pp_id),
        )

    # Update the new category's embedding from its members
    update_category_embedding(conn, target_id)

    return target_id


def _apply_add_category_split(conn, event, namer):
    """Apply an LLM-decided split. The payload already contains the
    LLM's sub-category names + descriptions + painpoint groupings
    (decided in propose_split_events). We just materialize them.
    """
    parent_id = event.payload["parent_category_id"]
    source_id = event.payload["source_category_id"]
    subcategories = event.payload["subcategories"]

    new_cat_ids = []
    for sub in subcategories:
        name = (sub.get("name") or "").strip()
        description = (sub.get("description") or "").strip()
        pp_ids = sub.get("painpoint_ids") or []
        if not name or not pp_ids:
            continue

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

        for pp_id in pp_ids:
            conn.execute(
                "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE id = ?",
                (sub_id, _now(), pp_id),
            )
        # Keep the new sub-category's embedding fresh as members populate it.
        update_category_embedding(conn, sub_id)

    # Retire the source category if it's now empty.
    remaining = conn.execute(
        "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (source_id,)
    ).fetchone()[0]
    if remaining == 0 and parent_id is not None:
        # Clean up the source's embedding too, otherwise the orphaned
        # vec row can still be returned by find_best_category and cause
        # a FK failure on the next promote.
        try:
            conn.execute("DELETE FROM category_vec WHERE rowid = ?", (source_id,))
        except Exception:
            pass
        conn.execute("DELETE FROM categories WHERE id = ?", (source_id,))
    elif remaining > 0:
        # Partial split: the LLM didn't put every member into a
        # sub-category, so the source category survives but its member
        # set shrank. Refresh its centroid to match the current members,
        # otherwise subsequent find_best_category calls would use the
        # stale pre-split centroid.
        update_category_embedding(conn, source_id)

    return new_cat_ids


def _apply_delete_category(conn, event, namer):
    cat_id = event.payload["category_id"]
    parent_id = event.payload["parent_id"]

    # Relink any surviving members to the parent (or Uncategorized if no parent).
    fallback_id = parent_id
    if fallback_id is None:
        fallback_id = _uncategorized_id(conn)

    # Check if there are actually any members to move (for the centroid
    # refresh decision below — we only need to refresh the fallback's
    # centroid if it gained new members).
    moved_count = conn.execute(
        "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat_id,)
    ).fetchone()[0]

    conn.execute(
        "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE category_id = ?",
        (fallback_id, _now(), cat_id),
    )
    # Clean up this category's embedding from category_vec before the
    # categories row is deleted — otherwise find_best_category can
    # return the orphaned rowid and the next promote FK-fails.
    try:
        conn.execute("DELETE FROM category_vec WHERE rowid = ?", (cat_id,))
    except Exception:
        pass
    conn.execute("DELETE FROM categories WHERE id = ?", (cat_id,))

    # Refresh the fallback parent's centroid — its member set just grew
    # by `moved_count` painpoints. Skip when no members moved (test-only
    # path where a delete is proposed against an empty category) or when
    # the fallback is Uncategorized (which is centroid-exempt; the
    # update function has its own guard, but we short-circuit for clarity).
    if moved_count > 0:
        update_category_embedding(conn, fallback_id)

    return cat_id


def _apply_merge_categories(conn, event, namer):
    """Apply a sibling-merge event.

    Cascade-safe: earlier merges in the same sweep may have already
    deleted one of the categories in this event (e.g., A→B merged in
    iteration N, then N+1 tries to merge B with C, but B is gone). We
    skip gracefully in that case rather than raising.
    """
    survivor_id = event.payload["survivor_id"]
    loser_id = event.payload["loser_id"]

    survivor_row = conn.execute(
        "SELECT name FROM categories WHERE id = ?", (survivor_id,)
    ).fetchone()
    loser_row = conn.execute(
        "SELECT name FROM categories WHERE id = ?", (loser_id,)
    ).fetchone()
    if survivor_row is None or loser_row is None:
        # One or both categories were merged away by an earlier event in
        # this sweep — skip this one, it's effectively already resolved.
        raise RuntimeError(
            f"merge skipped: survivor={survivor_id} exists={survivor_row is not None}, "
            f"loser={loser_id} exists={loser_row is not None} "
            f"(likely cascade from earlier merge)"
        )
    survivor_name = survivor_row["name"]
    loser_name = loser_row["name"]
    sample_titles = [
        r["title"] for r in conn.execute(
            "SELECT title FROM painpoints WHERE category_id IN (?, ?) LIMIT 10",
            (survivor_id, loser_id),
        ).fetchall()
    ]

    # Repoint painpoints and delete the loser
    conn.execute(
        "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE category_id = ?",
        (survivor_id, _now(), loser_id),
    )
    # Clean up loser's embedding from category_vec
    try:
        conn.execute("DELETE FROM category_vec WHERE rowid = ?", (loser_id,))
    except Exception:
        pass
    conn.execute("DELETE FROM categories WHERE id = ?", (loser_id,))

    # Use pre-fetched LLM response if available, else call inline.
    desc_resp = event.llm_result
    if desc_resp is None:
        try:
            desc_resp = namer.describe_merged_category(
                survivor_name, loser_name, sample_titles,
            )
        except Exception as e:
            log.warning("merge: describe_merged_category failed for '%s' + '%s': %s",
                        survivor_name, loser_name, e)
            desc_resp = None
    new_desc = ""
    if isinstance(desc_resp, dict):
        new_desc = (desc_resp.get("description") or "").strip()
    if new_desc:
        conn.execute(
            "UPDATE categories SET description = ? WHERE id = ?",
            (new_desc, survivor_id),
        )

    # Update the survivor's embedding with the new description
    update_category_embedding(conn, survivor_id)

    return survivor_id


_APPLIERS = {
    "add_category_new": _apply_add_category_new,
    "add_category_split": _apply_add_category_split,
    "delete_category": _apply_delete_category,
    "merge_categories": _apply_merge_categories,
    "reroute_painpoint": _apply_reroute_painpoint,
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
    """Insert an audit row for a category event. Catches its own
    exceptions — the sweep must not die because the audit log failed."""
    try:
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
    except Exception as e:
        log.warning(
            "log_event: failed to write audit row for %s: %s — sweep continues",
            event.event_type, e,
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


# ---------------------------------------------------------------------------
# LLM prefetch — lets the worker parallelize LLM calls across events so
# they happen concurrently instead of serially inside the lock.
# ---------------------------------------------------------------------------


def _prefetch_context(conn, event):
    """Read everything the LLM call needs from the DB, under the merge
    lock. Returns a context dict keyed by what prefetch_llm_one_event
    will need. Split out so DB reads happen under the lock while the
    network call happens outside it."""
    if event.event_type == "add_category_new":
        _roots, flat = _get_taxonomy_for_llm(conn)
        return {"existing_taxonomy": flat}

    if event.event_type == "merge_categories":
        survivor_id = event.payload["survivor_id"]
        loser_id = event.payload["loser_id"]
        survivor_row = conn.execute(
            "SELECT name FROM categories WHERE id = ?", (survivor_id,)
        ).fetchone()
        loser_row = conn.execute(
            "SELECT name FROM categories WHERE id = ?", (loser_id,)
        ).fetchone()
        if survivor_row is None or loser_row is None:
            return None
        sample_titles = [
            r["title"] for r in conn.execute(
                "SELECT title FROM painpoints WHERE category_id IN (?, ?) LIMIT 10",
                (survivor_id, loser_id),
            ).fetchall()
        ]
        return {
            "survivor_name": survivor_row["name"],
            "loser_name": loser_row["name"],
            "sample_titles": sample_titles,
        }

    return None  # event types without LLM calls in their apply path


def prefetch_llm_one_event(event, namer, context):
    """Make the LLM call for a single event using pre-read `context`.
    Stores the result on `event.llm_result`. Safe to call from a
    ThreadPoolExecutor — uses only the namer (thread-safe if the namer's
    underlying client is thread-safe, which OpenAI's SDK is) and the
    pre-read context dict. Does NOT touch the DB connection."""
    if context is None:
        return
    try:
        if event.event_type == "add_category_new":
            event.llm_result = namer.name_new_category(
                event.payload["sample_titles"],
                event.payload["sample_descriptions"],
                existing_taxonomy=context["existing_taxonomy"],
            )
        elif event.event_type == "merge_categories":
            event.llm_result = namer.describe_merged_category(
                context["survivor_name"],
                context["loser_name"],
                context["sample_titles"],
            )
    except Exception as e:
        # Leave llm_result as None — the apply function will fall back
        # to an inline call (or a sane default for merge description).
        log.warning(
            "prefetch LLM call failed for %s: %s — apply will retry inline",
            event.event_type, e,
        )


def parallel_namer_calls(call_specs, max_workers=5):
    """Generic helper: fan out arbitrary LLM calls across a thread pool.

    `call_specs` is an iterable of `(key, callable)` pairs. The callable
    must take no arguments (use functools.partial / lambda / closure).
    Returns a dict `{key: result_or_None}` — failures are logged and
    return None for that key (so callers can decide whether to skip,
    fall back, or treat as soft failure).

    Used by:
      - prefetch_llm_batch (apply-phase LLM calls bound to CategoryEvent)
      - propose_split_events (decide_split LLM call per candidate)
      - any future per-item LLM fan-out

    Network calls happen in parallel; the global OpenAI semaphore in
    llm.py caps total in-flight calls regardless of how many pools
    exist. Each call is INDEPENDENT — must not share a DB connection.
    """
    import concurrent.futures as cf

    specs = list(call_specs)
    if not specs:
        return {}

    results = {}

    def _run_one(key, fn):
        try:
            return key, fn()
        except Exception as e:
            log.warning("parallel_namer_calls: %r failed: %s", key, e)
            return key, None

    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_one, k, fn) for k, fn in specs]
        for f in futures:
            k, v = f.result()
            results[k] = v
    return results


def prefetch_llm_batch(conn, events, namer, max_workers=5):
    """Fan out LLM calls for a list of events across a thread pool so
    they happen concurrently instead of serially inside apply_with_test.

    Contexts (DB reads) are built sequentially under the caller's lock.
    The network calls happen in parallel without touching the DB.
    Results are stored on each event's `llm_result` attribute.

    Events whose type doesn't need an LLM call are silently skipped.
    """
    import concurrent.futures as cf

    # Read contexts sequentially (needs DB conn, fast)
    contexts = [(ev, _prefetch_context(conn, ev)) for ev in events]

    # Fire LLM calls in parallel (no DB, pure network)
    pending = [(ev, ctx) for ev, ctx in contexts if ctx is not None]
    if not pending:
        return
    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(prefetch_llm_one_event, ev, namer, ctx)
            for ev, ctx in pending
        ]
        # Wait for all to finish — exceptions are swallowed inside
        # prefetch_llm_one_event, so we just block on completion.
        for f in futures:
            f.result()
