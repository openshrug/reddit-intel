"""Category worker event types, proposers, and acceptance tests
(§5 of docs/_internal/PAINPOINT_INGEST_PLAN.md).

The worker runs four passes per sweep, each calling one of the
`propose_*_events` generators below. Each yielded event is then run
through `apply_with_test`, which evaluates the event's domain-specific
test FIRST and only enters a savepoint to apply the mutation if the test
passed (per §5.4 — savepoints protect against mid-apply failures, NOT
against test rejection, because every test is pre-mutation).
"""

import concurrent.futures as _cf
import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

from datetime import datetime, timedelta, timezone

from . import UNCATEGORIZED_NAME, _now, in_clause_placeholders, uncategorized_id
from .category_clustering import (
    category_member_titles,
    cluster_painpoints,
    inter_category_similarity,
)
from .category_retrieval import (
    CROSS_PARENT_REPARENT_MARGIN,
    CROSS_PARENT_REPARENT_MIN_COS,
    SIMILAR_CATEGORY_THRESHOLD,
    category_dense_cos,
    delete_category_fts,
    find_hybrid_candidates,
    find_similar_category,
    sync_category_fts,
)
from .embeddings import (
    CATEGORY_COSINE_THRESHOLD,
    MERGE_COSINE_THRESHOLD,
    add_member_to_centroid,
    delete_category_anchor,
    find_best_category_ranked,
    get_painpoint_embedding,
    iter_category_member_embeddings,
    leave_one_out_centroid_sim,
    rebuild_centroid_from_members,
    remove_member_from_centroid,
    store_category_anchor,
    update_category_embedding,
)
from .relevance import _parse_iso

MIN_SUB_CLUSTER_SIZE = 3
# Maximum fan-out for parallel LLM calls (prefetch + decide_split +
# uncat-review + painpoint_merge). Raised from 5 → 30 after observing
# uncat-review was the dominant sweep cost at ~135 calls × 2s / 5
# workers ≈ 55s. With 30 workers the same batch finishes in ~10s.
# The OPENAI_COMPLETION_SEMAPHORE (see llm.py) is sized to match —
# the sweep runs standalone (no concurrent extraction), so there's no
# other consumer to contend with on the completion bucket. Embeddings
# go through OPENAI_EMBEDDING_SEMAPHORE (separate, larger pool).
_LLM_PARALLEL_WORKERS = 30
# Lowered from 5 after a live-sweep pass over 145 Uncategorized
# members fired 0 add_category_new events: clusters of 3-4 related
# painpoints (screenshot tooling, paywall design, etc.) sat just below
# the old floor. 3 still excludes singletons/pairs that would name a
# category over noise.
SPLIT_RECHECK_DELTA = 10
MERGE_CATEGORY_THRESHOLD = 0.80
# Root-level pairs get a looser threshold: LLM-review fans out per
# painpoint and can't coordinate, so it spawns semantically-overlapping
# roots (e.g. 11 parallel "Dating X" roots in one sweep) whose anchor
# descriptions diverge enough to sit in the 0.70-0.78 band — real
# duplicates but just below 0.80. Child-under-parent pairs keep the
# stricter 0.80 because those are already scoped to one parent's topic
# and an 0.70 match there would over-merge genuine siblings.
MERGE_ROOT_CATEGORY_THRESHOLD = 0.70
# Painpoint-pair dedup at sweep time — embedding cosine is the CANDIDATE
# filter, LLM is the TIE-BREAKER. The promoter runs at
# MERGE_COSINE_THRESHOLD=0.60 and misses "same pain, different wording"
# pairs that cluster in the 0.50-0.60 zone. This sweep pass picks up
# those residual duplicates within the same category, confirmed by an
# LLM boolean call per candidate. Per-category caps the cost at O(N²)
# local compute + at most K LLM calls per category.
PAINPOINT_DEDUP_CANDIDATE_THRESHOLD = 0.50
# Max candidate pairs examined per category per sweep. Bounds LLM cost
# (~$0.001 per call × MAX_PAINPOINT_MERGES_PER_CATEGORY × N_categories).
# Within a single category, we rank pairs by cosine DESC and cut at this
# many — the highest-cos pairs are the most-likely duplicates.
MAX_PAINPOINT_MERGES_PER_CATEGORY = 20
# A category with no member added/updated in this many days is stale —
# propose delete. Replaces the old mass-based threshold (which was a size
# test dressed up as a quality test).
CATEGORY_STALE_DAYS = 30

# Step 5 (reroute). A painpoint is moved to a non-current category only
# when the alternative's cosine sim exceeds the current sim by at least
# REROUTE_MARGIN. Conservative so we don't thrash near-ties. Fixes
# residual singleton hijacking that the split step can't cluster.
# Raised from 0.08 → 0.10 after Phase-1 (reroute-includes-Uncat):
# with the full corpus now eligible for reroute, the old margin fired
# ~47 near-tie ping-pongs even at sweep 5 without Uncat pressure.
# 0.10 kills the tail while still letting genuine mis-routings move
# (real-world margins seen in logs: 0.15-0.55).
REROUTE_MARGIN = 0.10


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


# How many Uncategorized singletons the LLM reviews per sweep.
# None = unbounded (review every singleton). Capped int still works for
# tests that want to bound fan-out (see test_max_reviews_bounds_llm_calls).
# Removed the 50-cap after observing that multi-sweep runs plateau at
# ~43% Uncategorized: each sweep the LLM is handed the top-50 by
# signal×severity, approves only 3-8, and the remaining unreviewed tail
# never gets attention. At ~150 pendings × gpt-4.1-mini this costs
# ~$0.06 per sweep — bounded by the data volume itself, not a magic
# number. Reroute-includes-Uncat (see propose_reroute_events) removes
# most of the pressure anyway: many former singletons are now pulled
# into existing categories before the LLM sees them.
MAX_UNCAT_LLM_REVIEWS = None


def propose_uncategorized_events(conn, embedder=None):
    """Step 1: cluster the Uncategorized bucket via embedding cosine sim,
    yield one `add_category_new` event per cluster of size >=
    MIN_SUB_CLUSTER_SIZE.

    Singletons stay in Uncategorized waiting for siblings to arrive
    (or for the LLM-review pass to promote them — see
    propose_uncategorized_singleton_events below).
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
# Step 1b — LLM review of remaining Uncategorized singletons
# ---------------------------------------------------------------------------


def propose_uncategorized_singleton_events(
    conn, namer=None, embedder=None, max_reviews=MAX_UNCAT_LLM_REVIEWS,
):
    """For Uncategorized painpoints that weren't swept into a cluster,
    ask the LLM whether each warrants its own category. Emits
    `add_category_new` events for `create` decisions (with llm_result
    pre-populated so the apply step skips the redundant naming call).

    Prioritised by `signal_count × severity` — highest-signal Uncat
    painpoints get reviewed first. Capped by `max_reviews` so cost
    stays bounded (~$0.02 per sweep at gpt-4.1-mini rates).

    Painpoints that ended up in a qualifying cluster (Step 1) are
    excluded — we don't double-review painpoints already being
    promoted via clustering.
    """
    if namer is None:
        from .llm_naming import LLMNamer
        namer = LLMNamer()

    uncat_id = _uncategorized_id(conn)
    if uncat_id is None:
        return

    # Work out which Uncat painpoints already got picked up by clustering
    # this sweep — easier to recompute clusters locally than thread the
    # result through, and the clustering pass is cheap against
    # painpoint_vec.
    all_members = [
        dict(r) for r in conn.execute(
            "SELECT id, title, description, severity, signal_count "
            "FROM painpoints WHERE category_id = ?",
            (uncat_id,),
        ).fetchall()
    ]
    if not all_members:
        return

    clusters = cluster_painpoints(
        all_members, threshold=MERGE_COSINE_THRESHOLD,
        embedder=embedder, conn=conn,
    )
    clustered_ids = {
        p["id"] for cluster in clusters if len(cluster) >= MIN_SUB_CLUSTER_SIZE
        for p in cluster
    }

    singletons = [m for m in all_members if m["id"] not in clustered_ids]
    if not singletons:
        return

    # Pre-filter: painpoints whose nearest existing non-Uncat category is
    # already above CATEGORY_COSINE_THRESHOLD would have been routed there
    # at creation time if that category had existed — now that the
    # taxonomy has grown (seed additions, prior sweeps), the reroute step
    # will pull them into the right bucket without needing an LLM
    # opinion. Skipping them slashes the per-sweep LLM bill (was the
    # dominant cost: 170+ create-decisions in one run) and kills the
    # "LLM mints a new category whose anchor is a 0.6 cousin of an
    # existing one" feedback loop — the exact source of root sprawl.
    #
    # Singletons with cosine < CATEGORY_COSINE_THRESHOLD to the best
    # category are genuinely homeless: reroute cannot help (it would
    # match Uncategorized-or-nothing), so an LLM opinion is the only way
    # they ever leave the bucket. Those are the ones we review.
    homeless = []
    nearest_hints = {}
    for m in singletons:
        emb = get_painpoint_embedding(conn, m["id"])
        if emb is None:
            homeless.append(m)  # no vec row — treat as homeless
            continue
        ranked = find_best_category_ranked(conn, emb, limit=1)
        if not ranked:
            homeless.append(m)
            continue
        best_id, best_cos = ranked[0]
        if best_cos >= CATEGORY_COSINE_THRESHOLD:
            continue  # reroute will handle this one
        homeless.append(m)
        name_row = conn.execute(
            "SELECT name FROM categories WHERE id = ?", (best_id,),
        ).fetchone()
        if name_row is not None:
            nearest_hints[m["id"]] = (name_row["name"], best_cos)

    if not homeless:
        return

    # Prioritise by signal * severity. Null severity defaults to 5
    # (historical data; shouldn't happen with the hardened save path).
    def _priority(m):
        return (m["signal_count"] or 1) * (m["severity"] or 5)
    homeless.sort(key=_priority, reverse=True)
    batch = homeless if max_reviews is None else homeless[:max_reviews]

    _roots, flat = _get_taxonomy_for_llm(conn)

    # Fan out the LLM calls in parallel. Each call is independent; the
    # per-painpoint decision doesn't depend on the others.
    decisions, _errors = parallel_namer_calls(
        (
            m["id"],
            (lambda m=m: namer.decide_uncategorized(
                m["title"], m["description"] or "",
                m["signal_count"] or 1, m["severity"] or 5,
                flat,
                nearest_hint=nearest_hints.get(m["id"]),
            )),
        )
        for m in batch
    )

    for m in batch:
        decision = decisions.get(m["id"])
        if decision is None:
            continue
        if decision.action != "create":
            continue
        name = (decision.name or "").strip()
        desc = (decision.description or "").strip()
        parent = (decision.parent or "").strip() or None
        if not name or not desc:
            log.warning(
                "uncat-review: LLM said create for pp=%s but omitted "
                "name/description — skipping", m["id"],
            )
            continue

        yield CategoryEvent(
            event_type="add_category_new",
            payload={
                "painpoint_ids": [m["id"]],
                "sample_titles": [m["title"]],
                "sample_descriptions": [m["description"] or ""],
            },
            target_category=uncat_id,
            triggering_pp=m["id"],
            metric_name="uncat_llm_review",
            metric_value=float(_priority(m)),
            threshold=0.0,
            # Pre-populate the llm_result so _apply_add_category_new
            # reuses this decision instead of making a fresh naming
            # call inside the savepoint.
            llm_result={
                "name": name,
                "description": desc,
                "parent": parent,
            },
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
    decisions, _errors = parallel_namer_calls(
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
    """When this category's MEMBER SET last changed (add / remove / move).

    Reads `categories.member_set_last_changed_at` — a dedicated stamp
    bumped only by the incremental-centroid helpers. Using
    MAX(painpoints.last_updated) conflated "real membership activity"
    with "duplicate-pending bumps" (signal_count ++ fires last_updated
    but doesn't change the member set), so a spam of repeat hits could
    keep a semantically dead category looking fresh.

    Returns a timezone-aware UTC datetime, or None if the category has
    no members (category row exists but member_set_last_changed_at is
    NULL). Falls back to the old MAX(last_updated) signal only when the
    dedicated stamp is NULL *and* there are members — handles legacy
    DBs where the migration ran but no mutation has touched this
    category yet.
    """
    row = conn.execute(
        "SELECT member_set_last_changed_at FROM categories WHERE id = ?",
        (category_id,),
    ).fetchone()
    if row is not None and row["member_set_last_changed_at"]:
        return _parse_iso(row["member_set_last_changed_at"])

    fallback = conn.execute(
        "SELECT MAX(last_updated) AS ts FROM painpoints WHERE category_id = ?",
        (category_id,),
    ).fetchone()
    if fallback is None or fallback["ts"] is None:
        return None
    return _parse_iso(fallback["ts"])


def propose_delete_events(conn, now=None):
    """Step 3: propose `delete_category` for non-Uncategorized categories
    that are either (a) drained empty by reroute, or (b) stale past
    CATEGORY_STALE_DAYS.

    The empty-but-drained path catches ghost categories that reroute
    emptied mid-sweep — without it, a category with zero members sticks
    around for 30 days cluttering the tree (observed 50 such ghosts
    after 3 sweeps). The staleness path still handles populated-but-idle
    categories.

    Unborn categories (0 members, never had any) are left alone — they
    might be seed shelves awaiting population. We tell "unborn" from
    "drained" by `member_set_last_changed_at`: set once, drained → delete;
    still NULL → unborn, keep.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=CATEGORY_STALE_DAYS)

    rows = conn.execute(
        "SELECT id, name, parent_id, member_set_last_changed_at "
        "FROM categories WHERE name != ?",
        (UNCATEGORIZED_NAME,),
    ).fetchall()
    for cat in rows:
        member_count = conn.execute(
            "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat["id"],)
        ).fetchone()[0]

        # Drained-empty path: 0 members AND once had members (a real
        # delete, not an unborn seed shelf).
        if member_count == 0:
            if cat["member_set_last_changed_at"] is None:
                continue   # unborn seed shelf
            yield CategoryEvent(
                event_type="delete_category",
                payload={
                    "category_id": cat["id"],
                    "category_name": cat["name"],
                    "parent_id": cat["parent_id"],
                },
                target_category=cat["id"],
                metric_name="member_count",
                metric_value=0.0,
                threshold=0.0,
            )
            continue

        # Staleness path: populated but idle.
        last_activity = _category_last_activity(conn, cat["id"])
        if last_activity is None:
            continue   # unborn (shouldn't hit given member_count > 0, but safe)
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
    """Step 4: for each pair of categories sharing a parent (including
    root-level pairs where parent is NULL), compute inter-category
    embedding similarity. Above MERGE_CATEGORY_THRESHOLD → merge.

    Previously only non-NULL parents were compared, so the LLM review
    could spawn 11 parallel "Dating X" roots and none would ever merge
    (each call sees the taxonomy independently and can't coordinate).
    Allowing NULL-parent pairs lets the merge step collapse that sprawl.
    """
    parent_ids = [
        r["parent_id"] for r in conn.execute(
            "SELECT DISTINCT parent_id FROM categories"
        ).fetchall()
    ]
    for parent_id in parent_ids:
        if parent_id is None:
            siblings = conn.execute(
                "SELECT id, name FROM categories "
                "WHERE parent_id IS NULL AND name != ? ORDER BY id",
                (UNCATEGORIZED_NAME,),
            ).fetchall()
            threshold = MERGE_ROOT_CATEGORY_THRESHOLD
        else:
            siblings = conn.execute(
                "SELECT id, name FROM categories "
                "WHERE parent_id = ? AND name != ? ORDER BY id",
                (parent_id, UNCATEGORIZED_NAME),
            ).fetchall()
            threshold = MERGE_CATEGORY_THRESHOLD
        ids = [s["id"] for s in siblings]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                sim = inter_category_similarity(conn, a, b, embedder=embedder)
                if sim < threshold:
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
                    threshold=threshold,
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
    """Re-check under the lock so a member added between propose and
    apply keeps the category alive. Two acceptance paths:

    - member_count event: must still be 0 members. Something routed in
      after propose → reject so we don't delete a populated category.
    - category_age_days event: must still be stale (no activity since
      cutoff). A fresh member bump since propose → reject.
    """
    cat_id = event.payload["category_id"]
    member_count = conn.execute(
        "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat_id,)
    ).fetchone()[0]

    if event.metric_name == "member_count":
        if member_count > 0:
            return False, f"cascade: {member_count} members arrived after propose"
        return True, "still empty"

    # staleness path
    if member_count == 0:
        return True, "no members"
    last_activity = _category_last_activity(conn, cat_id)
    if last_activity is None:
        return True, "no activity stamp"
    cutoff = datetime.now(timezone.utc) - timedelta(days=CATEGORY_STALE_DAYS)
    if last_activity >= cutoff:
        return False, f"member activity at {last_activity.isoformat()} is fresh"
    return True, "still stale"


# ---------------------------------------------------------------------------
# Step 4.5 — painpoint-level dedup inside a category
# Residual duplicates that the promoter's 0.60 cosine merge missed
# (because wording varied enough to push cos into the 0.50-0.60 zone).
# Candidate pairs come from a within-category pairwise cosine scan;
# each candidate is confirmed by a cheap LLM boolean call before merge.
# ---------------------------------------------------------------------------


def propose_painpoint_merge_events(conn, namer=None, embedder=None):
    """For each category, find painpoint pairs with cosine ≥
    PAINPOINT_DEDUP_CANDIDATE_THRESHOLD and LLM-confirm whether they
    are duplicates. Emits one `painpoint_merge` event per confirmed pair.

    Within-category only: merging across categories would require a
    category move + a merge, and the category placement is handled by
    reroute. Keeping the scan local also bounds the O(N²) pairwise
    comparison to at most MAX_MEMBERS_PER_CATEGORY² per category.
    """
    if namer is None:
        from .llm_naming import LLMNamer
        namer = LLMNamer()

    uncat_id = _uncategorized_id(conn)
    cat_rows = conn.execute(
        "SELECT id FROM categories WHERE id != COALESCE(?, -1)",
        (uncat_id,),
    ).fetchall()

    # Phase 1: build candidate pairs from pairwise cosine (no LLM).
    # We collect (cat_id, pp_a, pp_b, cos) tuples, ranking by cos DESC
    # within each category, capped at MAX_PAINPOINT_MERGES_PER_CATEGORY.
    candidates_by_cat = {}
    for cat_row in cat_rows:
        cat_id = cat_row["id"]
        members = list(iter_category_member_embeddings(conn, cat_id))
        if len(members) < 2:
            continue

        meta_rows = conn.execute(
            f"SELECT id, title, description, signal_count "
            f"FROM painpoints WHERE id IN "
            f"({in_clause_placeholders(len(members))})",
            [m[0] for m in members],
        ).fetchall()
        meta = {r["id"]: r for r in meta_rows}

        pairs = []
        for i in range(len(members)):
            id_i, emb_i = members[i]
            for j in range(i + 1, len(members)):
                id_j, emb_j = members[j]
                cos = sum(x * y for x, y in zip(emb_i, emb_j))
                if cos >= PAINPOINT_DEDUP_CANDIDATE_THRESHOLD:
                    pairs.append((cos, id_i, id_j))

        if not pairs:
            continue
        pairs.sort(reverse=True)   # highest cos first
        candidates_by_cat[cat_id] = (pairs[:MAX_PAINPOINT_MERGES_PER_CATEGORY], meta)

    # Phase 2: LLM-confirm each candidate in parallel (fan out across all
    # categories, not just within one). Each decision is independent of
    # the rest, so the global fan-out is safe.
    specs = []
    pair_map = {}   # key -> (cat_id, pp_a, pp_b, cos, meta_a, meta_b)
    for cat_id, (pairs, meta) in candidates_by_cat.items():
        for cos, id_i, id_j in pairs:
            a, b = meta[id_i], meta[id_j]
            key = (cat_id, id_i, id_j)
            pair_map[key] = (cat_id, id_i, id_j, cos, a, b)
            specs.append(
                (key,
                 (lambda a=a, b=b: namer.decide_painpoint_merge(
                     a["title"], a["description"] or "",
                     b["title"], b["description"] or "",
                 )))
            )

    if not specs:
        return
    decisions, _errors = parallel_namer_calls(specs)

    # Phase 3: yield events, picking survivor = higher signal_count,
    # tie-break by lower id. Skip painpoints already involved in an
    # earlier event this sweep so cascades don't propose merging
    # already-retired painpoints.
    retired = set()
    for key, (cat_id, id_i, id_j, cos, a, b) in pair_map.items():
        decision = decisions.get(key)
        if decision is None or not decision.duplicates:
            continue
        if id_i in retired or id_j in retired:
            continue
        # Survivor = higher signal_count; tie-break by lower id.
        if (a["signal_count"] or 0) >= (b["signal_count"] or 0):
            survivor_id, loser_id = id_i, id_j
        else:
            survivor_id, loser_id = id_j, id_i
        retired.add(loser_id)
        yield CategoryEvent(
            event_type="painpoint_merge",
            payload={
                "survivor_id": survivor_id,
                "loser_id": loser_id,
                "category_id": cat_id,
            },
            target_category=cat_id,
            triggering_pp=loser_id,
            metric_name="painpoint_merge_cos",
            metric_value=cos,
            threshold=PAINPOINT_DEDUP_CANDIDATE_THRESHOLD,
        )


def _test_painpoint_merge(conn, event):
    """Cascade-safe: survivor and loser must both still exist and still
    be in the claimed category. An earlier event in the sweep may have
    already merged one of them."""
    survivor_id = event.payload["survivor_id"]
    loser_id = event.payload["loser_id"]
    expected_cat = event.payload["category_id"]
    rows = conn.execute(
        f"SELECT id, category_id FROM painpoints WHERE id IN "
        f"({in_clause_placeholders(2)})",
        [survivor_id, loser_id],
    ).fetchall()
    by_id = {r["id"]: r for r in rows}
    if survivor_id not in by_id or loser_id not in by_id:
        return False, "cascade: survivor or loser already retired"
    if by_id[survivor_id]["category_id"] != expected_cat:
        return False, "cascade: survivor moved categories"
    if by_id[loser_id]["category_id"] != expected_cat:
        return False, "cascade: loser moved categories"
    return True, f"cos {event.metric_value:.3f}"


def _apply_painpoint_merge(conn, event, namer, embedder=None):
    """Merge loser into survivor:
      - repoint all painpoint_sources rows from loser to survivor
      - sum signal_count into survivor
      - remove loser from category centroid (incremental delta)
      - delete loser's painpoint_vec and categories-embedding side-effects
      - delete loser from painpoints
    """
    del namer   # no LLM call in apply — decision already made in propose
    survivor_id = event.payload["survivor_id"]
    loser_id = event.payload["loser_id"]
    cat_id = event.payload["category_id"]

    # Aggregate signal_count before repointing sources. Repointing moves
    # the painpoint_sources rows to survivor; duplicates (both painpoints
    # linked to same pending — shouldn't happen but defend) are ignored.
    loser_signal = conn.execute(
        "SELECT signal_count FROM painpoints WHERE id = ?", (loser_id,),
    ).fetchone()["signal_count"]

    conn.execute(
        "UPDATE OR IGNORE painpoint_sources "
        "SET painpoint_id = ? WHERE painpoint_id = ?",
        (survivor_id, loser_id),
    )
    # Any rows that couldn't be repointed (PK collision on
    # (painpoint_id, pending_painpoint_id)) are stragglers — drop them.
    conn.execute(
        "DELETE FROM painpoint_sources WHERE painpoint_id = ?", (loser_id,),
    )

    conn.execute(
        "UPDATE painpoints SET signal_count = signal_count + ?, "
        "last_updated = ? WHERE id = ?",
        (loser_signal or 0, _now(), survivor_id),
    )

    # Centroid update: loser's embedding leaves the category.
    loser_emb = get_painpoint_embedding(conn, loser_id)
    if loser_emb is not None:
        remove_member_from_centroid(conn, cat_id, loser_emb)
    else:
        rebuild_centroid_from_members(conn, cat_id)

    # Delete loser everywhere.
    conn.execute("DELETE FROM painpoint_vec WHERE rowid = ?", (loser_id,))
    conn.execute("DELETE FROM painpoints WHERE id = ?", (loser_id,))

    update_category_embedding(conn, cat_id)
    return survivor_id


# ---------------------------------------------------------------------------
# Step 5 — per-painpoint reroute (handles singleton mis-routings)
# ---------------------------------------------------------------------------


def propose_reroute_events(conn, embedder=None):
    """Step 5: for every categorised painpoint that needs re-checking,
    find its best-matching category by embedding cosine; propose a
    reroute if the alternative beats current by at least REROUTE_MARGIN.

    Skip rule (scales O(N_pps × K_cats) → O(changed_pps × K_cats)):
    a painpoint is safe to skip when all three hold:
      - it has been reroute-checked at least once (`reroute_checked_at`
        is not NULL),
      - its own state hasn't moved since that check
        (painpoints.last_updated <= reroute_checked_at), AND
      - no category's centroid has moved since that check
        (max(categories.centroid_updated_at) <= reroute_checked_at).

    Stamp `reroute_checked_at = now()` for every painpoint we evaluated
    this sweep (whether we yielded a reroute event or not). Painpoints
    that get rerouted will have their last_updated bumped in the
    applier, so they're eligible on the next sweep anyway.

    Uses a LEAVE-ONE-OUT centroid to estimate the current-category fit —
    otherwise the painpoint's own embedding dominates the normalized
    centroid (especially in small categories), artificially inflating
    current_sim and blocking every reroute. Singleton categories (1
    member) have no leave-one-out centroid → current_sim = 0.

    Scans EVERY categorised painpoint including Uncategorized. Uncat
    painpoints have no coherent centroid (Uncategorized is the "no home"
    bucket, not a topic), so their current_sim is forced to 0 and any
    category that matches above REROUTE_MARGIN wins. Previously Uncat
    painpoints were excluded, which left ~40% of painpoints structurally
    unreachable by reroute even when an existing sub-category was a
    cos 0.45+ match. Never reroutes TO Uncategorized.
    """
    uncat_id = _uncategorized_id(conn)
    now = _now()

    # Latest centroid move across the whole tree — a single scalar so we
    # can short-circuit the skip check per painpoint. NULL (never
    # updated) is treated as "always recent enough to force a check".
    max_centroid_row = conn.execute(
        "SELECT MAX(centroid_updated_at) AS ts FROM categories"
    ).fetchone()
    global_last_centroid_move = max_centroid_row["ts"] if max_centroid_row else None

    cat_rows = conn.execute(
        """
        SELECT DISTINCT p.category_id AS cat_id
        FROM painpoints p
        JOIN painpoint_vec v ON v.rowid = p.id
        WHERE p.category_id IS NOT NULL
        """,
    ).fetchall()

    for cat_row in cat_rows:
        current_cat = cat_row["cat_id"]
        members = list(iter_category_member_embeddings(conn, current_cat))
        if not members:
            continue

        # Per-painpoint skip state + text (title + description) for the
        # BM25 side of hybrid retrieval. One batch query keyed by id.
        pp_ids = [pp_id for pp_id, _ in members]
        placeholders = in_clause_placeholders(len(pp_ids))
        meta_rows = conn.execute(
            f"SELECT id, title, description, last_updated, reroute_checked_at "
            f"FROM painpoints WHERE id IN ({placeholders})",
            pp_ids,
        ).fetchall()
        meta = {r["id"]: r for r in meta_rows}

        checked_this_pass = []
        for pp_id, emb in members:
            row = meta.get(pp_id)
            if row is not None and _reroute_skip_safe(
                row["reroute_checked_at"], row["last_updated"],
                global_last_centroid_move,
            ):
                continue

            if current_cat == uncat_id:
                # Uncategorized has no coherent centroid — it's the
                # "no home" bucket, not a topic. Force current_sim = 0
                # so any category above REROUTE_MARGIN wins.
                current_sim = 0.0
            else:
                loo_sim = leave_one_out_centroid_sim(members, pp_id, emb)
                current_sim = 0.0 if loo_sim is None else loo_sim

            # Hybrid retrieval: BM25 (on the painpoint's title +
            # description against category_fts) + dense cosine, fused
            # via RRF. Picks the top-RRF candidate whose dense cosine
            # we then use for the margin test — RRF gives us recall
            # (surfacing keyword-matching categories the dense top-K
            # missed), dense cos stays the accept metric so the
            # REROUTE_MARGIN=0.10 threshold keeps its semantic meaning.
            title = row["title"] if row is not None else ""
            desc = row["description"] if row is not None else ""
            query_text = f"{title or ''} {desc or ''}".strip()
            fused = find_hybrid_candidates(
                conn, emb, query_text,
                exclude_ids={current_cat},  # uncat excluded by the primitive
            )
            if fused:
                best_other_id, best_other_sim, _rrf = fused[0]
            else:
                best_other_id, best_other_sim = None, -1.0

            checked_this_pass.append(pp_id)

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

        # Stamp the checked painpoints so the next sweep can skip them
        # if nothing relevant changes in the meantime.
        if checked_this_pass:
            stamp_placeholders = in_clause_placeholders(len(checked_this_pass))
            conn.execute(
                f"UPDATE painpoints SET reroute_checked_at = ? "
                f"WHERE id IN ({stamp_placeholders})",
                [now, *checked_this_pass],
            )


def _reroute_skip_safe(reroute_checked_at, last_updated, global_last_centroid_move):
    """Return True if a painpoint can safely skip this sweep's reroute
    check. See propose_reroute_events docstring for the rule."""
    if reroute_checked_at is None:
        return False
    if last_updated is not None and last_updated > reroute_checked_at:
        return False
    if (
        global_last_centroid_move is not None
        and global_last_centroid_move > reroute_checked_at
    ):
        return False
    return True


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


def _apply_reroute_painpoint(conn, event, namer, embedder=None):
    del namer, embedder  # reroute is pure DB mutation
    pp_id = event.payload["painpoint_id"]
    from_id = event.payload["from_category_id"]
    to_id = event.payload["to_category_id"]

    emb = get_painpoint_embedding(conn, pp_id)
    conn.execute(
        "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE id = ?",
        (to_id, _now(), pp_id),
    )
    # Incrementally shift both centroids: from_id loses this member,
    # to_id gains it. Falls back to a full rebuild for either side if
    # the painpoint has no embedding row (shouldn't happen in practice
    # but keeps the state consistent).
    if emb is None:
        rebuild_centroid_from_members(conn, from_id)
        rebuild_centroid_from_members(conn, to_id)
    else:
        remove_member_from_centroid(conn, from_id, emb)
        add_member_to_centroid(conn, to_id, emb)
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
    "painpoint_merge": _test_painpoint_merge,
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


def _current_category_for(conn, pp_ids):
    """Return {pp_id: current_category_id} for every painpoint in pp_ids,
    read BEFORE we mutate them. Used by incremental-centroid bookkeeping
    so we know which source category's sum to decrement for each mover.
    """
    if not pp_ids:
        return {}
    placeholders = ",".join("?" * len(pp_ids))
    rows = conn.execute(
        f"SELECT id, category_id FROM painpoints WHERE id IN ({placeholders})",
        pp_ids,
    ).fetchall()
    return {r["id"]: r["category_id"] for r in rows}


def _apply_member_move_delta(conn, pp_ids, prev_category_for, target_id):
    """Update the (sum, count) caches for a batch move: each pp leaves
    its previous category and joins `target_id`. Painpoints that stay
    where they are (prev == target) are skipped. Categories fully drained
    by the move get a full rebuild so their sum resets to zero cleanly.
    """
    sources_touched = set()
    target_delta = 0
    for pp_id in pp_ids:
        prev = prev_category_for.get(pp_id)
        if prev == target_id:
            continue
        emb = get_painpoint_embedding(conn, pp_id)
        if emb is None:
            # No embedding → we can't do an incremental delta for this
            # one. Force a rebuild on both sides at the end.
            sources_touched.add(prev)
            target_delta = None if target_delta is None else None
            continue
        if prev is not None:
            remove_member_from_centroid(conn, prev, emb)
            sources_touched.add(prev)
        add_member_to_centroid(conn, target_id, emb)

    # If we couldn't delta-update cleanly anywhere, rebuild affected cats.
    if target_delta is None:
        rebuild_centroid_from_members(conn, target_id)
    for src in sources_touched:
        if src is None:
            continue
        # If this source ended up empty after the moves, make sure its
        # cached sum is zeroed cleanly (rebuild is cheap for empty cats).
        remaining = conn.execute(
            "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (src,)
        ).fetchone()[0]
        if remaining == 0:
            rebuild_centroid_from_members(conn, src)


def _upsert_category_by_name(conn, name, description, parent_id):
    """Return a category id for `name`, creating the row if it doesn't
    exist yet. Existing rows are reused as-is (description/parent are
    NOT overwritten — that prevents a later LLM pass from rewriting a
    curated seed's fields). Used by both add_category_new and
    add_category_split so the collision-handling is identical.
    """
    existing = conn.execute(
        "SELECT id FROM categories WHERE name = ?", (name,),
    ).fetchone()
    if existing is not None:
        return existing["id"]
    conn.execute(
        "INSERT INTO categories (name, parent_id, description, created_at) "
        "VALUES (?, ?, ?, ?)",
        (name, parent_id, description, _now()),
    )
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    sync_category_fts(conn, new_id, name, description)
    return new_id


def _drop_category_vec_state(conn, category_id):
    """Drop a category's vec rows (main centroid + anchor) ahead of a
    DELETE on the categories row — prevents find_best_category from
    returning an orphaned rowid that would FK-fail the next promote.

    Narrow to sqlite3.Error: if sqlite-vec's virtual table is
    unavailable we want to see the traceback. The previous
    `except Exception: pass` hid real bugs.
    """
    try:
        conn.execute(
            "DELETE FROM category_vec WHERE rowid = ?", (category_id,)
        )
    except sqlite3.Error as exc:
        log.warning(
            "category_vec delete failed for cat_id=%s: %s — continuing",
            category_id, exc,
        )
    delete_category_anchor(conn, category_id)
    delete_category_fts(conn, category_id)


def _resolve_parent_id(conn, parent_name):
    """Resolve a parent category name returned by the LLM to a parent_id.

    Prompt contract: LLM returns the EXACT name of an existing category
    or null. The previous 3-candidate fallback ("maybe it's a path,
    maybe it's the last segment, maybe it's the first segment") hid
    LLM drift instead of surfacing it. One lookup; if it misses we log
    and fall back to root placement — the log line tells us when the
    prompt contract is slipping.
    """
    if not parent_name:
        return None
    name = parent_name.strip()
    # Tolerance carve-out: the LLM occasionally emits "Root > Child"
    # instead of just "Child". Split on '>' and try the last segment
    # as a single retry — keeps us compatible with the most common
    # drift without the previous shotgun approach.
    if ">" in name:
        name = name.split(">")[-1].strip()
    row = conn.execute(
        "SELECT id FROM categories WHERE name = ?", (name,),
    ).fetchone()
    if row is not None:
        return row["id"]
    log.warning(
        "LLM proposed parent=%r but no category matched — "
        "falling back to root placement",
        parent_name,
    )
    return None


def _route_pps_to_category(conn, pp_ids, target_id):
    """Move pp_ids into target_id, updating centroid state + category_vec.
    Shared by the creation gate and the split-absorb path."""
    prev_rows = _current_category_for(conn, pp_ids)
    now = _now()
    for pp_id in pp_ids:
        conn.execute(
            "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE id = ?",
            (target_id, now, pp_id),
        )
    _apply_member_move_delta(conn, pp_ids, prev_rows, target_id)
    update_category_embedding(conn, target_id)


def _maybe_route_to_similar(conn, name, description, pp_ids, embedder):
    """Creation gate. If an existing category is >= SIMILAR_CATEGORY_THRESHOLD
    dense-cosine to the proposed (name, description), move `pp_ids`
    into it and return that category's id. Otherwise return None — the
    caller mints a new category.

    Returns None if `embedder` is None (hermetic test path without
    embedding capability) so tests that don't exercise the gate keep
    their old behavior. Production always passes an embedder.
    """
    if embedder is None:
        return None
    candidates = find_similar_category(conn, name, description, embedder)
    if not candidates:
        return None
    top_id, top_cos, _rrf = candidates[0]
    if top_cos < SIMILAR_CATEGORY_THRESHOLD:
        return None
    _route_pps_to_category(conn, pp_ids, top_id)
    return top_id


def _decide_split_sub_fate(
    conn, name, description, embedder, default_parent_id,
):
    """Decide how to handle a proposed split sub-category.

    Returns one of:
      - ("absorb", existing_cat_id)  — gate fired, route pp_ids there
      - ("replant", better_parent_id) — new sub under a different root
      - ("create", default_parent_id) — new sub under source's parent

    The cross-parent replant path catches cases where split carved out
    a cluster that semantically belongs under a different root than
    the split source's. Example from the live E2E: an AI/ML source was
    split into a sub about "content strategy and market perception";
    those 8 painpoints are really an App Business / marketing topic
    that got trapped under AI/ML because split inherits parent blindly.

    Sharing ONE `embedder.embed()` call across all three decisions
    keeps the per-sub cost to one HTTP round-trip.
    """
    if embedder is None:
        return "create", default_parent_id
    text = f"{name or ''} {description or ''}".strip()
    if not text:
        return "create", default_parent_id

    embedding = embedder.embed(text)
    fused = find_hybrid_candidates(conn, embedding, text)
    if not fused:
        return "create", default_parent_id

    top_id, top_cos, _rrf = fused[0]

    # Gate: near-duplicate → absorb.
    if top_cos >= SIMILAR_CATEGORY_THRESHOLD:
        return "absorb", top_id

    # Replant: does the top candidate live under a different root than
    # the split source, with a decisively better fit?
    if top_cos < CROSS_PARENT_REPARENT_MIN_COS:
        return "create", default_parent_id
    default_cos = category_dense_cos(conn, embedding, default_parent_id)
    if top_cos - default_cos < CROSS_PARENT_REPARENT_MARGIN:
        return "create", default_parent_id

    # Find the top candidate's root. If it IS a root, use it as the
    # new parent; if it's a child, use its parent so new-sub stays at
    # depth-2 (no depth-3 sub-of-sub categories in this taxonomy).
    row = conn.execute(
        "SELECT id, parent_id FROM categories WHERE id = ?", (top_id,),
    ).fetchone()
    if row is None:
        return "create", default_parent_id
    better_parent = row["parent_id"] if row["parent_id"] is not None else row["id"]
    if better_parent == default_parent_id:
        # Top candidate is already under the default parent — no replant.
        return "create", default_parent_id
    return "replant", better_parent


def _apply_add_category_new(conn, event, namer, embedder=None):
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
    moved_ids = event.payload["painpoint_ids"]

    # Creation gate: if a category already exists that's semantically
    # close to the proposed one, route the would-be members to it
    # instead of minting a near-duplicate. See db/category_retrieval.py
    # for the fusion logic; threshold tuned from the E2E sibling-cosine
    # probe (distinct-sibling max 0.53, duplicate floor 0.60).
    target_id = _maybe_route_to_similar(
        conn, name, description, moved_ids, embedder,
    )
    if target_id is None:
        target_id = _upsert_category_by_name(conn, name, description, parent_id)
        # Anchor the category to its declared identity before members land,
        # so update_category_embedding can blend rather than drift with the
        # first few arrivals.
        if embedder is not None:
            store_category_anchor(conn, target_id, name, description, embedder)
    else:
        log.info(
            "add_category_new: '%s' collapsed into existing cat_id=%s "
            "at >= SIMILAR_CATEGORY_THRESHOLD — routing %d members",
            name, target_id, len(moved_ids),
        )
    # Track previous categories so we can decrement their cached sums.
    prev_rows = _current_category_for(conn, moved_ids)
    for pp_id in moved_ids:
        conn.execute(
            "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE id = ?",
            (target_id, _now(), pp_id),
        )
    _apply_member_move_delta(conn, moved_ids, prev_rows, target_id)
    update_category_embedding(conn, target_id)

    return target_id


def _apply_add_category_split(conn, event, namer, embedder=None):
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

        # One hybrid retrieval call decides all three outcomes:
        # - absorb: near-duplicate exists → route pp_ids to it.
        # - replant: the proposed sub semantically belongs under a
        #   DIFFERENT root than the split source's parent. Create
        #   the sub there instead of inheriting the wrong parent.
        # - create: no close match and source's parent is still the
        #   best fit → mint the sub under the default parent.
        action, target = _decide_split_sub_fate(
            conn, name, description, embedder, default_parent_id=parent_id,
        )
        if action == "absorb":
            log.info(
                "add_category_split: sub '%s' collapsed into existing "
                "cat_id=%s at >= SIMILAR_CATEGORY_THRESHOLD — routing "
                "%d members", name, target, len(pp_ids),
            )
            _route_pps_to_category(conn, pp_ids, target)
            continue
        sub_parent_id = target
        if action == "replant":
            log.info(
                "add_category_split: sub '%s' replanted from default "
                "parent=%s to parent=%s based on hybrid-global match",
                name, parent_id, target,
            )

        sub_id = _upsert_category_by_name(conn, name, description, sub_parent_id)
        new_cat_ids.append(sub_id)

        if embedder is not None:
            store_category_anchor(conn, sub_id, name, description, embedder)

        prev_rows = _current_category_for(conn, pp_ids)
        for pp_id in pp_ids:
            conn.execute(
                "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE id = ?",
                (sub_id, _now(), pp_id),
            )
        _apply_member_move_delta(conn, pp_ids, prev_rows, sub_id)
        # Keep the new sub-category's embedding fresh as members populate it.
        update_category_embedding(conn, sub_id)

    # Retire the source category if it's now empty.
    remaining = conn.execute(
        "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (source_id,)
    ).fetchone()[0]
    if remaining == 0 and parent_id is not None:
        _drop_category_vec_state(conn, source_id)
        conn.execute("DELETE FROM categories WHERE id = ?", (source_id,))
    elif remaining > 0:
        # Partial split: the source lost members — its cached sum was
        # already decremented by _apply_member_move_delta, so we just
        # need to re-blend category_vec. If the cache somehow drifted,
        # the sum/count rebuild fallback inside update_category_embedding
        # picks it up.
        update_category_embedding(conn, source_id)

    return new_cat_ids


def _apply_delete_category(conn, event, namer, embedder=None):
    del namer, embedder  # not needed for a pure DB mutation
    cat_id = event.payload["category_id"]
    parent_id = event.payload["parent_id"]

    # Relink any surviving members to the parent (or Uncategorized if no parent).
    fallback_id = parent_id
    if fallback_id is None:
        fallback_id = _uncategorized_id(conn)

    # Grab the IDs BEFORE the move so we can rebuild the fallback's
    # centroid incrementally. Bulk moves are cheaper with a single
    # rebuild-on-target than N individual deltas, so we just rebuild
    # the fallback after the move completes.
    moved_ids = [
        r["id"] for r in conn.execute(
            "SELECT id FROM painpoints WHERE category_id = ?", (cat_id,),
        ).fetchall()
    ]

    conn.execute(
        "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE category_id = ?",
        (fallback_id, _now(), cat_id),
    )
    _drop_category_vec_state(conn, cat_id)
    conn.execute("DELETE FROM categories WHERE id = ?", (cat_id,))

    if moved_ids:
        # Fallback gained a batch of members. Single rebuild is cheaper
        # than N increments for bulk moves. Skips cleanly for
        # Uncategorized (update_category_embedding is a no-op there).
        rebuild_centroid_from_members(conn, fallback_id)
        update_category_embedding(conn, fallback_id)

    return cat_id


def _apply_merge_categories(conn, event, namer, embedder=None):
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

    # Repoint painpoints from loser → survivor, and any children of the
    # loser (for root merges, the loser may own sub-categories) become
    # children of the survivor so the subtree isn't orphaned.
    conn.execute(
        "UPDATE painpoints SET category_id = ?, last_updated = ? WHERE category_id = ?",
        (survivor_id, _now(), loser_id),
    )
    conn.execute(
        "UPDATE categories SET parent_id = ? WHERE parent_id = ?",
        (survivor_id, loser_id),
    )
    _drop_category_vec_state(conn, loser_id)
    conn.execute("DELETE FROM categories WHERE id = ?", (loser_id,))
    # Survivor absorbed the loser's members wholesale — one rebuild is
    # cheaper than per-member deltas for a bulk merge.
    rebuild_centroid_from_members(conn, survivor_id)

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
        # Description changed → re-anchor to match the new text.
        if embedder is not None:
            store_category_anchor(conn, survivor_id, survivor_name, new_desc, embedder)
        # Keep BM25 index in lock-step with the new description so the
        # next creation-gate lookup sees the survivor's expanded scope.
        sync_category_fts(conn, survivor_id, survivor_name, new_desc)

    # Update the survivor's embedding with the new description
    update_category_embedding(conn, survivor_id)

    return survivor_id


_APPLIERS = {
    "add_category_new": _apply_add_category_new,
    "add_category_split": _apply_add_category_split,
    "delete_category": _apply_delete_category,
    "merge_categories": _apply_merge_categories,
    "painpoint_merge": _apply_painpoint_merge,
    "reroute_painpoint": _apply_reroute_painpoint,
}


def apply_event(conn, event, namer, embedder=None):
    fn = _APPLIERS.get(event.event_type)
    if fn is None:
        raise ValueError(f"unknown event type {event.event_type}")
    return fn(conn, event, namer, embedder)


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


def apply_with_test(conn, event, namer, embedder=None):
    """Test first, apply only on accept; savepoint protects against
    mid-apply failures, not against test rejection (per §5.4).

    `embedder` is threaded through to the _apply_* functions that need
    to compute a category anchor embedding (new / split / merged
    categories). When None, anchor writes are skipped and the category
    falls back to pure member-mean behavior — preserves backwards
    compatibility for tests that don't care about the anchor."""
    ok, reason = run_acceptance_test(conn, event)
    if not ok:
        log_event(conn, event, accepted=False, reason=reason)
        return False

    conn.execute("SAVEPOINT cat_event")
    try:
        apply_event(conn, event, namer, embedder)
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


def parallel_namer_calls(call_specs, max_workers=_LLM_PARALLEL_WORKERS):
    """Fan out LLM calls across a thread pool. Returns
    `(results, errors)` where `results` is `{key: result}` for
    successful calls and `errors` is `{key: exception}` for failed
    ones. Previously swallowed exceptions as `None`, which made
    callers unable to distinguish "LLM returned None" from "LLM call
    raised" — the new shape lets them decide.

    Each call is INDEPENDENT — must not share a DB connection. The
    global OpenAI semaphore in llm.py caps total in-flight calls
    regardless of how many pools exist.
    """
    specs = list(call_specs)
    if not specs:
        return {}, {}

    results = {}
    errors = {}

    def _run_one(key, fn):
        try:
            return key, fn(), None
        except Exception as exc:   # noqa: BLE001 — surfaced to caller
            return key, None, exc

    with _cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_one, k, fn) for k, fn in specs]
        for f in futures:
            key, value, exc = f.result()
            if exc is not None:
                errors[key] = exc
                log.warning("parallel_namer_calls: %r failed: %s", key, exc)
            else:
                results[key] = value
    return results, errors


def prefetch_llm_batch(conn, events, namer, max_workers=_LLM_PARALLEL_WORKERS):
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
