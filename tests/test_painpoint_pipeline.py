"""Integration tests for the painpoint ingest pipeline.

Covers every step of the pipeline as described in
docs/PAINPOINT_INGEST_PLAN.md §9 — relevance computation, multi-source
max aggregation, Layer A SQL prefilter (single + multi-match), Layer B
text MinHash, merge_painpoints invariants, source inheritance,
merge-lock thread safety, the four sweep passes (Uncategorized cluster,
split, delete, merge), trigger discipline, idempotency, and the audit
log.

Tests run against a fresh temp DB per function so they don't interfere.
The LLM is always mocked via FakeNamer — no network calls, no API costs.

Run with:  pytest tests/test_painpoint_pipeline.py -v
"""

import json
import threading
import time

import pytest

import db
from db.posts import upsert_post, upsert_comment
from db.painpoints import (
    add_pending_source,
    get_uncategorized_id,
    merge_painpoints,
    promote_pending,
    save_pending_painpoint,
)
from db.relevance import (
    MIN_RELEVANCE_TO_PROMOTE,
    compute_painpoint_relevance,
    compute_pending_relevance,
    per_source_relevance,
)
from db.similarity import (
    PainpointLSH,
    SIM_THRESHOLD,
    exact_source_lookup,
    get_pending_sources,
    make_minhash,
)
from db.category_clustering import cluster_painpoints, inter_category_similarity
from db.category_events import (
    MIN_CATEGORY_RELEVANCE,
    MIN_SUB_CLUSTER_SIZE,
    SPLIT_RECHECK_DELTA,
    SWEEP_CLUSTER_THRESHOLD,
    apply_with_test,
    propose_delete_events,
    propose_merge_events,
    propose_split_events,
    propose_uncategorized_events,
)
from db.llm_naming import FakeNamer
from db.locks import merge_lock
from category_worker import run_sweep


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Per-test temp DB. Each test gets its own SQLite file plus a clean
    LSH pickle alongside it, so no cross-test state leaks."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    yield path


def _make_post(name, *, score=200, num_comments=50, age_seconds=3600,
               title="X", subreddit="test", signal_score=None):
    """Insert a post via upsert_post and optionally set its signal_score."""
    pid = upsert_post({
        "name": name,
        "subreddit": subreddit,
        "title": title,
        "selftext": "",
        "permalink": f"/r/{subreddit}/{name}",
        "score": score,
        "num_comments": num_comments,
        "upvote_ratio": 0.95,
        "created_utc": time.time() - age_seconds,
        "is_self": True,
    })
    if signal_score is not None:
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE posts SET signal_score = ? WHERE id = ?",
                (signal_score, pid),
            )
            conn.commit()
        finally:
            conn.close()
    return pid


def _make_comment(post_id, name, *, score=10, age_seconds=3600, body="hi"):
    return upsert_comment(post_id, {
        "name": name,
        "parent_id": None,
        "author": "u",
        "body": body,
        "score": score,
        "controversiality": 0,
        "permalink": f"/r/test/comments/{name}",
        "created_utc": time.time() - age_seconds,
        "depth": 0,
    })


# ===========================================================================
# Step 1 — relevance computation (§2 of the plan)
# ===========================================================================


class TestRelevance:
    """Per-source relevance formula behaviour (§2.2)."""

    def test_fresh_severe_beats_old_mild(self, fresh_db):
        fresh = _make_post("t3_fresh", score=500, num_comments=200, age_seconds=60)
        old = _make_post("t3_old", score=500, num_comments=200, age_seconds=60 * 86400)

        conn = db.get_db()
        try:
            fresh_post = conn.execute("SELECT * FROM posts WHERE id = ?", (fresh,)).fetchone()
            old_post = conn.execute("SELECT * FROM posts WHERE id = ?", (old,)).fetchone()
            r_fresh_severe = per_source_relevance(fresh_post, None, severity=10)
            r_old_mild = per_source_relevance(old_post, None, severity=2)
            assert r_fresh_severe > r_old_mild * 5
        finally:
            conn.close()

    def test_comment_rooted_gets_boost(self, fresh_db):
        post_id = _make_post("t3_a", score=100, num_comments=10, age_seconds=3600)
        comment_id = _make_comment(post_id, "t1_a", score=80, age_seconds=3600)

        conn = db.get_db()
        try:
            post = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
            comment = conn.execute(
                "SELECT * FROM comments WHERE id = ?", (comment_id,)
            ).fetchone()
            r_post_only = per_source_relevance(post, None, severity=5)
            r_with_comment = per_source_relevance(post, comment, severity=5)
            assert r_with_comment > r_post_only
        finally:
            conn.close()

    def test_signal_score_fallback_when_null(self, fresh_db):
        """When posts.signal_score is NULL, the inline approximation kicks in."""
        without_sig = _make_post("t3_a", score=100, num_comments=20)
        with_sig = _make_post("t3_b", score=100, num_comments=20, signal_score=42.0)

        conn = db.get_db()
        try:
            p_no = conn.execute("SELECT * FROM posts WHERE id = ?", (without_sig,)).fetchone()
            p_yes = conn.execute("SELECT * FROM posts WHERE id = ?", (with_sig,)).fetchone()

            r_no = per_source_relevance(p_no, None, severity=5)
            r_yes = per_source_relevance(p_yes, None, severity=5)

            # The cached column dominates: 42 vs ~3.7 from log1p approx.
            assert r_yes > r_no * 5
        finally:
            conn.close()

    def test_severity_changes_relevance(self, fresh_db):
        post = _make_post("t3_a", score=100, num_comments=10)
        conn = db.get_db()
        try:
            p = conn.execute("SELECT * FROM posts WHERE id = ?", (post,)).fetchone()
            r1 = per_source_relevance(p, None, severity=1)
            r10 = per_source_relevance(p, None, severity=10)
            # severity_mult: 0.6 → 1.5, ratio 2.5
            assert 2.4 < (r10 / r1) < 2.6
        finally:
            conn.close()


class TestRelevanceMaxAggregation:
    """§2.3 — multi-source painpoints aggregate via max, not sum/mean."""

    def test_max_over_sources(self, fresh_db):
        # One fresh+severe source and several old+mild — relevance must equal
        # the fresh one alone (max), not the average and not the sum.
        fresh = _make_post("t3_fresh", score=500, num_comments=200, age_seconds=60)
        old1 = _make_post("t3_old1", score=10, num_comments=2, age_seconds=60 * 86400)
        old2 = _make_post("t3_old2", score=10, num_comments=2, age_seconds=60 * 86400)

        # Save a pending pp on the fresh post, attach old1 and old2 as extras
        pp_id = save_pending_painpoint(fresh, "Some pain", description="d", severity=8)
        add_pending_source(pp_id, old1, comment_id=None)
        add_pending_source(pp_id, old2, comment_id=None)

        # Compute the per-source relevance for the fresh source alone, then
        # the multi-source aggregate. They should be equal.
        conn = db.get_db()
        try:
            fresh_post = conn.execute("SELECT * FROM posts WHERE id = ?", (fresh,)).fetchone()
            fresh_alone = per_source_relevance(fresh_post, None, severity=8)
        finally:
            conn.close()

        agg = compute_pending_relevance(pp_id)
        assert agg == pytest.approx(fresh_alone, rel=1e-6)

    def test_single_source_is_trivial_case(self, fresh_db):
        post_id = _make_post("t3_a", score=100, num_comments=20)
        pp_id = save_pending_painpoint(post_id, "A", description="d", severity=5)

        agg = compute_pending_relevance(pp_id)

        conn = db.get_db()
        try:
            p = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
        finally:
            conn.close()
        direct = per_source_relevance(p, None, severity=5)

        assert agg == pytest.approx(direct, rel=1e-6)


# ===========================================================================
# Step 2 — similarity layers (§3 of the plan)
# ===========================================================================


class TestSimilarityLayerA:
    """§3.1 — exact source SQL prefilter."""

    def test_zero_match_when_no_painpoint_exists(self, fresh_db):
        post_id = _make_post("t3_a")
        pp_id = save_pending_painpoint(post_id, "X", description="d", severity=8)

        conn = db.get_db()
        try:
            sources = get_pending_sources(conn, pp_id)
            matches = exact_source_lookup(conn, sources)
        finally:
            conn.close()

        assert matches == set()

    def test_one_match_via_shared_post(self, fresh_db):
        post_id = _make_post("t3_a", score=200, num_comments=50)
        pp1_id = save_pending_painpoint(post_id, "First pain", severity=8)
        merged_id = promote_pending(pp1_id)
        assert merged_id is not None

        # Second pending pp from the same post — Layer A should find merged_id.
        pp2_id = save_pending_painpoint(
            post_id, "Different wording entirely about something else",
            severity=8,
        )
        conn = db.get_db()
        try:
            sources = get_pending_sources(conn, pp2_id)
            matches = exact_source_lookup(conn, sources)
        finally:
            conn.close()

        assert matches == {merged_id}

    def test_layer_a_short_circuits_layer_b(self, fresh_db):
        """When Layer A finds a match, the new pending must link to it
        deterministically — Layer B is never consulted."""
        post_id = _make_post("t3_a")
        pp1 = save_pending_painpoint(post_id, "First pain", severity=8)
        merged = promote_pending(pp1)

        # Same post, completely different text. Layer A wins regardless of
        # whether Layer B would have matched.
        pp2 = save_pending_painpoint(
            post_id, "Completely unrelated wording xyzzy plugh",
            severity=8,
        )
        result = promote_pending(pp2)

        assert result == merged

        conn = db.get_db()
        try:
            count = conn.execute(
                "SELECT signal_count FROM painpoints WHERE id = ?", (merged,)
            ).fetchone()["signal_count"]
            assert count == 2
        finally:
            conn.close()


class TestSimilarityLayerB:
    """§3.2 — text MinHash + LSH fallback when Layer A misses."""

    def test_near_duplicate_titles_link(self, fresh_db):
        """Two pending pps from DIFFERENT posts with very similar titles
        must end up in the same merged painpoint via Layer B."""
        post1 = _make_post("t3_a", score=200, num_comments=50)
        post2 = _make_post("t3_b", score=200, num_comments=50)

        pp1 = save_pending_painpoint(
            post1, "GitHub Actions timeout on monorepo builds", severity=8,
        )
        pp2 = save_pending_painpoint(
            post2, "GitHub Actions timeout on monorepo CI", severity=8,
        )

        merged1 = promote_pending(pp1)
        merged2 = promote_pending(pp2)

        assert merged1 == merged2, "near-duplicate titles should link via Layer B"
        conn = db.get_db()
        try:
            count = conn.execute(
                "SELECT signal_count FROM painpoints WHERE id = ?", (merged1,)
            ).fetchone()["signal_count"]
            assert count == 2
        finally:
            conn.close()

    def test_dissimilar_titles_create_separate_painpoints(self, fresh_db):
        post1 = _make_post("t3_a")
        post2 = _make_post("t3_b")

        pp1 = save_pending_painpoint(
            post1, "Llama 3 stalls on Apple Silicon high context", severity=8,
        )
        pp2 = save_pending_painpoint(
            post2, "Postgres locking issues during bulk insert operations", severity=8,
        )

        merged1 = promote_pending(pp1)
        merged2 = promote_pending(pp2)

        assert merged1 != merged2

    def test_layer_b_minhash_signature_reasonable(self, fresh_db):
        """Sanity check the MinHash on similar titles produces high jaccard."""
        a = make_minhash("GitHub Actions timeout on monorepo builds")
        b = make_minhash("GitHub Actions timeout on monorepo CI")
        assert a.jaccard(b) >= SIM_THRESHOLD


class TestLayerAMultiMatchMerge:
    """§3.5 multi-match branch — multi-source pending bridges two existing
    merged painpoints, merge_painpoints fires and the new pending links to
    the survivor."""

    def test_multi_source_pending_merges_spanned_painpoints(self, fresh_db):
        # Seed two merged painpoints with disjoint source sets.
        post_a = _make_post("t3_a")
        post_b = _make_post("t3_b")
        post_c = _make_post("t3_c")
        post_d = _make_post("t3_d")

        pp_a = save_pending_painpoint(post_a, "Pain about A", severity=8)
        pp_b = save_pending_painpoint(post_b, "Different wording xyzzy plugh", severity=8)
        # Different wording so Layer B doesn't link them at promote time.

        merged_99 = promote_pending(pp_a)
        merged_100 = promote_pending(pp_b)
        assert merged_99 != merged_100, "disjoint sources + dissimilar text → 2 painpoints"

        # Multi-source pending pp that bridges A and B (and adds C and D).
        bridge = save_pending_painpoint(post_c, "Bridge pain", severity=8)
        add_pending_source(bridge, post_a)   # already cited by merged_99
        add_pending_source(bridge, post_b)   # already cited by merged_100
        add_pending_source(bridge, post_d)   # new

        result = promote_pending(bridge)
        # Survivor must be one of the two; the other is gone.
        assert result in (merged_99, merged_100)
        survivor = result
        loser = merged_100 if survivor == merged_99 else merged_99

        conn = db.get_db()
        try:
            # Loser row is gone.
            assert conn.execute(
                "SELECT id FROM painpoints WHERE id = ?", (loser,)
            ).fetchone() is None

            # Survivor's signal_count is the sum (1 + 1 + 1 for the bridge link).
            sig = conn.execute(
                "SELECT signal_count FROM painpoints WHERE id = ?", (survivor,)
            ).fetchone()["signal_count"]
            assert sig >= 2

            # All sources from both pre-merge painpoints, plus the bridge,
            # are now linked under the survivor.
            sources = conn.execute(
                """
                SELECT DISTINCT pps.post_id
                FROM painpoint_sources ps
                JOIN pending_painpoint_all_sources pps
                  ON pps.pending_painpoint_id = ps.pending_painpoint_id
                WHERE ps.painpoint_id = ?
                """,
                (survivor,),
            ).fetchall()
            post_ids = {r["post_id"] for r in sources}
            assert post_a in post_ids
            assert post_b in post_ids
            assert post_c in post_ids
            assert post_d in post_ids
        finally:
            conn.close()


class TestMergePainpointsInvariants:
    """§3.5 step 6 — direct unit test of the merge_painpoints function."""

    def test_signal_count_summed(self, fresh_db):
        post_a = _make_post("t3_a")
        post_b = _make_post("t3_b")

        pp_a = save_pending_painpoint(post_a, "Pain A xyz", severity=5)
        pp_b = save_pending_painpoint(post_b, "Pain B abc qrs xyz", severity=5)
        survivor_id = promote_pending(pp_a)
        loser_id = promote_pending(pp_b)
        assert survivor_id != loser_id, "test setup needs two distinct painpoints"

        # Bump the loser's signal_count so we can verify the sum
        conn = db.get_db()
        try:
            conn.execute(
                "UPDATE painpoints SET signal_count = 5 WHERE id = ?", (loser_id,)
            )
            conn.execute(
                "UPDATE painpoints SET signal_count = 3 WHERE id = ?", (survivor_id,)
            )
            conn.commit()

            with merge_lock(conn):
                merge_painpoints(conn, survivor_id, loser_id)

            row = conn.execute(
                "SELECT signal_count FROM painpoints WHERE id = ?", (survivor_id,)
            ).fetchone()
            assert row["signal_count"] == 8

            assert conn.execute(
                "SELECT id FROM painpoints WHERE id = ?", (loser_id,)
            ).fetchone() is None
        finally:
            conn.close()

    def test_first_seen_min(self, fresh_db):
        post_a = _make_post("t3_a")
        post_b = _make_post("t3_b")

        pp_a = save_pending_painpoint(post_a, "Pain A pqr", severity=5)
        survivor_id = promote_pending(pp_a)

        time.sleep(0.01)   # ensure later timestamp on the loser
        pp_b = save_pending_painpoint(post_b, "Pain B mno xyz qrs", severity=5)
        loser_id = promote_pending(pp_b)
        assert survivor_id != loser_id

        conn = db.get_db()
        try:
            survivor_ts_before = conn.execute(
                "SELECT first_seen FROM painpoints WHERE id = ?", (survivor_id,)
            ).fetchone()["first_seen"]
            loser_ts = conn.execute(
                "SELECT first_seen FROM painpoints WHERE id = ?", (loser_id,)
            ).fetchone()["first_seen"]
            assert survivor_ts_before < loser_ts

            with merge_lock(conn):
                merge_painpoints(conn, survivor_id, loser_id)

            survivor_ts_after = conn.execute(
                "SELECT first_seen FROM painpoints WHERE id = ?", (survivor_id,)
            ).fetchone()["first_seen"]
            assert survivor_ts_after == survivor_ts_before  # min stays
        finally:
            conn.close()

    def test_survivor_category_unchanged(self, fresh_db):
        """§3.5 step 7 — survivor.category_id is the survivor's pre-merge
        value regardless of what category the loser was in. Deliberate-but-
        arbitrary choice."""
        post_a = _make_post("t3_a")
        post_b = _make_post("t3_b")
        pp_a = save_pending_painpoint(post_a, "Pain A foo", severity=5)
        pp_b = save_pending_painpoint(post_b, "Pain B bar baz", severity=5)
        survivor_id = promote_pending(pp_a)
        loser_id = promote_pending(pp_b)

        conn = db.get_db()
        try:
            # Move the loser to a different category so we can detect a change
            new_cat_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('OtherCat', NULL, 'd', datetime('now')) RETURNING id"
            ).fetchone()["id"]
            conn.execute(
                "UPDATE painpoints SET category_id = ? WHERE id = ?",
                (new_cat_id, loser_id),
            )
            survivor_cat_before = conn.execute(
                "SELECT category_id FROM painpoints WHERE id = ?", (survivor_id,)
            ).fetchone()["category_id"]
            assert survivor_cat_before != new_cat_id

            conn.commit()
            with merge_lock(conn):
                merge_painpoints(conn, survivor_id, loser_id)

            survivor_cat_after = conn.execute(
                "SELECT category_id FROM painpoints WHERE id = ?", (survivor_id,)
            ).fetchone()["category_id"]
            assert survivor_cat_after == survivor_cat_before
        finally:
            conn.close()

    def test_lsh_index_loses_loser_signature(self, fresh_db):
        post_a = _make_post("t3_a")
        post_b = _make_post("t3_b")
        pp_a = save_pending_painpoint(post_a, "Pain A foo", severity=5)
        pp_b = save_pending_painpoint(post_b, "Pain B bar baz", severity=5)

        # Promote both with a shared LSH index so we can inspect it.
        conn = db.get_db()
        try:
            lsh = PainpointLSH.load_or_build(conn)
        finally:
            conn.close()

        survivor_id = promote_pending(pp_a, lsh_index=lsh)
        loser_id = promote_pending(pp_b, lsh_index=lsh)
        assert survivor_id in lsh
        assert loser_id in lsh

        conn = db.get_db()
        try:
            with merge_lock(conn):
                merge_painpoints(conn, survivor_id, loser_id, lsh_index=lsh)
        finally:
            conn.close()

        assert survivor_id in lsh
        assert loser_id not in lsh


class TestNewPainpointInheritsPendingSources:
    """§3.5 — when a new merged pp is created in Uncategorized, its sources
    are exactly the triggering pending pp's sources, no more no less."""

    def test_multi_source_pending_creates_full_inheritance(self, fresh_db):
        post_a = _make_post("t3_a")
        post_b = _make_post("t3_b")
        post_c = _make_post("t3_c")

        pp_id = save_pending_painpoint(
            post_a, "Brand new pain qrs xyz nothing matches", severity=8,
        )
        add_pending_source(pp_id, post_b)
        add_pending_source(pp_id, post_c)

        merged = promote_pending(pp_id)
        assert merged is not None

        conn = db.get_db()
        try:
            sources = conn.execute(
                """
                SELECT DISTINCT pps.post_id
                FROM painpoint_sources ps
                JOIN pending_painpoint_all_sources pps
                  ON pps.pending_painpoint_id = ps.pending_painpoint_id
                WHERE ps.painpoint_id = ?
                ORDER BY pps.post_id
                """,
                (merged,),
            ).fetchall()
            post_ids = {r["post_id"] for r in sources}
            assert post_ids == {post_a, post_b, post_c}, (
                "new merged pp must inherit exactly the pending pp's source set"
            )
        finally:
            conn.close()


# ===========================================================================
# Step 3 — drop step (§2.4)
# ===========================================================================


class TestPromoteDropsLowRelevance:
    def test_low_relevance_pending_is_hard_deleted(self, fresh_db):
        # Score=0, num_comments=0, age=10 years → relevance ≈ 0
        old = _make_post(
            "t3_old", score=0, num_comments=0, age_seconds=10 * 365 * 86400
        )
        pp_id = save_pending_painpoint(old, "Cold dead pain", severity=1)

        # Confirm relevance is below the threshold
        rel = compute_pending_relevance(pp_id)
        assert rel < MIN_RELEVANCE_TO_PROMOTE

        result = promote_pending(pp_id)
        assert result is None, "below-threshold pending must drop, returning None"

        conn = db.get_db()
        try:
            # Pending row is hard-deleted (cascades to pending_painpoint_sources).
            assert conn.execute(
                "SELECT id FROM pending_painpoints WHERE id = ?", (pp_id,)
            ).fetchone() is None
            # No merged painpoint was created either.
            assert conn.execute(
                "SELECT COUNT(*) FROM painpoints"
            ).fetchone()[0] == 0
        finally:
            conn.close()


# ===========================================================================
# Step 4 — merge lock & concurrency (§4)
# ===========================================================================


class TestMergeLock:
    """The merge lock serialises writers via SQLite BEGIN IMMEDIATE.

    Two threads (rather than two processes) for in-process simplicity. Cross-
    process correctness comes for free from the same primitive.
    """

    def test_concurrent_promotes_dont_create_duplicates(self, fresh_db):
        # Pre-create a merged painpoint that both threads will try to link to.
        post_a = _make_post("t3_a", score=200, num_comments=50)
        seed_pp = save_pending_painpoint(post_a, "Shared pain abc", severity=8)
        merged = promote_pending(seed_pp)

        # Two pending pps from new posts but with very-similar titles, so both
        # threads' Layer B will find `merged`.
        post_b = _make_post("t3_b", score=200, num_comments=50)
        post_c = _make_post("t3_c", score=200, num_comments=50)
        pp_b = save_pending_painpoint(post_b, "Shared pain abc", severity=8)
        pp_c = save_pending_painpoint(post_c, "Shared pain abc", severity=8)

        results = {}

        def worker(pp_id):
            results[pp_id] = promote_pending(pp_id)

        t1 = threading.Thread(target=worker, args=(pp_b,))
        t2 = threading.Thread(target=worker, args=(pp_c,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        # Both should have linked into `merged`, no duplicates created.
        assert results[pp_b] == merged
        assert results[pp_c] == merged

        conn = db.get_db()
        try:
            count = conn.execute("SELECT COUNT(*) FROM painpoints").fetchone()[0]
            assert count == 1, f"expected 1 painpoint after concurrent link, got {count}"
            sig = conn.execute(
                "SELECT signal_count FROM painpoints WHERE id = ?", (merged,)
            ).fetchone()["signal_count"]
            assert sig == 3, f"expected signal_count=3 (3 pending pps linked), got {sig}"
        finally:
            conn.close()


# ===========================================================================
# Step 5 — promoter contract (§3, §5)
# ===========================================================================


class TestPromoterDoesNotTouchCategories:
    """§3.5 — the promoter is purely additive. It never creates, deletes, or
    renames categories. New painpoints with no Layer A/B match land in the
    Uncategorized sentinel."""

    def test_promote_does_not_create_new_categories(self, fresh_db):
        conn = db.get_db()
        try:
            cat_count_before = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        finally:
            conn.close()

        # Promote 3 painpoints with maximally distinct titles so Layer B
        # doesn't link them together — we want 3 separate merged painpoints
        # in Uncategorized.
        distinct_titles = [
            "Llama inference stalls Apple Silicon high context",
            "Postgres bulk insert deadlock under heavy load",
            "Webpack chunking memory leak large monorepo",
        ]
        for i, title in enumerate(distinct_titles):
            post_id = _make_post(f"t3_{i}", score=200, num_comments=50)
            pp_id = save_pending_painpoint(post_id, title, severity=8)
            promote_pending(pp_id)

        conn = db.get_db()
        try:
            cat_count_after = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
            assert cat_count_after == cat_count_before, (
                "promoter must not touch the categories table"
            )

            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            in_uncat = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]
            assert in_uncat == 3, "all new painpoints must land in Uncategorized"
        finally:
            conn.close()


# ===========================================================================
# Step 6 — category worker sweep behaviour (§5)
# ===========================================================================


def _seed_uncategorized_painpoints(titles, severity=8):
    """Bypass the promoter to seed Uncategorized with N painpoints whose
    titles would otherwise have linked at Layer B. Used by sweep tests
    that need a clusterable Uncategorized state.

    Each iteration uses its own connection — holding an outer connection
    while opening inner ones causes WAL deadlocks under SQLite's busy
    semantics, so we keep every operation isolated.
    """
    ids = []
    for i, title in enumerate(titles):
        post_id = _make_post(f"t3_seed_{i}", score=200, num_comments=50)
        pp_id = save_pending_painpoint(post_id, title, severity=severity)

        conn = db.get_db()
        try:
            uncat_id = get_uncategorized_id(conn=conn)
            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated) VALUES (?, ?, ?, 1, ?, ?, ?)",
                (title, "", severity, uncat_id, now, now),
            )
            new_pp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                "VALUES (?, ?)",
                (new_pp_id, pp_id),
            )
            conn.commit()
            ids.append(new_pp_id)
        finally:
            conn.close()
    return ids


class TestSweepProcessesUncategorized:
    """§5.1 step 1 — clusters Uncategorized into new categories via the
    LLM namer."""

    def test_cluster_promoted_to_new_category(self, fresh_db):
        # Seed enough painpoints with similar (above SWEEP_CLUSTER_THRESHOLD)
        # but not promote-time-equal text to form one cluster.
        titles = [
            "GitHub Actions timeout monorepo build slow",
            "GitHub Actions timeout monorepo build hangs",
            "GitHub Actions timeout monorepo build fails",
            "GitHub Actions timeout monorepo build retry",
            "GitHub Actions timeout monorepo build flake",
        ]
        _seed_uncategorized_painpoints(titles)

        namer = FakeNamer()
        summary = run_sweep(namer=namer)
        assert summary["uncategorized"]["proposed"] >= 1
        assert summary["uncategorized"]["accepted"] >= 1

        conn = db.get_db()
        try:
            # The seeded painpoints are no longer in Uncategorized
            uncat_id = get_uncategorized_id(conn=conn)
            in_uncat = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]
            assert in_uncat == 0

            # A new category was created with the FakeNamer's name
            new_cat = conn.execute(
                "SELECT id, name FROM categories WHERE name LIKE 'AutoCat-%'"
            ).fetchone()
            assert new_cat is not None
        finally:
            conn.close()

    def test_singleton_stays_in_uncategorized(self, fresh_db):
        _seed_uncategorized_painpoints([
            "Solo unrelated pain mno xyz",
        ])

        namer = FakeNamer()
        run_sweep(namer=namer)

        conn = db.get_db()
        try:
            uncat_id = get_uncategorized_id(conn=conn)
            in_uncat = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]
            assert in_uncat == 1
        finally:
            conn.close()


class TestSplitTest:
    """§5.1 step 2 + §5.2 add_category_split test."""

    def test_split_grown_bucket(self, fresh_db):
        """Manually populate a real category with two distinct sub-clusters
        each ≥ MIN_SUB_CLUSTER_SIZE, force a split-check, expect a split."""
        conn = db.get_db()
        try:
            # Find a non-Uncategorized seeded category to use as the parent.
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()
            assert parent_cat is not None
            cat_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Bloated', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat["id"],),
            ).fetchone()["id"]

            # Two sub-clusters, each of size 5
            cluster_a_titles = [
                "GitHub Actions timeout monorepo build slow",
                "GitHub Actions timeout monorepo build hangs",
                "GitHub Actions timeout monorepo build fails",
                "GitHub Actions timeout monorepo build retry",
                "GitHub Actions timeout monorepo build flake",
            ]
            cluster_b_titles = [
                "Postgres connection pool exhausted under load high",
                "Postgres connection pool exhausted under load big",
                "Postgres connection pool exhausted under load fast",
                "Postgres connection pool exhausted under load slow",
                "Postgres connection pool exhausted under load now",
            ]
            now = db._now()
            for t in cluster_a_titles + cluster_b_titles:
                conn.execute(
                    "INSERT INTO painpoints (title, description, severity, signal_count, "
                    "category_id, first_seen, last_updated) VALUES (?, '', 5, 1, ?, ?, ?)",
                    (t, cat_id, now, now),
                )
            # Force last_check to 0 so SPLIT_RECHECK_DELTA fires
            conn.execute(
                "UPDATE categories SET painpoint_count_at_last_check = 0 WHERE id = ?",
                (cat_id,),
            )
            conn.commit()
        finally:
            conn.close()

        namer = FakeNamer()
        summary = run_sweep(namer=namer)
        assert summary["split"]["accepted"] >= 1, f"expected split, got {summary}"

        conn = db.get_db()
        try:
            # The bloated category should now have ≤ 0 members (re-pointed)
            remaining = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat_id,)
            ).fetchone()[0]
            assert remaining == 0
        finally:
            conn.close()


class TestSplitTriggerDiscipline:
    """§5.1 step 2 — split-check must NOT re-fire on a stable bucket."""

    def test_no_recheck_when_below_delta(self, fresh_db):
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            cat_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at, "
                "painpoint_count_at_last_check) "
                "VALUES ('StableCat', ?, 'd', datetime('now'), 100) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]
            # Far fewer than 100 painpoints; the delta is negative
            now = db._now()
            for i in range(3):
                conn.execute(
                    "INSERT INTO painpoints (title, description, severity, signal_count, "
                    "category_id, first_seen, last_updated) VALUES (?, '', 5, 1, ?, ?, ?)",
                    (f"P{i}", cat_id, now, now),
                )
            conn.commit()

            events = list(propose_split_events(conn))
        finally:
            conn.close()

        for e in events:
            assert e.payload["source_category_id"] != cat_id, (
                "stable bucket should not get split-checked"
            )

    def test_recheck_resets_after_check(self, fresh_db):
        """After the split-check runs (whether or not split was accepted),
        painpoint_count_at_last_check is updated so we don't re-check until
        another SPLIT_RECHECK_DELTA painpoints arrive."""
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            cat_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at, "
                "painpoint_count_at_last_check) "
                "VALUES ('GrowsButNoSplit', ?, 'd', datetime('now'), 0) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]
            # 12 painpoints with totally distinct titles → no clusters of
            # MIN_SUB_CLUSTER_SIZE, so split is proposed but rejected at the
            # cluster-count test. The trigger-snapshot should still update.
            now = db._now()
            for i in range(12):
                conn.execute(
                    "INSERT INTO painpoints (title, description, severity, signal_count, "
                    "category_id, first_seen, last_updated) VALUES (?, '', 5, 1, ?, ?, ?)",
                    (f"DistinctPain{i}_{i*97}_qrs_{i}", cat_id, now, now),
                )
            conn.commit()

            list(propose_split_events(conn))   # exhaust the generator

            row = conn.execute(
                "SELECT painpoint_count_at_last_check FROM categories WHERE id = ?",
                (cat_id,),
            ).fetchone()
            assert row["painpoint_count_at_last_check"] == 12
        finally:
            conn.close()


class TestDeleteTest:
    """§5.1 step 3 + §5.2 delete_category test."""

    def test_dead_category_deleted(self, fresh_db):
        """A category whose only members are old, low-traction painpoints
        with relevance < MIN_RELEVANCE_TO_PROMOTE gets deleted."""
        # Create the source post + pending pp first, outside any held conn,
        # to avoid WAL deadlocks.
        old_post = _make_post(
            "t3_old", score=0, num_comments=0, age_seconds=10 * 365 * 86400,
        )
        pp_id = save_pending_painpoint(old_post, "Stale", severity=1)

        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            dead_cat = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('DeadCat', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]

            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES ('Stale member', '', 1, 1, ?, ?, ?, 0.0, ?)",
                (dead_cat, now, now, now),
            )
            new_pp = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                "VALUES (?, ?)",
                (new_pp, pp_id),
            )
            conn.commit()
        finally:
            conn.close()

        namer = FakeNamer()
        summary = run_sweep(namer=namer)
        assert summary["delete"]["accepted"] >= 1, f"expected delete, got {summary}"

        conn = db.get_db()
        try:
            assert conn.execute(
                "SELECT id FROM categories WHERE id = ?", (dead_cat,)
            ).fetchone() is None
        finally:
            conn.close()

    def test_live_member_blocks_delete(self, fresh_db):
        """A category whose total mass is below threshold but contains a
        single high-relevance painpoint must NOT be deleted (safety check)."""
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            cat_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('LowMassButLive', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]

            # One live painpoint (tiny relevance for the category mass to fall
            # below 1.0, but still > MIN_RELEVANCE_TO_PROMOTE for the safety
            # check). MIN_CATEGORY_RELEVANCE = 1.0, MIN_RELEVANCE_TO_PROMOTE = 0.5.
            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES ('Live but mass below threshold', '', 5, 1, ?, ?, ?, 0.7, ?)",
                (cat_id, now, now, now),
            )
            conn.commit()
        finally:
            conn.close()

        namer = FakeNamer()
        run_sweep(namer=namer)

        conn = db.get_db()
        try:
            # Category survives because its lone member has relevance > 0.5.
            assert conn.execute(
                "SELECT id FROM categories WHERE id = ?", (cat_id,)
            ).fetchone() is not None

            # An audit row should still exist marking the rejection.
            row = conn.execute(
                "SELECT accepted FROM category_events WHERE event_type = 'delete_category' "
                "AND target_category = ?",
                (cat_id,),
            ).fetchone()
            assert row is not None
            assert row["accepted"] == 0
        finally:
            conn.close()


class TestMergeTest:
    """§5.1 step 4 + §5.2 merge_categories test."""

    def test_similar_siblings_merged(self, fresh_db):
        """Two sibling categories whose members are textually near-identical
        should get merged."""
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            cat_a = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('SibA', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]
            cat_b = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('SibB', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]
            now = db._now()
            # Each sibling has a near-identical title member with relevance
            # ≥ MIN_CATEGORY_RELEVANCE so the delete pass doesn't kill them
            # before the merge pass runs.
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES ('GitHub Actions timeout monorepo builds slow', '', 5, 1, ?, ?, ?, 2.0, ?)",
                (cat_a, now, now, now),
            )
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES ('GitHub Actions timeout monorepo builds slow CI', '', 5, 1, ?, ?, ?, 2.0, ?)",
                (cat_b, now, now, now),
            )
            conn.commit()
        finally:
            conn.close()

        namer = FakeNamer()
        summary = run_sweep(namer=namer)
        assert summary["merge"]["accepted"] >= 1, f"expected merge, got {summary}"

        conn = db.get_db()
        try:
            # One of the two siblings is gone
            present = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM categories WHERE id IN (?, ?)", (cat_a, cat_b)
                ).fetchall()
            ]
            assert len(present) == 1
            survivor = present[0]
            # All previously-loser painpoints now point at the survivor.
            members = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (survivor,)
            ).fetchone()[0]
            assert members == 2
        finally:
            conn.close()

    def test_unrelated_siblings_not_merged(self, fresh_db):
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            cat_a = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Unrel-A', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]
            cat_b = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Unrel-B', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]
            now = db._now()
            # Cached relevance high enough to clear the delete-mass threshold.
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES ('Apple Silicon LLM inference issue', '', 5, 1, ?, ?, ?, 2.0, ?)",
                (cat_a, now, now, now),
            )
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES ('Postgres bulk insert lock contention nightmare', '', 5, 1, ?, ?, ?, 2.0, ?)",
                (cat_b, now, now, now),
            )
            conn.commit()
        finally:
            conn.close()

        namer = FakeNamer()
        run_sweep(namer=namer)

        conn = db.get_db()
        try:
            # Both still present
            count = conn.execute(
                "SELECT COUNT(*) FROM categories WHERE id IN (?, ?)", (cat_a, cat_b)
            ).fetchone()[0]
            assert count == 2
        finally:
            conn.close()


# ===========================================================================
# Step 7 — sweep idempotency, lock serialisation, audit log
# ===========================================================================


class TestSweepIsIdempotent:
    def test_back_to_back_sweep_makes_no_changes(self, fresh_db):
        """Two sweeps in a row: the second produces zero accepted events
        and no row in `painpoints` or `categories` changes."""
        # Set up some pipeline state
        titles = [
            "GitHub Actions timeout monorepo build slow",
            "GitHub Actions timeout monorepo build hangs",
            "GitHub Actions timeout monorepo build fails",
            "GitHub Actions timeout monorepo build retry",
            "GitHub Actions timeout monorepo build flake",
        ]
        _seed_uncategorized_painpoints(titles)

        namer = FakeNamer()
        first = run_sweep(namer=namer)

        conn = db.get_db()
        try:
            cat_snapshot = list(conn.execute(
                "SELECT id, name, parent_id FROM categories ORDER BY id"
            ).fetchall())
            pp_snapshot = list(conn.execute(
                "SELECT id, category_id FROM painpoints ORDER BY id"
            ).fetchall())
        finally:
            conn.close()

        second = run_sweep(namer=namer)

        # No accepted mutations on the second sweep
        for step in ("uncategorized", "split", "delete", "merge"):
            assert second[step]["accepted"] == 0, f"{step} should be idempotent"

        conn = db.get_db()
        try:
            cat_after = list(conn.execute(
                "SELECT id, name, parent_id FROM categories ORDER BY id"
            ).fetchall())
            pp_after = list(conn.execute(
                "SELECT id, category_id FROM painpoints ORDER BY id"
            ).fetchall())
            assert [tuple(r) for r in cat_after] == [tuple(r) for r in cat_snapshot]
            assert [tuple(r) for r in pp_after] == [tuple(r) for r in pp_snapshot]
        finally:
            conn.close()


class TestSweepLockSerialisesPromoter:
    """The promoter blocks while a sweep holds the lock."""

    def test_promoter_blocks_during_sweep(self, fresh_db):
        # Set up the pipeline
        post_id = _make_post("t3_a", score=200, num_comments=50)
        pp_id = save_pending_painpoint(post_id, "Some pain", severity=8)

        # A "sweep" that holds the lock for a measurable interval
        ready = threading.Event()
        release = threading.Event()

        def long_sweep():
            conn = db.get_db()
            try:
                with merge_lock(conn, timeout=30):
                    ready.set()
                    release.wait(5.0)   # hold the lock until released
            finally:
                conn.close()

        sweep_thread = threading.Thread(target=long_sweep)
        sweep_thread.start()
        assert ready.wait(2.0), "sweep thread should have acquired the lock"

        # Now try a promote in another thread; it should block.
        promote_done = threading.Event()
        promote_result = []

        def promote():
            promote_result.append(promote_pending(pp_id))
            promote_done.set()

        promoter_thread = threading.Thread(target=promote)
        promoter_thread.start()

        # Confirm the promoter is still blocked after a moment
        assert not promote_done.wait(0.3), (
            "promoter should still be blocked while sweep holds the lock"
        )

        # Release the sweep; the promoter should now make progress.
        release.set()
        sweep_thread.join(2.0)
        assert promote_done.wait(2.0), "promoter should complete once lock is free"
        promoter_thread.join(2.0)

        assert promote_result[0] is not None


class TestCategoryEventLog:
    """§5.3 — every sweep step that proposes an event writes a category_events
    row with metric_name, metric_value, threshold filled in, regardless of
    accept/reject."""

    def test_audit_columns_populated(self, fresh_db):
        # Trigger a delete via the live-member-blocks-delete fixture
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            cat_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('AuditCat', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]

            # One painpoint with high relevance — delete will be rejected
            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES ('Live one', '', 5, 1, ?, ?, ?, 0.7, ?)",
                (cat_id, now, now, now),
            )
            conn.commit()
        finally:
            conn.close()

        namer = FakeNamer()
        run_sweep(namer=namer)

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT * FROM category_events WHERE event_type = 'delete_category' "
                "AND target_category = ?",
                (cat_id,),
            ).fetchone()
            assert row is not None
            assert row["metric_name"] == "category_mass"
            assert row["threshold"] == MIN_CATEGORY_RELEVANCE
            assert row["accepted"] == 0
            payload = json.loads(row["payload_json"])
            assert payload["category_id"] == cat_id
        finally:
            conn.close()


# ===========================================================================
# End-to-end smoke
# ===========================================================================


class TestEndToEndSmoke:
    def test_promote_then_sweep(self, fresh_db):
        """Promote a pending painpoint end-to-end, then run a sweep, then
        verify the painpoint exists in some category and the sweep summary
        is well-formed."""
        post_id = _make_post("t3_a", score=200, num_comments=50)
        pp_id = save_pending_painpoint(
            post_id, "End to end smoke pain qrs", severity=7,
        )
        merged = promote_pending(pp_id)
        assert merged is not None

        namer = FakeNamer()
        summary = run_sweep(namer=namer)
        assert all(k in summary for k in ("uncategorized", "split", "delete", "merge"))
        for step in summary.values():
            assert "proposed" in step and "accepted" in step


# ===========================================================================
# Stress tests — concurrent promoters + sweeps must not deadlock
# ===========================================================================


class TestStressNoDeadlocks:
    """Hammer the merge_lock with many concurrent promoters and intermixed
    sweep runs. Every operation must complete within a hard deadline; no
    operation may hang indefinitely. Final database state must be
    consistent.

    These tests use threading rather than subprocesses for simplicity; the
    BEGIN IMMEDIATE primitive serialises writers regardless of which OS
    process they're in, so the failure mode (deadlock) would be the same.
    """

    HARD_TIMEOUT_SEC = 60

    def _seed_pending_pps(self, n):
        """Insert N posts and N pending painpoints with varied titles. Each
        pending pp gets enough traction to survive the relevance drop."""
        ids = []
        for i in range(n):
            post_id = _make_post(
                f"t3_stress_{i}", score=200, num_comments=50, age_seconds=3600,
            )
            # Mix of similar and dissimilar titles to exercise both Layer A
            # and Layer B paths.
            cluster_idx = i % 5
            base_titles = [
                "Llama inference stalls Apple Silicon",
                "Postgres deadlock under heavy concurrent load",
                "Webpack chunking crashes large monorepos",
                "Kubernetes pod eviction during rolling update",
                "FastAPI startup slow with many dependencies",
            ]
            title = f"{base_titles[cluster_idx]} variant {i}"
            pp_id = save_pending_painpoint(
                post_id, title,
                description=f"Description for stress pp {i}",
                severity=7,
            )
            ids.append(pp_id)
        return ids

    def test_concurrent_promoters_no_deadlock(self, fresh_db):
        """Four threads each draining a chunk of the pending queue
        concurrently. None may hang; final state must be consistent."""
        N_PENDINGS = 60
        N_WORKERS = 4
        pp_ids = self._seed_pending_pps(N_PENDINGS)

        # Partition the work
        chunks = [pp_ids[i::N_WORKERS] for i in range(N_WORKERS)]
        results = {i: [] for i in range(N_WORKERS)}
        errors = {}

        def worker(worker_id, my_chunk):
            try:
                for pp_id in my_chunk:
                    results[worker_id].append(promote_pending(pp_id))
            except Exception as e:
                errors[worker_id] = e

        threads = [
            threading.Thread(target=worker, args=(i, chunks[i]))
            for i in range(N_WORKERS)
        ]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.HARD_TIMEOUT_SEC)
            assert not t.is_alive(), (
                f"thread {t.name} hung past {self.HARD_TIMEOUT_SEC}s — likely deadlock"
            )
        elapsed = time.monotonic() - start

        # No exceptions in any worker
        assert errors == {}, f"workers raised: {errors}"

        # Every pending pp got a result (linked or dropped)
        all_results = sum((results[i] for i in range(N_WORKERS)), [])
        assert len(all_results) == N_PENDINGS

        # Database invariants
        conn = db.get_db()
        try:
            # Every pending pp is either gone (dropped) or linked into a painpoint
            unmerged = conn.execute(
                """
                SELECT pp.id FROM pending_painpoints pp
                LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id
                WHERE ps.painpoint_id IS NULL
                """
            ).fetchall()
            assert len(unmerged) == 0, f"orphan pending pps: {[r[0] for r in unmerged]}"

            # signal_count consistency: each painpoint's signal_count must equal
            # the number of pending_painpoint_id rows pointing at it
            mismatches = conn.execute(
                """
                SELECT p.id, p.signal_count, COUNT(ps.pending_painpoint_id) AS actual
                FROM painpoints p
                LEFT JOIN painpoint_sources ps ON ps.painpoint_id = p.id
                GROUP BY p.id
                HAVING p.signal_count != actual
                """
            ).fetchall()
            assert mismatches == [], (
                f"signal_count mismatches: {[dict(r) for r in mismatches]}"
            )

            # No painpoint_sources orphans (every row points at a real painpoint)
            orphan_sources = conn.execute(
                """
                SELECT ps.pending_painpoint_id FROM painpoint_sources ps
                LEFT JOIN painpoints p ON p.id = ps.painpoint_id
                WHERE p.id IS NULL
                """
            ).fetchall()
            assert orphan_sources == []
        finally:
            conn.close()

        print(
            f"\n  stress: {N_PENDINGS} pendings × {N_WORKERS} workers in {elapsed:.2f}s"
        )

    def test_sweep_concurrent_with_promoters_no_deadlock(self, fresh_db):
        """Three promoter threads draining a queue while a sweep thread runs
        a sweep mid-stream. No thread may hang; final state must be
        consistent."""
        N_PENDINGS = 40
        N_WORKERS = 3
        pp_ids = self._seed_pending_pps(N_PENDINGS)

        chunks = [pp_ids[i::N_WORKERS] for i in range(N_WORKERS)]
        errors = []

        def promoter_worker(my_chunk):
            try:
                for pp_id in my_chunk:
                    promote_pending(pp_id)
                    # Tiny stagger so the sweep has a chance to interleave
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("promoter", e))

        sweep_summaries = []

        def sweep_worker():
            try:
                # Wait briefly so promoters have inserted some painpoints first
                time.sleep(0.05)
                summary = run_sweep(namer=FakeNamer())
                sweep_summaries.append(summary)
            except Exception as e:
                errors.append(("sweep", e))

        promoter_threads = [
            threading.Thread(target=promoter_worker, args=(chunks[i],))
            for i in range(N_WORKERS)
        ]
        sweep_thread = threading.Thread(target=sweep_worker)

        for t in promoter_threads:
            t.start()
        sweep_thread.start()

        for t in promoter_threads:
            t.join(timeout=self.HARD_TIMEOUT_SEC)
            assert not t.is_alive(), f"promoter thread hung: likely deadlock"
        sweep_thread.join(timeout=self.HARD_TIMEOUT_SEC)
        assert not sweep_thread.is_alive(), "sweep thread hung: likely deadlock"

        assert errors == [], f"errors during stress: {errors}"
        assert len(sweep_summaries) == 1

        # Database invariants again
        conn = db.get_db()
        try:
            unmerged = conn.execute(
                """
                SELECT pp.id FROM pending_painpoints pp
                LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id
                WHERE ps.painpoint_id IS NULL
                """
            ).fetchall()
            assert len(unmerged) == 0

            mismatches = conn.execute(
                """
                SELECT p.id, p.signal_count, COUNT(ps.pending_painpoint_id) AS actual
                FROM painpoints p
                LEFT JOIN painpoint_sources ps ON ps.painpoint_id = p.id
                GROUP BY p.id
                HAVING p.signal_count != actual
                """
            ).fetchall()
            assert mismatches == []
        finally:
            conn.close()

    def test_repeated_sweeps_no_deadlock(self, fresh_db):
        """Run 10 sweeps in a row to make sure the lock release path is
        clean. Should be a no-op after the first one but must not hang."""
        # Seed something so the first sweep has work
        _seed_uncategorized_painpoints([
            "GitHub Actions timeout monorepo build slow",
            "GitHub Actions timeout monorepo build hangs",
            "GitHub Actions timeout monorepo build fails",
            "GitHub Actions timeout monorepo build retry",
            "GitHub Actions timeout monorepo build flake",
        ])

        namer = FakeNamer()
        start = time.monotonic()
        for i in range(10):
            run_sweep(namer=namer)
            # Each sweep must complete promptly
            assert time.monotonic() - start < self.HARD_TIMEOUT_SEC, (
                f"sweep {i} took too long; likely deadlock"
            )

    def test_lock_acquire_release_hammer(self, fresh_db):
        """Many threads rapidly acquiring/releasing the merge_lock with no
        actual work inside. Validates the BEGIN IMMEDIATE retry loop and
        the COMMIT/ROLLBACK cleanup paths under contention."""
        ITERATIONS = 30
        N_THREADS = 6
        errors = []

        def hammer():
            try:
                for _ in range(ITERATIONS):
                    conn = db.get_db()
                    try:
                        with merge_lock(conn, timeout=10):
                            conn.execute(
                                "SELECT COUNT(*) FROM painpoints"
                            ).fetchone()
                    finally:
                        conn.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.HARD_TIMEOUT_SEC)
            assert not t.is_alive(), "hammer thread hung — likely deadlock"
        assert errors == [], f"hammer errors: {errors}"


# ===========================================================================
# Realistic workflow simulation with synthetic data
# ===========================================================================


# Synthetic Reddit-like data: 8 distinct "topics" (which the LLM would
# extract under varied phrasings), each with ~6 paraphrased titles. We
# generate posts whose painpoint titles fall into these topics so that
# after promotion + sweep we expect to see categories forming around them.

_SYNTHETIC_TOPIC_TITLES = {
    "llm_inference_apple_silicon": [
        "Llama 3 70B inference stalls on M3 Max long context",
        "Llama 3 70B hangs on Apple Silicon at 40k context",
        "Llama 3 inference stalls on M3 Max big context lengths",
        "Llama 3 70B hangs on M3 Max during long inference runs",
        "Llama 3 inference stalls running on M3 Max chips",
        "Llama 3 70B hangs on Apple M3 Max with long context",
    ],
    "github_actions_timeout": [
        "GitHub Actions timeout on monorepo build pipelines",
        "GitHub Actions timeout running monorepo build jobs",
        "GitHub Actions timeout on monorepo build CI runs",
        "GitHub Actions timeout in monorepo build workflows",
        "GitHub Actions timeout monorepo build steps fail",
        "GitHub Actions timeout monorepo build runs hanging",
    ],
    "postgres_pool": [
        "Postgres connection pool exhausted during high traffic",
        "Postgres connection pool exhausted under heavy traffic",
        "Postgres connection pool exhausted high concurrent traffic",
        "Postgres connection pool exhausted during traffic spikes",
        "Postgres connection pool exhausted under traffic surge",
        "Postgres connection pool exhausted at peak traffic load",
    ],
    "k8s_eviction": [
        "Kubernetes pod eviction during cluster autoscaling events",
        "Kubernetes pod eviction during autoscaling cluster events",
        "Kubernetes pod eviction during node autoscaling cycles",
        "Kubernetes pod eviction during cluster scaling actions",
        "Kubernetes pod eviction during autoscaler node events",
        "Kubernetes pod eviction during cluster autoscale runs",
    ],
    "fastapi_startup": [
        "FastAPI startup slow large dependency injection graph",
        "FastAPI startup slow with large DI dependency graph",
        "FastAPI startup slow with large dependency graph trees",
        "FastAPI startup slow large dependency graph imports",
        "FastAPI startup slow with deep dependency graph wiring",
        "FastAPI startup slow with large injected dependency graph",
    ],
}


class TestRealisticWorkflow:
    """Simulate a realistic Reddit ingest run with synthetic data, then run
    the full pipeline (promoter → sweep) and check the final state matches
    what we'd expect from a real workflow.
    """

    def _generate_pending_pps(self):
        """Create posts + pending pps following the synthetic topic
        distribution. Returns dict topic→list of pp_ids."""
        out = {}
        post_counter = 0
        for topic, titles in _SYNTHETIC_TOPIC_TITLES.items():
            ids = []
            for title in titles:
                post_counter += 1
                post_id = _make_post(
                    f"t3_synth_{post_counter}",
                    title=f"Reddit post #{post_counter}",
                    score=200, num_comments=40,
                    age_seconds=3600 * (post_counter % 24),  # vary fresh ↔ stale
                )
                pp_id = save_pending_painpoint(
                    post_id, title,
                    description=f"User complaint about {topic}",
                    severity=6 + (post_counter % 4),
                )
                ids.append(pp_id)
            out[topic] = ids
        return out

    def test_full_workflow_with_synthetic_data(self, fresh_db):
        """End-to-end realistic run:

        1. Generate synthetic pending pps following 5 topics × ~6 paraphrases
           each (30 pendings total).
        2. Promote them all sequentially (just like the promoter daemon).
        3. Run a sweep with the FakeNamer.
        4. Verify the final state matches the design's expectations:
           - signal_count totals are consistent
           - painpoints have made it out of Uncategorized when clusterable
           - the sweep produced events
           - no orphan pending pps
        """
        topic_pp_ids = self._generate_pending_pps()
        all_pp_ids = sum(topic_pp_ids.values(), [])
        assert len(all_pp_ids) == sum(len(v) for v in _SYNTHETIC_TOPIC_TITLES.values())

        # Promote everything (sequential, like the daemon)
        promote_results = []
        for pp_id in all_pp_ids:
            promote_results.append(promote_pending(pp_id))

        # None should drop — synthetic data is fresh and meaty
        assert all(r is not None for r in promote_results), (
            "no synthetic pending pp should fall below MIN_RELEVANCE_TO_PROMOTE"
        )

        conn = db.get_db()
        try:
            painpoint_count_after_promote = conn.execute(
                "SELECT COUNT(*) FROM painpoints"
            ).fetchone()[0]
            n_topics = len(_SYNTHETIC_TOPIC_TITLES)
            n_pendings = len(all_pp_ids)
            # Layer B (SIM_THRESHOLD=0.65) catches the closest paraphrases
            # but not all of them — that's by design (high precision at
            # promote time). The remaining singletons sit in Uncategorized
            # waiting for the sweep clusterer (SWEEP_CLUSTER_THRESHOLD=0.40)
            # to group them. So expect SOMEWHERE between n_topics and
            # n_pendings merged painpoints after promotion.
            assert n_topics <= painpoint_count_after_promote <= n_pendings, (
                f"painpoint count {painpoint_count_after_promote} outside "
                f"plausible range [{n_topics}, {n_pendings}]"
            )

            # signal_count consistency
            mismatches = conn.execute(
                """
                SELECT p.id, p.signal_count, COUNT(ps.pending_painpoint_id) AS actual
                FROM painpoints p
                LEFT JOIN painpoint_sources ps ON ps.painpoint_id = p.id
                GROUP BY p.id
                HAVING p.signal_count != actual
                """
            ).fetchall()
            assert mismatches == []

            # Total signal_count over all painpoints == total pending pps linked
            total_links = conn.execute(
                "SELECT SUM(signal_count) FROM painpoints"
            ).fetchone()[0]
            assert total_links == n_pendings, (
                f"expected {n_pendings} links, got {total_links}"
            )

            # How many painpoints landed in the Uncategorized sentinel?
            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            in_uncat_before_sweep = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]
        finally:
            conn.close()

        # Run the sweep — Uncategorized singletons that share topic vocab
        # at the lower SWEEP_CLUSTER_THRESHOLD should now form clusters.
        namer = FakeNamer()
        summary = run_sweep(namer=namer)
        for step in summary.values():
            assert "proposed" in step and "accepted" in step

        # After the sweep, Uncategorized should have shrunk if there was
        # anything clusterable. We don't assert it went to zero (some
        # singletons may not share vocab with anyone), but it must not
        # have GROWN.
        conn = db.get_db()
        try:
            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            in_uncat_after_sweep = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]
            assert in_uncat_after_sweep <= in_uncat_before_sweep, (
                f"Uncategorized grew during sweep: {in_uncat_before_sweep} → "
                f"{in_uncat_after_sweep}"
            )
        finally:
            conn.close()

        # Run the sweep again — should be idempotent (no new accepted events)
        summary2 = run_sweep(namer=namer)
        for step_name, step in summary2.items():
            assert step["accepted"] == 0, (
                f"second sweep should be idempotent, but {step_name} accepted "
                f"{step['accepted']}"
            )

        # No orphan pending pps after the full workflow
        conn = db.get_db()
        try:
            orphans = conn.execute(
                """
                SELECT pp.id FROM pending_painpoints pp
                LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id
                WHERE ps.painpoint_id IS NULL
                """
            ).fetchall()
            assert orphans == []
        finally:
            conn.close()

    def test_workflow_with_multi_source_extraction(self, fresh_db):
        """Realistic batched LLM extraction case: one pending pp drawn from
        multiple posts (the LLM clustered them at extraction time)."""
        # Three posts whose content the LLM has determined are about the
        # same pain. Single pending pp with multi-source.
        post_ids = []
        for i in range(3):
            pid = _make_post(
                f"t3_multi_{i}", score=300, num_comments=80,
                age_seconds=3600,
            )
            post_ids.append(pid)

        pp_id = save_pending_painpoint(
            post_ids[0],
            "Cross-post pain extracted by LLM batched mode",
            description="LLM clustered three posts into one painpoint",
            severity=8,
        )
        add_pending_source(pp_id, post_ids[1])
        add_pending_source(pp_id, post_ids[2])

        merged = promote_pending(pp_id)
        assert merged is not None

        conn = db.get_db()
        try:
            sources = conn.execute(
                """
                SELECT DISTINCT pps.post_id FROM painpoint_sources ps
                JOIN pending_painpoint_all_sources pps
                  ON pps.pending_painpoint_id = ps.pending_painpoint_id
                WHERE ps.painpoint_id = ?
                """,
                (merged,),
            ).fetchall()
            sourced_posts = {r["post_id"] for r in sources}
            assert sourced_posts == set(post_ids), (
                "multi-source pending pp's full source set must be inherited"
            )
        finally:
            conn.close()

        # Now add another multi-source pending that bridges the existing
        # painpoint with a new pain — should trigger the Layer A multi-match
        # merge if there's another existing painpoint sharing one of the
        # sources. Set that up:
        new_post = _make_post("t3_multi_new", score=300, num_comments=80)
        unrelated_pp = save_pending_painpoint(
            new_post,
            "Totally separate Postgres deadlock pain xyz qrs",
            severity=8,
        )
        unrelated_merged = promote_pending(unrelated_pp)
        assert unrelated_merged != merged

        # Bridge pp: spans new_post (in unrelated_merged) and post_ids[0] (in merged)
        bridge_post = _make_post("t3_bridge", score=300, num_comments=80)
        bridge_pp = save_pending_painpoint(
            bridge_post, "Bridge pain pqr xyz", severity=8,
        )
        add_pending_source(bridge_pp, new_post)
        add_pending_source(bridge_pp, post_ids[0])

        bridge_result = promote_pending(bridge_pp)
        assert bridge_result in (merged, unrelated_merged)

        # Whichever survived, the loser is gone
        conn = db.get_db()
        try:
            survivor_count = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE id IN (?, ?)",
                (merged, unrelated_merged),
            ).fetchone()[0]
            assert survivor_count == 1, (
                "Layer A multi-match should have merged the two painpoints"
            )
        finally:
            conn.close()

    def test_workflow_relevance_decay_drops_stale(self, fresh_db):
        """Realistic mixed batch: some pps are fresh and survive promotion,
        some are old and get dropped."""
        fresh_post = _make_post(
            "t3_fresh", score=200, num_comments=50, age_seconds=3600,
        )
        old_post = _make_post(
            "t3_old", score=0, num_comments=0, age_seconds=10 * 365 * 86400,
        )

        fresh_pp = save_pending_painpoint(
            fresh_post, "Fresh interesting pain qrs", severity=8,
        )
        stale_pp = save_pending_painpoint(
            old_post, "Stale low-traction pain pqr", severity=1,
        )

        fresh_result = promote_pending(fresh_pp)
        stale_result = promote_pending(stale_pp)

        assert fresh_result is not None, "fresh pp must promote"
        assert stale_result is None, "stale low-traction pp must drop"

        conn = db.get_db()
        try:
            # Stale pp row was hard-deleted
            assert conn.execute(
                "SELECT id FROM pending_painpoints WHERE id = ?", (stale_pp,)
            ).fetchone() is None
            # Fresh pp is linked
            assert conn.execute(
                "SELECT id FROM pending_painpoints WHERE id = ?", (fresh_pp,)
            ).fetchone() is not None
            # Exactly one merged painpoint exists
            assert conn.execute(
                "SELECT COUNT(*) FROM painpoints"
            ).fetchone()[0] == 1
        finally:
            conn.close()
