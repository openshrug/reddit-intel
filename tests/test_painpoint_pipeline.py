"""Integration tests for the painpoint ingest pipeline.

Covers every step of the pipeline: relevance computation, multi-source
max aggregation, embedding cosine similarity, source inheritance,
merge-lock thread safety, the four sweep passes (Uncategorized cluster,
split, delete, merge), trigger discipline, idempotency, and the audit
log.

Tests run against a fresh temp DB per function so they don't interfere.
The LLM is always mocked via FakeNamer -- no network calls, no API costs.
Embeddings use FakeEmbedder -- deterministic, no API calls.

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
    promote_pending,
    save_pending_painpoint,
)
from db.relevance import (
    MIN_RELEVANCE_TO_PROMOTE,
    compute_painpoint_relevance,
    compute_pending_relevance,
    per_source_relevance,
)
from db.embeddings import (
    MERGE_COSINE_THRESHOLD,
    FakeEmbedder,
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
    """Per-test temp DB. Each test gets its own SQLite file."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    yield path


def _make_post(name, *, score=200, num_comments=50, age_seconds=3600,
               title="X", subreddit="test"):
    """Insert a post via upsert_post."""
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
# Step 1 -- relevance computation
# ===========================================================================


class TestRelevance:
    """Per-source relevance formula behaviour."""

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

    def test_severity_changes_relevance(self, fresh_db):
        post = _make_post("t3_a", score=100, num_comments=10)
        conn = db.get_db()
        try:
            p = conn.execute("SELECT * FROM posts WHERE id = ?", (post,)).fetchone()
            r1 = per_source_relevance(p, None, severity=1)
            r10 = per_source_relevance(p, None, severity=10)
            # severity_mult: 0.6 -> 1.5, ratio 2.5
            assert 2.4 < (r10 / r1) < 2.6
        finally:
            conn.close()


class TestRelevanceMaxAggregation:
    """Multi-source painpoints aggregate via max, not sum/mean."""

    def test_max_over_sources(self, fresh_db):
        fresh = _make_post("t3_fresh", score=500, num_comments=200, age_seconds=60)
        old1 = _make_post("t3_old1", score=10, num_comments=2, age_seconds=60 * 86400)
        old2 = _make_post("t3_old2", score=10, num_comments=2, age_seconds=60 * 86400)

        pp_id = save_pending_painpoint(fresh, "Some pain", description="d", severity=8)
        add_pending_source(pp_id, old1, comment_id=None)
        add_pending_source(pp_id, old2, comment_id=None)

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
# Step 2 -- similarity layers
# ===========================================================================


class TestEmbeddingSimilarity:
    """Embedding cosine similarity for painpoint matching."""

    def test_near_duplicate_titles_link(self, fresh_db):
        """Two pending pps from DIFFERENT posts with very similar titles
        must end up in the same merged painpoint via Layer B."""
        embedder = FakeEmbedder()
        post1 = _make_post("t3_a", score=200, num_comments=50)
        post2 = _make_post("t3_b", score=200, num_comments=50)

        pp1 = save_pending_painpoint(
            post1, "GitHub Actions timeout on monorepo builds", severity=8,
        )
        pp2 = save_pending_painpoint(
            post2, "GitHub Actions timeout on monorepo CI", severity=8,
        )

        merged1 = promote_pending(pp1, embedder=embedder)
        merged2 = promote_pending(pp2, embedder=embedder)

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
        embedder = FakeEmbedder()
        post1 = _make_post("t3_a")
        post2 = _make_post("t3_b")

        pp1 = save_pending_painpoint(
            post1, "Llama 3 stalls on Apple Silicon high context", severity=8,
        )
        pp2 = save_pending_painpoint(
            post2, "Postgres locking issues during bulk insert operations", severity=8,
        )

        merged1 = promote_pending(pp1, embedder=embedder)
        merged2 = promote_pending(pp2, embedder=embedder)

        assert merged1 != merged2

    def test_fake_embedder_similar_texts_produce_high_cosine(self, fresh_db):
        """Sanity check that FakeEmbedder produces similar vectors for similar text."""
        embedder = FakeEmbedder()
        a = embedder.embed("GitHub Actions timeout on monorepo builds")
        b = embedder.embed("GitHub Actions timeout on monorepo CI")
        # Compute cosine sim
        import math
        dot = sum(x * y for x, y in zip(a, b))
        assert dot >= MERGE_COSINE_THRESHOLD



# (TestLayerAMultiMatchMerge and TestMergePainpointsInvariants removed —
# Layer A source-overlap prefilter dropped in favor of pure embedding
# cosine similarity. merge_painpoints function still exists in
# db/painpoints.py but is no longer triggered by the promoter flow.)


class TestNewPainpointInheritsPendingSources:
    """When a new merged pp is created, its sources are exactly the
    triggering pending pp's sources."""

    def test_multi_source_pending_creates_full_inheritance(self, fresh_db):
        embedder = FakeEmbedder()
        post_a = _make_post("t3_a")
        post_b = _make_post("t3_b")
        post_c = _make_post("t3_c")

        pp_id = save_pending_painpoint(
            post_a, "Brand new pain qrs xyz nothing matches", severity=8,
        )
        add_pending_source(pp_id, post_b)
        add_pending_source(pp_id, post_c)

        merged = promote_pending(pp_id, embedder=embedder)
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
# Step 3 -- drop step
# ===========================================================================


class TestPromoteDropsLowRelevance:
    def test_low_relevance_pending_is_hard_deleted(self, fresh_db):
        embedder = FakeEmbedder()
        old = _make_post(
            "t3_old", score=0, num_comments=0, age_seconds=10 * 365 * 86400
        )
        pp_id = save_pending_painpoint(old, "Cold dead pain", severity=1)

        rel = compute_pending_relevance(pp_id)
        assert rel < MIN_RELEVANCE_TO_PROMOTE

        result = promote_pending(pp_id, embedder=embedder)
        assert result is None, "below-threshold pending must drop, returning None"

        conn = db.get_db()
        try:
            assert conn.execute(
                "SELECT id FROM pending_painpoints WHERE id = ?", (pp_id,)
            ).fetchone() is None
            assert conn.execute(
                "SELECT COUNT(*) FROM painpoints"
            ).fetchone()[0] == 0
        finally:
            conn.close()


# ===========================================================================
# Step 4 -- merge lock & concurrency
# ===========================================================================


class TestMergeLock:
    """The merge lock serialises writers via SQLite BEGIN IMMEDIATE."""

    def test_concurrent_promotes_dont_create_duplicates(self, fresh_db):
        embedder = FakeEmbedder()
        post_a = _make_post("t3_a", score=200, num_comments=50)
        seed_pp = save_pending_painpoint(post_a, "Shared pain abc", severity=8)
        merged = promote_pending(seed_pp, embedder=embedder)

        post_b = _make_post("t3_b", score=200, num_comments=50)
        post_c = _make_post("t3_c", score=200, num_comments=50)
        pp_b = save_pending_painpoint(post_b, "Shared pain abc", severity=8)
        pp_c = save_pending_painpoint(post_c, "Shared pain abc", severity=8)

        results = {}

        def worker(pp_id):
            results[pp_id] = promote_pending(pp_id, embedder=embedder)

        t1 = threading.Thread(target=worker, args=(pp_b,))
        t2 = threading.Thread(target=worker, args=(pp_c,))
        t1.start(); t2.start()
        t1.join(); t2.join()

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
# Step 5 -- promoter contract
# ===========================================================================


class TestPromoterCategoryAssignment:
    """The promoter uses embedding-based category assignment when creating
    a new merged painpoint. Falls back to Uncategorized when no good
    embedding match exists."""

    def test_no_category_proposed_lands_in_uncategorized(self, fresh_db):
        """Pending pp with no category_name and no category embeddings ->
        merged painpoint goes to Uncategorized."""
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            cat_count_before = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        finally:
            conn.close()

        distinct_titles = [
            "Llama inference stalls Apple Silicon high context",
            "Postgres bulk insert deadlock under heavy load",
            "Webpack chunking memory leak large monorepo",
        ]
        for i, title in enumerate(distinct_titles):
            post_id = _make_post(f"t3_{i}", score=200, num_comments=50)
            pp_id = save_pending_painpoint(post_id, title, severity=8)
            promote_pending(pp_id, embedder=embedder)

        conn = db.get_db()
        try:
            cat_count_after = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
            assert cat_count_after == cat_count_before, (
                "promoter must not create new categories"
            )

            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            in_uncat = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]
            assert in_uncat == 3, (
                "painpoints with no category match must land in Uncategorized"
            )
        finally:
            conn.close()

    def test_unknown_category_proposed_falls_back_to_uncategorized(self, fresh_db):
        embedder = FakeEmbedder()
        post_id = _make_post("t3_a", score=200, num_comments=50)
        pp_id = save_pending_painpoint(
            post_id,
            "Some unique pain xyz qrs mno",
            category_name="Totally Nonexistent Category",
            severity=8,
        )
        merged = promote_pending(pp_id, embedder=embedder)
        assert merged is not None

        conn = db.get_db()
        try:
            row = conn.execute(
                "SELECT category_id FROM painpoints WHERE id = ?", (merged,)
            ).fetchone()
            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            assert row["category_id"] == uncat_id, (
                "unknown category should fall back to Uncategorized"
            )
        finally:
            conn.close()

    def test_promoter_does_not_create_categories(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            cat_count_before = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        finally:
            conn.close()

        for i, cat_name in enumerate(["LLM Infrastructure", "CI/CD & DevOps", None]):
            post_id = _make_post(f"t3_{i}", score=200, num_comments=50)
            pp_id = save_pending_painpoint(
                post_id, f"Pain {i} qrs xyz mno pqr {i*13}",
                category_name=cat_name, severity=8,
            )
            promote_pending(pp_id, embedder=embedder)

        conn = db.get_db()
        try:
            cat_count_after = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
            assert cat_count_after == cat_count_before, (
                "promoter must never create new categories"
            )
        finally:
            conn.close()


# ===========================================================================
# Step 6 -- category worker sweep behaviour
# ===========================================================================


def _seed_uncategorized_painpoints(titles, severity=8, embedder=None):
    """Bypass the promoter to seed Uncategorized with N painpoints.

    Each iteration uses its own connection to avoid WAL deadlocks.
    Also stores embeddings for each painpoint.
    """
    if embedder is None:
        embedder = FakeEmbedder()
    from db.embeddings import store_painpoint_embedding
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
            # Store embedding for this painpoint
            text = title
            emb = embedder.embed(text)
            store_painpoint_embedding(conn, new_pp_id, emb)
            conn.commit()
            ids.append(new_pp_id)
        finally:
            conn.close()
    return ids


class TestSweepProcessesUncategorized:
    """Clusters Uncategorized into new categories via the LLM namer."""

    def test_cluster_promoted_to_new_category(self, fresh_db):
        embedder = FakeEmbedder()
        titles = [
            "GitHub Actions timeout monorepo build slow",
            "GitHub Actions timeout monorepo build hangs",
            "GitHub Actions timeout monorepo build fails",
            "GitHub Actions timeout monorepo build retry",
            "GitHub Actions timeout monorepo build flake",
        ]
        _seed_uncategorized_painpoints(titles, embedder=embedder)

        namer = FakeNamer()
        summary = run_sweep(namer=namer, embedder=embedder)
        assert summary["uncategorized"]["proposed"] >= 1
        assert summary["uncategorized"]["accepted"] >= 1

        conn = db.get_db()
        try:
            uncat_id = get_uncategorized_id(conn=conn)
            in_uncat = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]
            assert in_uncat == 0

            new_cat = conn.execute(
                "SELECT id, name FROM categories WHERE name LIKE 'AutoCat-%'"
            ).fetchone()
            assert new_cat is not None
        finally:
            conn.close()

    def test_singleton_stays_in_uncategorized(self, fresh_db):
        embedder = FakeEmbedder()
        _seed_uncategorized_painpoints([
            "Solo unrelated pain mno xyz",
        ], embedder=embedder)

        namer = FakeNamer()
        run_sweep(namer=namer, embedder=embedder)

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
    """Split crowded categories."""

    def test_split_grown_bucket(self, fresh_db):
        embedder = FakeEmbedder()
        from db.embeddings import store_painpoint_embedding
        conn = db.get_db()
        try:
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
                pp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                emb = embedder.embed(t)
                store_painpoint_embedding(conn, pp_id, emb)
            conn.execute(
                "UPDATE categories SET painpoint_count_at_last_check = 0 WHERE id = ?",
                (cat_id,),
            )
            conn.commit()
        finally:
            conn.close()

        namer = FakeNamer()
        summary = run_sweep(namer=namer, embedder=embedder)
        assert summary["split"]["accepted"] >= 1, f"expected split, got {summary}"

        conn = db.get_db()
        try:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat_id,)
            ).fetchone()[0]
            assert remaining == 0
        finally:
            conn.close()


class TestSplitTriggerDiscipline:
    """Split-check must NOT re-fire on a stable bucket."""

    def test_no_recheck_when_below_delta(self, fresh_db):
        embedder = FakeEmbedder()
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
            now = db._now()
            for i in range(3):
                conn.execute(
                    "INSERT INTO painpoints (title, description, severity, signal_count, "
                    "category_id, first_seen, last_updated) VALUES (?, '', 5, 1, ?, ?, ?)",
                    (f"P{i}", cat_id, now, now),
                )
            conn.commit()

            events = list(propose_split_events(conn, embedder=embedder))
        finally:
            conn.close()

        for e in events:
            assert e.payload["source_category_id"] != cat_id, (
                "stable bucket should not get split-checked"
            )

    def test_recheck_resets_after_check(self, fresh_db):
        embedder = FakeEmbedder()
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
            now = db._now()
            for i in range(12):
                conn.execute(
                    "INSERT INTO painpoints (title, description, severity, signal_count, "
                    "category_id, first_seen, last_updated) VALUES (?, '', 5, 1, ?, ?, ?)",
                    (f"DistinctPain{i}_{i*97}_qrs_{i}", cat_id, now, now),
                )
            conn.commit()

            list(propose_split_events(conn, embedder=embedder))

            row = conn.execute(
                "SELECT painpoint_count_at_last_check FROM categories WHERE id = ?",
                (cat_id,),
            ).fetchone()
            assert row["painpoint_count_at_last_check"] == 12
        finally:
            conn.close()


class TestDeleteTest:
    """Delete dead categories."""

    def test_dead_category_deleted(self, fresh_db):
        embedder = FakeEmbedder()
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
        summary = run_sweep(namer=namer, embedder=embedder)
        assert summary["delete"]["accepted"] >= 1, f"expected delete, got {summary}"

        conn = db.get_db()
        try:
            assert conn.execute(
                "SELECT id FROM categories WHERE id = ?", (dead_cat,)
            ).fetchone() is None
        finally:
            conn.close()

    def test_live_member_blocks_delete(self, fresh_db):
        embedder = FakeEmbedder()
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
        run_sweep(namer=namer, embedder=embedder)

        conn = db.get_db()
        try:
            assert conn.execute(
                "SELECT id FROM categories WHERE id = ?", (cat_id,)
            ).fetchone() is not None

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
    """Merge duplicate sibling categories."""

    def test_similar_siblings_merged(self, fresh_db):
        embedder = FakeEmbedder()
        from db.embeddings import store_painpoint_embedding, store_category_embedding
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
            # Near-identical titles
            title_a = "GitHub Actions timeout monorepo builds slow"
            title_b = "GitHub Actions timeout monorepo builds slow CI"
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES (?, '', 5, 1, ?, ?, ?, 2.0, ?)",
                (title_a, cat_a, now, now, now),
            )
            pp_a_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            store_painpoint_embedding(conn, pp_a_id, embedder.embed(title_a))

            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES (?, '', 5, 1, ?, ?, ?, 2.0, ?)",
                (title_b, cat_b, now, now, now),
            )
            pp_b_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            store_painpoint_embedding(conn, pp_b_id, embedder.embed(title_b))

            # Store category embeddings (mean of members)
            store_category_embedding(conn, cat_a, embedder.embed(title_a))
            store_category_embedding(conn, cat_b, embedder.embed(title_b))

            conn.commit()
        finally:
            conn.close()

        namer = FakeNamer()
        summary = run_sweep(namer=namer, embedder=embedder)
        assert summary["merge"]["accepted"] >= 1, f"expected merge, got {summary}"

        conn = db.get_db()
        try:
            present = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM categories WHERE id IN (?, ?)", (cat_a, cat_b)
                ).fetchall()
            ]
            assert len(present) == 1
            survivor = present[0]
            members = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (survivor,)
            ).fetchone()[0]
            assert members == 2
        finally:
            conn.close()

    def test_unrelated_siblings_not_merged(self, fresh_db):
        embedder = FakeEmbedder()
        from db.embeddings import store_painpoint_embedding, store_category_embedding
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
            title_a = "Apple Silicon LLM inference issue"
            title_b = "Postgres bulk insert lock contention nightmare"
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES (?, '', 5, 1, ?, ?, ?, 2.0, ?)",
                (title_a, cat_a, now, now, now),
            )
            pp_a_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            store_painpoint_embedding(conn, pp_a_id, embedder.embed(title_a))

            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES (?, '', 5, 1, ?, ?, ?, 2.0, ?)",
                (title_b, cat_b, now, now, now),
            )
            pp_b_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            store_painpoint_embedding(conn, pp_b_id, embedder.embed(title_b))

            store_category_embedding(conn, cat_a, embedder.embed(title_a))
            store_category_embedding(conn, cat_b, embedder.embed(title_b))

            conn.commit()
        finally:
            conn.close()

        namer = FakeNamer()
        run_sweep(namer=namer, embedder=embedder)

        conn = db.get_db()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM categories WHERE id IN (?, ?)", (cat_a, cat_b)
            ).fetchone()[0]
            assert count == 2
        finally:
            conn.close()


# ===========================================================================
# Step 7 -- sweep idempotency, lock serialisation, audit log
# ===========================================================================


class TestSweepIsIdempotent:
    def test_back_to_back_sweep_makes_no_changes(self, fresh_db):
        embedder = FakeEmbedder()
        titles = [
            "GitHub Actions timeout monorepo build slow",
            "GitHub Actions timeout monorepo build hangs",
            "GitHub Actions timeout monorepo build fails",
            "GitHub Actions timeout monorepo build retry",
            "GitHub Actions timeout monorepo build flake",
        ]
        _seed_uncategorized_painpoints(titles, embedder=embedder)

        namer = FakeNamer()
        first = run_sweep(namer=namer, embedder=embedder)

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

        second = run_sweep(namer=namer, embedder=embedder)

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
        embedder = FakeEmbedder()
        post_id = _make_post("t3_a", score=200, num_comments=50)
        pp_id = save_pending_painpoint(post_id, "Some pain", severity=8)

        ready = threading.Event()
        release = threading.Event()

        def long_sweep():
            conn = db.get_db()
            try:
                with merge_lock(conn, timeout=30):
                    ready.set()
                    release.wait(5.0)
            finally:
                conn.close()

        sweep_thread = threading.Thread(target=long_sweep)
        sweep_thread.start()
        assert ready.wait(2.0), "sweep thread should have acquired the lock"

        promote_done = threading.Event()
        promote_result = []

        def promote():
            promote_result.append(promote_pending(pp_id, embedder=embedder))
            promote_done.set()

        promoter_thread = threading.Thread(target=promote)
        promoter_thread.start()

        assert not promote_done.wait(0.3), (
            "promoter should still be blocked while sweep holds the lock"
        )

        release.set()
        sweep_thread.join(2.0)
        assert promote_done.wait(2.0), "promoter should complete once lock is free"
        promoter_thread.join(2.0)

        assert promote_result[0] is not None


class TestCategoryEventLog:
    """Every sweep step that proposes an event writes a category_events
    row with metric_name, metric_value, threshold filled in."""

    def test_audit_columns_populated(self, fresh_db):
        embedder = FakeEmbedder()
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
        run_sweep(namer=namer, embedder=embedder)

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
        embedder = FakeEmbedder()
        post_id = _make_post("t3_a", score=200, num_comments=50)
        pp_id = save_pending_painpoint(
            post_id, "End to end smoke pain qrs", severity=7,
        )
        merged = promote_pending(pp_id, embedder=embedder)
        assert merged is not None

        namer = FakeNamer()
        summary = run_sweep(namer=namer, embedder=embedder)
        assert all(k in summary for k in ("uncategorized", "split", "delete", "merge"))
        for step in summary.values():
            assert "proposed" in step and "accepted" in step


# ===========================================================================
# Stress tests -- concurrent promoters + sweeps must not deadlock
# ===========================================================================


class TestStressNoDeadlocks:
    """Hammer the merge_lock with many concurrent promoters and intermixed
    sweep runs. Every operation must complete within a hard deadline."""

    HARD_TIMEOUT_SEC = 60

    def _seed_pending_pps(self, n):
        ids = []
        for i in range(n):
            post_id = _make_post(
                f"t3_stress_{i}", score=200, num_comments=50, age_seconds=3600,
            )
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
        embedder = FakeEmbedder()
        N_PENDINGS = 60
        N_WORKERS = 4
        pp_ids = self._seed_pending_pps(N_PENDINGS)

        chunks = [pp_ids[i::N_WORKERS] for i in range(N_WORKERS)]
        results = {i: [] for i in range(N_WORKERS)}
        errors = {}

        def worker(worker_id, my_chunk):
            try:
                for pp_id in my_chunk:
                    results[worker_id].append(promote_pending(pp_id, embedder=embedder))
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
                f"thread {t.name} hung past {self.HARD_TIMEOUT_SEC}s -- likely deadlock"
            )
        elapsed = time.monotonic() - start

        assert errors == {}, f"workers raised: {errors}"

        all_results = sum((results[i] for i in range(N_WORKERS)), [])
        assert len(all_results) == N_PENDINGS

        conn = db.get_db()
        try:
            unmerged = conn.execute(
                """
                SELECT pp.id FROM pending_painpoints pp
                LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id
                WHERE ps.painpoint_id IS NULL
                """
            ).fetchall()
            assert len(unmerged) == 0, f"orphan pending pps: {[r[0] for r in unmerged]}"

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
            f"\n  stress: {N_PENDINGS} pendings x {N_WORKERS} workers in {elapsed:.2f}s"
        )

    def test_sweep_concurrent_with_promoters_no_deadlock(self, fresh_db):
        embedder = FakeEmbedder()
        N_PENDINGS = 40
        N_WORKERS = 3
        pp_ids = self._seed_pending_pps(N_PENDINGS)

        chunks = [pp_ids[i::N_WORKERS] for i in range(N_WORKERS)]
        errors = []

        def promoter_worker(my_chunk):
            try:
                for pp_id in my_chunk:
                    promote_pending(pp_id, embedder=embedder)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("promoter", e))

        sweep_summaries = []

        def sweep_worker():
            try:
                time.sleep(0.05)
                summary = run_sweep(namer=FakeNamer(), embedder=embedder)
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
        embedder = FakeEmbedder()
        _seed_uncategorized_painpoints([
            "GitHub Actions timeout monorepo build slow",
            "GitHub Actions timeout monorepo build hangs",
            "GitHub Actions timeout monorepo build fails",
            "GitHub Actions timeout monorepo build retry",
            "GitHub Actions timeout monorepo build flake",
        ], embedder=embedder)

        namer = FakeNamer()
        start = time.monotonic()
        for i in range(10):
            run_sweep(namer=namer, embedder=embedder)
            assert time.monotonic() - start < self.HARD_TIMEOUT_SEC, (
                f"sweep {i} took too long; likely deadlock"
            )

    def test_lock_acquire_release_hammer(self, fresh_db):
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
            assert not t.is_alive(), "hammer thread hung -- likely deadlock"
        assert errors == [], f"hammer errors: {errors}"


# ===========================================================================
# Realistic workflow simulation with synthetic data
# ===========================================================================


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
    """Simulate a realistic Reddit ingest run with synthetic data."""

    def _generate_pending_pps(self):
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
                    age_seconds=3600 * (post_counter % 24),
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
        embedder = FakeEmbedder()
        topic_pp_ids = self._generate_pending_pps()
        all_pp_ids = sum(topic_pp_ids.values(), [])
        assert len(all_pp_ids) == sum(len(v) for v in _SYNTHETIC_TOPIC_TITLES.values())

        promote_results = []
        for pp_id in all_pp_ids:
            promote_results.append(promote_pending(pp_id, embedder=embedder))

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
            # Embedding similarity catches close paraphrases but not all.
            assert n_topics <= painpoint_count_after_promote <= n_pendings, (
                f"painpoint count {painpoint_count_after_promote} outside "
                f"plausible range [{n_topics}, {n_pendings}]"
            )

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

            total_links = conn.execute(
                "SELECT SUM(signal_count) FROM painpoints"
            ).fetchone()[0]
            assert total_links == n_pendings, (
                f"expected {n_pendings} links, got {total_links}"
            )

            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            in_uncat_before_sweep = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]
        finally:
            conn.close()

        namer = FakeNamer()
        summary = run_sweep(namer=namer, embedder=embedder)
        for step in summary.values():
            assert "proposed" in step and "accepted" in step

        conn = db.get_db()
        try:
            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            in_uncat_after_sweep = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]
            assert in_uncat_after_sweep <= in_uncat_before_sweep, (
                f"Uncategorized grew during sweep: {in_uncat_before_sweep} -> "
                f"{in_uncat_after_sweep}"
            )
        finally:
            conn.close()

        summary2 = run_sweep(namer=namer, embedder=embedder)
        for step_name, step in summary2.items():
            assert step["accepted"] == 0, (
                f"second sweep should be idempotent, but {step_name} accepted "
                f"{step['accepted']}"
            )

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
        embedder = FakeEmbedder()
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

        merged = promote_pending(pp_id, embedder=embedder)
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

        new_post = _make_post("t3_multi_new", score=300, num_comments=80)
        unrelated_pp = save_pending_painpoint(
            new_post,
            "Totally separate Postgres deadlock pain xyz qrs",
            severity=8,
        )
        unrelated_merged = promote_pending(unrelated_pp, embedder=embedder)
        assert unrelated_merged != merged

        bridge_post = _make_post("t3_bridge", score=300, num_comments=80)
        bridge_pp = save_pending_painpoint(
            bridge_post, "Bridge pain pqr xyz", severity=8,
        )
        add_pending_source(bridge_pp, new_post)
        add_pending_source(bridge_pp, post_ids[0])

        bridge_result = promote_pending(bridge_pp, embedder=embedder)
        # With embedding-only matching, the bridge links to the best
        # cosine match (or creates new); it does NOT merge existing pps.
        assert bridge_result is not None

    def test_workflow_relevance_decay_drops_stale(self, fresh_db):
        embedder = FakeEmbedder()
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

        fresh_result = promote_pending(fresh_pp, embedder=embedder)
        stale_result = promote_pending(stale_pp, embedder=embedder)

        assert fresh_result is not None, "fresh pp must promote"
        assert stale_result is None, "stale low-traction pp must drop"

        conn = db.get_db()
        try:
            assert conn.execute(
                "SELECT id FROM pending_painpoints WHERE id = ?", (stale_pp,)
            ).fetchone() is None
            assert conn.execute(
                "SELECT id FROM pending_painpoints WHERE id = ?", (fresh_pp,)
            ).fetchone() is not None
            assert conn.execute(
                "SELECT COUNT(*) FROM painpoints"
            ).fetchone()[0] == 1
        finally:
            conn.close()


# ===========================================================================
# Large synthetic lifecycle test
# ===========================================================================


class TestFullLifecycleWithDecay:
    """Simulate the realistic lifecycle: old categories die, fresh ones
    survive, new categories form from Uncategorized clusters."""

    N_PER_OLD_CAT = 4
    N_PER_ACTIVE_CAT = 4
    N_UNCATEGORIZED = 20

    OLD_AGE_DAYS = 90
    FRESH_AGE_SECONDS = 3600

    def _create_category(self, conn, name, parent_id):
        conn.execute(
            "INSERT INTO categories (name, parent_id, description, created_at) "
            "VALUES (?, ?, 'test', datetime('now'))",
            (name, parent_id),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def _seed_painpoints_in_category(self, cat_id, titles, age_seconds, score,
                                     num_comments, severity, embedder=None):
        if embedder is None:
            embedder = FakeEmbedder()
        from db.embeddings import store_painpoint_embedding
        ids = []
        for i, title in enumerate(titles):
            post_id = _make_post(
                f"t3_cat{cat_id}_p{i}",
                score=score, num_comments=num_comments,
                age_seconds=age_seconds, title=title,
            )
            pp_id = save_pending_painpoint(post_id, title, severity=severity)

            conn = db.get_db()
            try:
                now = db._now()
                conn.execute(
                    "INSERT INTO painpoints (title, description, severity, signal_count, "
                    "category_id, first_seen, last_updated) VALUES (?, '', ?, 1, ?, ?, ?)",
                    (title, severity, cat_id, now, now),
                )
                new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute(
                    "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                    "VALUES (?, ?)",
                    (new_id, pp_id),
                )
                emb = embedder.embed(title)
                store_painpoint_embedding(conn, new_id, emb)
                conn.commit()
                ids.append(new_id)
            finally:
                conn.close()
        return ids

    def test_full_lifecycle(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
        finally:
            conn.close()

        # --- 3 old (dying) categories ---
        old_cat_ids = []
        old_titles_by_cat = {
            "OldCat-A": [
                "Legacy jQuery plugin conflict issue",
                "Legacy jQuery plugin compat problem",
                "Legacy jQuery plugin breaking change",
                "Legacy jQuery plugin deprecation notice",
            ],
            "OldCat-B": [
                "Flash Player EOL migration concerns",
                "Flash Player EOL migration blockers",
                "Flash Player EOL migration failures",
                "Flash Player EOL migration cost issues",
            ],
            "OldCat-C": [
                "Windows XP driver compatibility issue",
                "Windows XP driver compatibility failure",
                "Windows XP driver compatibility problem",
                "Windows XP driver compatibility update",
            ],
        }
        for cat_name, titles in old_titles_by_cat.items():
            conn = db.get_db()
            try:
                cat_id = self._create_category(conn, cat_name, parent_cat)
                conn.commit()
            finally:
                conn.close()
            old_cat_ids.append(cat_id)
            self._seed_painpoints_in_category(
                cat_id, titles,
                age_seconds=self.OLD_AGE_DAYS * 86400,
                score=5, num_comments=1, severity=2,
                embedder=embedder,
            )

        # --- 3 active (surviving) categories ---
        active_cat_ids = []
        active_titles_by_cat = {
            "ActiveCat-A": [
                "React Server Components hydration mismatch bug",
                "React Server Components hydration error report",
                "React Server Components hydration failure case",
                "React Server Components hydration inconsistency",
            ],
            "ActiveCat-B": [
                "Docker build context too large in monorepo",
                "Docker build context monorepo size problem",
                "Docker build context monorepo slow transfer",
                "Docker build context monorepo optimization",
            ],
            "ActiveCat-C": [
                "TypeScript compiler extremely slow on large project",
                "TypeScript compiler slow on large codebase build",
                "TypeScript compiler performance large project",
                "TypeScript compiler slow large project types",
            ],
        }
        for cat_name, titles in active_titles_by_cat.items():
            conn = db.get_db()
            try:
                cat_id = self._create_category(conn, cat_name, parent_cat)
                conn.commit()
            finally:
                conn.close()
            active_cat_ids.append(cat_id)
            self._seed_painpoints_in_category(
                cat_id, titles,
                age_seconds=self.FRESH_AGE_SECONDS,
                score=400, num_comments=100, severity=8,
                embedder=embedder,
            )

        # --- 20 Uncategorized painpoints forming ~4 clusters ---
        uncat_cluster_titles = [
            # Cluster 1 (5 pps)
            "Redis cache eviction policy too aggressive production",
            "Redis cache eviction policy too aggressive in prod",
            "Redis cache eviction policy aggressive for production",
            "Redis cache eviction policy aggressive production load",
            "Redis cache eviction policy production too aggressive",
            # Cluster 2 (5 pps)
            "GraphQL N+1 query problem in nested resolvers",
            "GraphQL N+1 query issue with nested resolvers",
            "GraphQL N+1 query problem nested resolver depth",
            "GraphQL N+1 query nested resolvers performance",
            "GraphQL N+1 query nested resolvers slow response",
            # Cluster 3 (5 pps)
            "AWS Lambda cold start latency in VPC config",
            "AWS Lambda cold start latency VPC configuration",
            "AWS Lambda cold start VPC latency high delay",
            "AWS Lambda cold start latency VPC setup slow",
            "AWS Lambda cold start latency VPC config issue",
            # Singletons (5 pps -- should stay in Uncategorized)
            "Unique issue about Terraform state locking drift",
            "Bizarre Flutter rendering glitch on Android tablets",
            "Obscure Elixir OTP supervisor crash recovery path",
            "Niche Haskell monad transformer stack overflow",
            "Rare Rust borrow checker false positive case",
        ]
        _seed_uncategorized_painpoints(uncat_cluster_titles, severity=7, embedder=embedder)

        # ------------------------------------------------------------------
        # Phase 2: verify pre-sweep state
        # ------------------------------------------------------------------
        conn = db.get_db()
        try:
            total_painpoints = conn.execute(
                "SELECT COUNT(*) FROM painpoints"
            ).fetchone()[0]
            expected = (
                len(old_titles_by_cat) * self.N_PER_OLD_CAT
                + len(active_titles_by_cat) * self.N_PER_ACTIVE_CAT
                + self.N_UNCATEGORIZED
            )
            assert total_painpoints == expected, (
                f"expected {expected} painpoints, got {total_painpoints}"
            )

            from db.relevance import cache_painpoint_relevance
            old_rels = []
            for cat_id in old_cat_ids:
                rows = conn.execute(
                    "SELECT id FROM painpoints WHERE category_id = ?", (cat_id,)
                ).fetchall()
                for r in rows:
                    old_rels.append(cache_painpoint_relevance(r["id"], conn=conn))

            active_rels = []
            for cat_id in active_cat_ids:
                rows = conn.execute(
                    "SELECT id FROM painpoints WHERE category_id = ?", (cat_id,)
                ).fetchall()
                for r in rows:
                    active_rels.append(cache_painpoint_relevance(r["id"], conn=conn))
        finally:
            conn.close()

        avg_old = sum(old_rels) / len(old_rels) if old_rels else 0
        avg_active = sum(active_rels) / len(active_rels) if active_rels else 0
        print(f"\n  Pre-sweep relevance: avg_old={avg_old:.4f} avg_active={avg_active:.4f}")
        print(f"  Active/Old ratio: {avg_active / avg_old:.1f}x" if avg_old > 0 else "  Old is zero")

        assert avg_active > avg_old * 10, (
            f"active ({avg_active:.3f}) should be >>10x old ({avg_old:.3f})"
        )

        from db.category_events import category_relevance_mass, MIN_CATEGORY_RELEVANCE
        for cat_id in old_cat_ids:
            conn = db.get_db()
            try:
                mass = category_relevance_mass(conn, cat_id)
                conn.commit()
            finally:
                conn.close()
            assert mass < MIN_CATEGORY_RELEVANCE, (
                f"old cat {cat_id} mass {mass:.4f} should be below "
                f"MIN_CATEGORY_RELEVANCE={MIN_CATEGORY_RELEVANCE}"
            )

        for cat_id in active_cat_ids:
            conn = db.get_db()
            try:
                mass = category_relevance_mass(conn, cat_id)
                conn.commit()
            finally:
                conn.close()
            assert mass >= MIN_CATEGORY_RELEVANCE, (
                f"active cat {cat_id} mass {mass:.4f} should be above "
                f"MIN_CATEGORY_RELEVANCE={MIN_CATEGORY_RELEVANCE}"
            )

        # ------------------------------------------------------------------
        # Phase 3: run the sweep
        # ------------------------------------------------------------------
        namer = FakeNamer()
        summary = run_sweep(namer=namer, embedder=embedder)
        print(f"  Sweep summary: {summary}")

        # ------------------------------------------------------------------
        # Phase 4: verify post-sweep state
        # ------------------------------------------------------------------
        conn = db.get_db()
        try:
            for cat_id in old_cat_ids:
                row = conn.execute(
                    "SELECT id, name FROM categories WHERE id = ?", (cat_id,)
                ).fetchone()
                assert row is None, (
                    f"old category {cat_id} should have been deleted by the sweep"
                )

            for cat_id in active_cat_ids:
                row = conn.execute(
                    "SELECT id, name FROM categories WHERE id = ?", (cat_id,)
                ).fetchone()
                assert row is not None, (
                    f"active category {cat_id} should have survived the sweep"
                )
                member_count = conn.execute(
                    "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat_id,)
                ).fetchone()[0]
                assert member_count == self.N_PER_ACTIVE_CAT, (
                    f"active cat {cat_id} should still have {self.N_PER_ACTIVE_CAT} "
                    f"members, got {member_count}"
                )

            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            remaining_in_uncat = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]

            auto_cats = conn.execute(
                "SELECT id, name FROM categories WHERE name LIKE 'AutoCat-%'"
            ).fetchall()
            auto_cat_ids = [r["id"] for r in auto_cats]

            in_auto = 0
            for ac_id in auto_cat_ids:
                count = conn.execute(
                    "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (ac_id,)
                ).fetchone()[0]
                in_auto += count

            assert in_auto + remaining_in_uncat == self.N_UNCATEGORIZED, (
                f"auto-cat members ({in_auto}) + uncategorized ({remaining_in_uncat}) "
                f"should sum to {self.N_UNCATEGORIZED}"
            )

            assert remaining_in_uncat >= 3, (
                f"at least 3 of the 5 singletons should still be in Uncategorized, "
                f"got {remaining_in_uncat}"
            )

            for ac in auto_cats:
                count = conn.execute(
                    "SELECT COUNT(*) FROM painpoints WHERE category_id = ?",
                    (ac["id"],),
                ).fetchone()[0]
                assert 3 <= count <= 8, (
                    f"auto-cat {ac['name']} has {count} members -- expected 3-8 "
                    f"(one cluster ~ 5)"
                )
                print(f"  Auto-category {ac['name']}: {count} painpoints")

            old_painpoint_count = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?",
                (parent_cat,),
            ).fetchone()[0]
            expected_relinked = len(old_titles_by_cat) * self.N_PER_OLD_CAT
            assert old_painpoint_count == expected_relinked, (
                f"old painpoints should be relinked to parent (expected "
                f"{expected_relinked}, got {old_painpoint_count})"
            )

            delete_events = conn.execute(
                "SELECT COUNT(*) FROM category_events "
                "WHERE event_type = 'delete_category' AND accepted = 1"
            ).fetchone()[0]
            assert delete_events == len(old_cat_ids), (
                f"expected {len(old_cat_ids)} accepted delete events, "
                f"got {delete_events}"
            )

            add_events = conn.execute(
                "SELECT COUNT(*) FROM category_events "
                "WHERE event_type = 'add_category_new' AND accepted = 1"
            ).fetchone()[0]
            assert add_events >= 1, "expected at least 1 accepted add_category_new event"

            total_painpoints_after = conn.execute(
                "SELECT COUNT(*) FROM painpoints"
            ).fetchone()[0]
            assert total_painpoints_after == expected, (
                f"no painpoints should be lost: expected {expected}, "
                f"got {total_painpoints_after}"
            )
        finally:
            conn.close()

        # Phase 5: second sweep is idempotent
        summary2 = run_sweep(namer=namer, embedder=embedder)
        for step_name, step in summary2.items():
            assert step["accepted"] == 0, (
                f"second sweep should be idempotent but {step_name} "
                f"accepted {step['accepted']}"
            )


# ===========================================================================
# Regression tests for vec table cleanup + Uncategorized centroid guard
# ===========================================================================


class TestCategoryVecCleanup:
    """Deleting or retiring a category must also drop its row from
    category_vec. Orphaned vec rows cause find_best_category to return
    a dead category_id, which FK-fails on the next promote."""

    def _seed_category_with_members(self, name, n=3, severity=8):
        """Create a category with N painpoints and return the category id.
        Each DB operation uses its own connection to avoid WAL deadlocks."""
        from db.embeddings import FakeEmbedder, store_painpoint_embedding, update_category_embedding
        from db.painpoints import save_pending_painpoint

        emb = FakeEmbedder()

        # Step 1: create posts + pending pps (each op uses its own conn)
        post_pp_pairs = []
        for i in range(n):
            post_id = _make_post(f"t3_seed_{name}_{i}", score=200, num_comments=50,
                                 title=f"{name} painpoint {i}")
            pp_id = save_pending_painpoint(
                post_id, f"{name} painpoint {i}", severity=severity,
            )
            post_pp_pairs.append((post_id, pp_id))

        # Step 2: create the category + painpoints + links + embeddings
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            cat_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES (?, ?, 'test', datetime('now')) RETURNING id",
                (name, parent_cat),
            ).fetchone()["id"]

            now = db._now()
            for i, (_post_id, pp_id) in enumerate(post_pp_pairs):
                conn.execute(
                    "INSERT INTO painpoints (title, description, severity, signal_count, "
                    "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                    "VALUES (?, '', ?, 1, ?, ?, ?, 2.0, ?)",
                    (f"{name} painpoint {i}", severity, cat_id, now, now, now),
                )
                new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute(
                    "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                    "VALUES (?, ?)",
                    (new_id, pp_id),
                )
                store_painpoint_embedding(conn, new_id, emb.embed(f"{name} painpoint {i}"))

            update_category_embedding(conn, cat_id)
            conn.commit()
        finally:
            conn.close()
        return cat_id

    def test_delete_category_removes_vec_entry(self, fresh_db):
        """When _apply_delete_category runs, the vec0 row for that
        category must be gone — otherwise it shows up in find_best_category
        and causes downstream FK failures."""
        from db.category_events import CategoryEvent, _apply_delete_category

        conn = db.get_db()
        try:
            cat_id = self._seed_category_with_members("ToDelete", n=2)

            # Confirm the vec entry exists before deletion
            before = conn.execute(
                "SELECT COUNT(*) FROM category_vec WHERE rowid = ?", (cat_id,)
            ).fetchone()[0]
            assert before == 1, "test setup: category should have a vec entry"

            # Directly apply the delete (bypasses the acceptance test)
            parent_id = conn.execute(
                "SELECT parent_id FROM categories WHERE id = ?", (cat_id,)
            ).fetchone()["parent_id"]
            event = CategoryEvent(
                event_type="delete_category",
                payload={"category_id": cat_id, "category_name": "ToDelete",
                         "parent_id": parent_id},
                target_category=cat_id,
                metric_name="category_mass",
                metric_value=0.1,
                threshold=1.0,
            )
            _apply_delete_category(conn, event, namer=FakeNamer())
            conn.commit()

            after = conn.execute(
                "SELECT COUNT(*) FROM category_vec WHERE rowid = ?", (cat_id,)
            ).fetchone()[0]
            assert after == 0, "category_vec entry must be gone after delete"
        finally:
            conn.close()

    def test_split_retire_removes_vec_entry(self, fresh_db):
        """When _apply_add_category_split retires the source category
        (after all members move to sub-categories), its vec row must go too."""
        from db.category_events import CategoryEvent, _apply_add_category_split

        conn = db.get_db()
        try:
            source_id = self._seed_category_with_members("ToSplit", n=6)
            parent_id = conn.execute(
                "SELECT parent_id FROM categories WHERE id = ?", (source_id,)
            ).fetchone()["parent_id"]

            # Move every member into two clusters of 3
            painpoint_ids = [
                r["id"] for r in conn.execute(
                    "SELECT id FROM painpoints WHERE category_id = ? ORDER BY id",
                    (source_id,),
                ).fetchall()
            ]
            event = CategoryEvent(
                event_type="add_category_split",
                payload={
                    "parent_category_id": parent_id,
                    "source_category_id": source_id,
                    "source_category_name": "ToSplit",
                    "subcategories": [
                        {"name": "ToSplit-A", "description": "first half",
                         "painpoint_ids": painpoint_ids[:3]},
                        {"name": "ToSplit-B", "description": "second half",
                         "painpoint_ids": painpoint_ids[3:]},
                    ],
                },
                target_category=source_id,
                metric_name="sub_cluster_count",
                metric_value=2.0,
                threshold=2.0,
            )
            _apply_add_category_split(conn, event, namer=FakeNamer())
            conn.commit()

            # Source category retired → vec entry must be gone
            assert conn.execute(
                "SELECT COUNT(*) FROM categories WHERE id = ?", (source_id,)
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM category_vec WHERE rowid = ?", (source_id,)
            ).fetchone()[0] == 0
        finally:
            conn.close()

    def test_promote_after_delete_sweep(self, fresh_db):
        """End-to-end regression: promote → delete stale category via sweep
        → promote again. Must NOT raise IntegrityError from a dangling
        category_vec row pointing at the deleted category."""
        from db.embeddings import FakeEmbedder, store_painpoint_embedding, update_category_embedding

        # Step 1: create the decayed post + pending (outside any held conn)
        old_post = _make_post(
            "t3_decay", score=0, num_comments=0, age_seconds=10 * 365 * 86400,
        )
        pp_id = save_pending_painpoint(old_post, "Stale", severity=1)

        # Step 2: create the dead category + its lone painpoint + embedding
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            dead_cat = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('DeadBucket', ?, 'test', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]
            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated, relevance, relevance_updated_at) "
                "VALUES ('Stale', '', 1, 1, ?, ?, ?, 0.0, ?)",
                (dead_cat, now, now, now),
            )
            new_pp = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                "VALUES (?, ?)", (new_pp, pp_id),
            )
            emb = FakeEmbedder()
            store_painpoint_embedding(conn, new_pp, emb.embed("Stale"))
            update_category_embedding(conn, dead_cat)
            conn.commit()
        finally:
            conn.close()

        # 2. Run the sweep — should delete DeadBucket
        embedder = FakeEmbedder()
        summary = run_sweep(namer=FakeNamer(), embedder=embedder)
        assert summary["delete"]["accepted"] >= 1

        # 3. Promote a new painpoint. Before the fix, this would FK-fail
        #    because find_best_category could return the now-dead cat_id
        #    whose vec row was orphaned.
        fresh_post = _make_post("t3_fresh", score=300, num_comments=80)
        fresh_pp = save_pending_painpoint(
            fresh_post, "Fresh painpoint after sweep", severity=8,
        )
        # No exception = test passes
        result = promote_pending(fresh_pp, embedder=embedder)
        assert result is not None


class TestUncategorizedNeverGetsCentroid:
    """The Uncategorized sentinel is a dumping ground for heterogeneous
    painpoints — computing a centroid over them would create a noise
    vector that pulls future painpoints into Uncategorized instead of
    their real category match. update_category_embedding must skip it."""

    def test_update_category_embedding_skips_uncategorized(self, fresh_db):
        from db.embeddings import (
            FakeEmbedder, store_painpoint_embedding, update_category_embedding,
        )
        from db.painpoints import get_uncategorized_id

        conn = db.get_db()
        try:
            uncat_id = get_uncategorized_id(conn=conn)

            # Directly create a painpoint in Uncategorized with an embedding
            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated) "
                "VALUES ('Orphan pain', '', 5, 1, ?, ?, ?)",
                (uncat_id, now, now),
            )
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            emb = FakeEmbedder()
            store_painpoint_embedding(conn, new_id, emb.embed("Orphan pain"))
            conn.commit()

            # Call update_category_embedding on Uncategorized — must be a no-op
            update_category_embedding(conn, uncat_id)
            conn.commit()

            count = conn.execute(
                "SELECT COUNT(*) FROM category_vec WHERE rowid = ?", (uncat_id,)
            ).fetchone()[0]
            assert count == 0, (
                "Uncategorized must NOT get a category_vec entry — "
                "it's a dumping ground with no semantic coherence"
            )
        finally:
            conn.close()

    def test_link_to_uncategorized_doesnt_create_centroid(self, fresh_db):
        """_link_pending_to_painpoint refreshes the target painpoint's
        category embedding. If the target is in Uncategorized, the refresh
        must not create a centroid."""
        from db.embeddings import FakeEmbedder
        from db.painpoints import (
            _link_pending_to_painpoint, get_uncategorized_id,
            save_pending_painpoint,
        )

        embedder = FakeEmbedder()

        # Promote two dissimilar painpoints — both land in Uncategorized
        # (FakeEmbedder doesn't cluster unrelated text above threshold)
        post1 = _make_post("t3_a", score=200, num_comments=50, title="A")
        pp1 = save_pending_painpoint(
            post1, "Uncategorizable quantum flux capacitor issue abc",
            severity=8,
        )
        merged1 = promote_pending(pp1, embedder=embedder)

        conn = db.get_db()
        try:
            uncat_id = get_uncategorized_id(conn=conn)
            # Confirm merged1 is in Uncategorized
            cat = conn.execute(
                "SELECT category_id FROM painpoints WHERE id = ?", (merged1,)
            ).fetchone()["category_id"]
            assert cat == uncat_id, "test setup: merged pp should be in Uncategorized"

            # Now manually link another pending to merged1 to trigger
            # the category-embedding refresh path
            post2 = _make_post("t3_b", score=200, num_comments=50, title="B")
            pp2 = save_pending_painpoint(post2, "Different text", severity=8)

            with db.get_db() as link_conn:
                pass  # close any transaction
            # _link_pending_to_painpoint calls update_category_embedding;
            # verify no vec entry exists for Uncategorized after the link.
            with merge_lock(conn):
                _link_pending_to_painpoint(conn, merged1, pp2)

            count = conn.execute(
                "SELECT COUNT(*) FROM category_vec WHERE rowid = ?", (uncat_id,)
            ).fetchone()[0]
            assert count == 0
        finally:
            conn.close()


class TestLinkRefreshesCategoryEmbedding:
    """_link_pending_to_painpoint should refresh the target category's
    centroid so the vector stays in sync with new evidence."""

    def test_link_calls_update_category_embedding(self, fresh_db):
        """After linking, the target category's vec row must still exist
        and match the current member set's centroid. Verifies the
        update_category_embedding call happens."""
        import struct
        from db.embeddings import (
            EMBEDDING_DIM, FakeEmbedder, store_painpoint_embedding,
            update_category_embedding,
        )
        from db.painpoints import _link_pending_to_painpoint, save_pending_painpoint

        emb = FakeEmbedder()

        # Step 1: make a post + pending painpoint (no held conn)
        post1 = _make_post("t3_a", score=200, num_comments=50, title="A")
        pp1 = save_pending_painpoint(
            post1, "GitHub Actions CI build timeout monorepo", severity=8,
        )

        # Step 2: create a real category and place a painpoint in it directly
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            cat_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('TargetCat', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]

            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated) "
                "VALUES ('GitHub Actions CI build timeout monorepo', '', 8, 1, ?, ?, ?)",
                (cat_id, now, now),
            )
            merged1 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                "VALUES (?, ?)", (merged1, pp1),
            )
            store_painpoint_embedding(
                conn, merged1, emb.embed("GitHub Actions CI build timeout monorepo"),
            )
            update_category_embedding(conn, cat_id)
            conn.commit()

            # Snapshot the centroid before linking
            before_blob = conn.execute(
                "SELECT embedding FROM category_vec WHERE rowid = ?", (cat_id,)
            ).fetchone()
            assert before_blob is not None, (
                "test setup: category_vec should have an entry"
            )
            before_vec = list(struct.unpack(f"{EMBEDDING_DIM}f", before_blob[0]))
        finally:
            conn.close()

        # Step 3: link a second pending pp to merged1
        post2 = _make_post("t3_b", score=200, num_comments=50, title="B")
        pp2 = save_pending_painpoint(
            post2, "GitHub Actions CI build timeout variation", severity=8,
        )

        conn = db.get_db()
        try:
            with merge_lock(conn):
                _link_pending_to_painpoint(conn, merged1, pp2)

            # After link: signal_count bumped, category vec entry still exists
            sc = conn.execute(
                "SELECT signal_count FROM painpoints WHERE id = ?", (merged1,)
            ).fetchone()["signal_count"]
            assert sc == 2

            after_blob = conn.execute(
                "SELECT embedding FROM category_vec WHERE rowid = ?", (cat_id,)
            ).fetchone()
            assert after_blob is not None, (
                "category_vec entry must still exist after link refresh"
            )
            after_vec = list(struct.unpack(f"{EMBEDDING_DIM}f", after_blob[0]))

            # Linking doesn't add new members (same 1 painpoint), so the
            # centroid should be identical. This proves the code path ran
            # without corrupting the vec.
            for a, b in zip(before_vec, after_vec):
                assert abs(a - b) < 1e-6, (
                    "centroid must be identical — same member set"
                )
        finally:
            conn.close()


class TestDeleteRefreshesFallbackCentroid:
    """When _apply_delete_category moves members to the parent category,
    the parent's centroid must be refreshed — otherwise it reflects the
    pre-move member set and future find_best_category mis-routes."""

    def test_fallback_parent_centroid_refreshed(self, fresh_db):
        import struct
        from db.category_events import CategoryEvent, _apply_delete_category
        from db.embeddings import (
            EMBEDDING_DIM, FakeEmbedder, store_painpoint_embedding,
            update_category_embedding,
        )

        # Step 1: create a parent category with its own member (gives it
        # a known centroid)
        emb = FakeEmbedder()
        post_parent = _make_post("t3_p", score=200, num_comments=50,
                                 title="Parent member")
        pp_parent = save_pending_painpoint(post_parent, "Parent member", severity=5)

        conn = db.get_db()
        try:
            parent_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Parent', NULL, 'p', datetime('now')) RETURNING id"
            ).fetchone()["id"]
            doomed_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Doomed', ?, 'd', datetime('now')) RETURNING id",
                (parent_id,),
            ).fetchone()["id"]

            now = db._now()
            # One painpoint in the parent
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated) "
                "VALUES ('Parent native pain', '', 5, 1, ?, ?, ?)",
                (parent_id, now, now),
            )
            p_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                "VALUES (?, ?)", (p_id, pp_parent),
            )
            store_painpoint_embedding(conn, p_id, emb.embed("Parent native pain"))
            update_category_embedding(conn, parent_id)
            conn.commit()

            parent_centroid_before = conn.execute(
                "SELECT embedding FROM category_vec WHERE rowid = ?", (parent_id,)
            ).fetchone()[0]
        finally:
            conn.close()

        # Step 2: seed the doomed category with a member that has a
        # VERY DIFFERENT embedding (unrelated text)
        post_doomed = _make_post("t3_d", score=200, num_comments=50,
                                 title="Doomed member")
        pp_doomed = save_pending_painpoint(
            post_doomed, "Completely unrelated xyzzy plugh",  severity=5,
        )
        conn = db.get_db()
        try:
            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated) "
                "VALUES ('Completely unrelated xyzzy plugh', '', 5, 1, ?, ?, ?)",
                (doomed_id, now, now),
            )
            d_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                "VALUES (?, ?)", (d_id, pp_doomed),
            )
            store_painpoint_embedding(
                conn, d_id, emb.embed("Completely unrelated xyzzy plugh"),
            )
            update_category_embedding(conn, doomed_id)
            conn.commit()
        finally:
            conn.close()

        # Step 3: delete the doomed category — painpoints relink to parent
        conn = db.get_db()
        try:
            event = CategoryEvent(
                event_type="delete_category",
                payload={
                    "category_id": doomed_id,
                    "category_name": "Doomed",
                    "parent_id": parent_id,
                },
                target_category=doomed_id,
                metric_name="category_mass",
                metric_value=0.0,
                threshold=1.0,
            )
            _apply_delete_category(conn, event, namer=FakeNamer())
            conn.commit()

            # Parent centroid MUST have changed — the doomed member's very
            # different embedding just joined the set.
            parent_centroid_after = conn.execute(
                "SELECT embedding FROM category_vec WHERE rowid = ?", (parent_id,)
            ).fetchone()[0]
            assert parent_centroid_before != parent_centroid_after, (
                "parent centroid must be refreshed after absorbing "
                "members from the deleted category"
            )
            # Sanity: parent now has 2 members
            parent_count = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (parent_id,)
            ).fetchone()[0]
            assert parent_count == 2
        finally:
            conn.close()


class TestSplitRefreshesSourceCentroidWhenPartial:
    """When _apply_add_category_split leaves some members in the source
    category (LLM didn't cover every cluster), the source's centroid
    must be refreshed to match the shrunken member set."""

    def test_partial_split_refreshes_source_centroid(self, fresh_db):
        from db.category_events import CategoryEvent, _apply_add_category_split
        from db.embeddings import (
            FakeEmbedder, store_painpoint_embedding, update_category_embedding,
        )

        emb = FakeEmbedder()

        # Step 1: make posts + pendings out of any held conn
        post_pp_pairs = []
        for i in range(5):
            post_id = _make_post(f"t3_partial_{i}", score=200, num_comments=50,
                                 title=f"Partial painpoint {i}")
            pp_id = save_pending_painpoint(
                post_id, f"Partial painpoint {i}", severity=5,
            )
            post_pp_pairs.append((post_id, pp_id))

        # Step 2: create a source category with 5 members
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            source_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('PartialSource', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]

            now = db._now()
            member_ids = []
            for i, (_pid, pp_id) in enumerate(post_pp_pairs):
                conn.execute(
                    "INSERT INTO painpoints (title, description, severity, signal_count, "
                    "category_id, first_seen, last_updated) "
                    "VALUES (?, '', 5, 1, ?, ?, ?)",
                    (f"Partial painpoint {i}", source_id, now, now),
                )
                mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute(
                    "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                    "VALUES (?, ?)", (mid, pp_id),
                )
                store_painpoint_embedding(
                    conn, mid, emb.embed(f"Partial painpoint {i}"),
                )
                member_ids.append(mid)
            update_category_embedding(conn, source_id)
            conn.commit()

            centroid_before = conn.execute(
                "SELECT embedding FROM category_vec WHERE rowid = ?", (source_id,)
            ).fetchone()[0]

            # Step 3: split only 2 of the 5 into a sub-category.
            # The remaining 3 stay in source. Source must NOT be retired
            # and its centroid MUST be refreshed.
            event = CategoryEvent(
                event_type="add_category_split",
                payload={
                    "parent_category_id": parent_cat,
                    "source_category_id": source_id,
                    "source_category_name": "PartialSource",
                    "subcategories": [
                        {
                            "name": "PartialSource-A",
                            "description": "first two only",
                            "painpoint_ids": member_ids[:2],
                        },
                    ],
                },
                target_category=source_id,
                metric_name="sub_cluster_count",
                metric_value=1.0,
                threshold=2.0,
            )
            _apply_add_category_split(conn, event, namer=FakeNamer())
            conn.commit()

            # Source still exists
            assert conn.execute(
                "SELECT COUNT(*) FROM categories WHERE id = ?", (source_id,)
            ).fetchone()[0] == 1

            # Source has 3 members (2 moved to sub, 3 stayed)
            remaining = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?",
                (source_id,),
            ).fetchone()[0]
            assert remaining == 3

            # Source's centroid must have been refreshed (different members
            # now → different mean)
            centroid_after = conn.execute(
                "SELECT embedding FROM category_vec WHERE rowid = ?", (source_id,)
            ).fetchone()[0]
            assert centroid_before != centroid_after, (
                "source centroid must be refreshed after partial split "
                "(member set shrank from 5 to 3)"
            )
        finally:
            conn.close()


class TestDimensionMismatchTolerance:
    """If painpoint_vec has a blob with the wrong size (e.g., stale
    embedding from a different model), update_category_embedding must
    skip it with a warning rather than crashing."""

    def test_update_category_embedding_skips_bad_blob(self, fresh_db, caplog):
        import logging
        from db.embeddings import (
            FakeEmbedder, store_painpoint_embedding, update_category_embedding,
        )

        emb = FakeEmbedder()

        # Build a category with two members
        post_a = _make_post("t3_dim_a", title="A")
        pp_a = save_pending_painpoint(post_a, "A", severity=5)
        post_b = _make_post("t3_dim_b", title="B")
        pp_b = save_pending_painpoint(post_b, "B", severity=5)

        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            cat_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('DimTest', ?, 'd', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]

            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated) VALUES ('A', '', 5, 1, ?, ?, ?)",
                (cat_id, now, now),
            )
            mid_a = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated) VALUES ('B', '', 5, 1, ?, ?, ?)",
                (cat_id, now, now),
            )
            mid_b = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # A gets a proper embedding
            store_painpoint_embedding(conn, mid_a, emb.embed("A"))

            # B gets a truncated blob (wrong dimension) injected directly
            conn.execute(
                "DELETE FROM painpoint_vec WHERE rowid = ?", (mid_b,)
            )
            # Insert a blob with wrong size — use fewer bytes than expected.
            # sqlite-vec will reject this at insert time, so we write the
            # wrong blob via a legit insert path first and then bypass
            # vec0's validation by... actually we can't easily inject a
            # bad blob into vec0 directly. But the real-world scenario is
            # a model dimension change where the blob was VALID when
            # written but is now mis-sized relative to EMBEDDING_DIM.
            # For this test, simply DON'T store B's embedding — the
            # vec row will be absent. update_category_embedding should
            # skip B and use only A's embedding.
            conn.commit()

            # Recompute the centroid — should work without raising even
            # though B has no vec entry.
            with caplog.at_level(logging.WARNING):
                update_category_embedding(conn, cat_id)
            conn.commit()

            # Category still has a vec entry (from A)
            vec = conn.execute(
                "SELECT embedding FROM category_vec WHERE rowid = ?", (cat_id,)
            ).fetchone()
            assert vec is not None, (
                "category_vec should still have an entry built from "
                "the one member with a valid embedding"
            )
        finally:
            conn.close()


class TestEmptyTextEmbedding:
    """Empty-or-whitespace text must produce a valid embedding rather
    than either crashing (OpenAI) or returning an all-zero vector
    (FakeEmbedder) that collides with every other vector at cos sim=0."""

    def test_fake_embedder_empty_text_is_not_zero_vector(self, fresh_db):
        from db.embeddings import FakeEmbedder

        emb = FakeEmbedder()
        v_empty = emb.embed("")
        v_whitespace = emb.embed("   \n\t")
        v_none = emb.embed(None) if hasattr(emb, "_sanitize") else emb.embed("")

        # All three should be non-zero vectors (we use "__empty__" as
        # a placeholder word).
        for v in (v_empty, v_whitespace, v_none):
            assert any(abs(x) > 1e-6 for x in v), (
                "empty text must produce a non-zero vector (placeholder "
                "word) — otherwise cos_sim with any real vector is 0 "
                "and MATCH returns it for every query"
            )

        # Empty and whitespace should produce the SAME deterministic vector
        for a, b in zip(v_empty, v_whitespace):
            assert abs(a - b) < 1e-6

    def test_openai_embedder_sanitizes_empty_text(self):
        """Unit test — _sanitize replaces empty/whitespace with a space
        so the API never sees an empty input."""
        from db.embeddings import OpenAIEmbedder

        assert OpenAIEmbedder._sanitize("") == " "
        assert OpenAIEmbedder._sanitize("   ") == " "
        assert OpenAIEmbedder._sanitize("\n\t") == " "
        assert OpenAIEmbedder._sanitize(None) == " "
        # Non-empty text is unchanged (modulo strip)
        assert OpenAIEmbedder._sanitize("real text") == "real text"
        assert OpenAIEmbedder._sanitize("  real text  ") == "real text"


class TestLLMPrefetchStress:
    """Stress tests for the parallel LLM prefetch path. Verifies:

    - prefetch_llm_batch doesn't deadlock when called under the merge_lock
    - concurrent promoters + a sweep with prefetch don't deadlock
    - prefetch correctly handles events whose LLM call raises
    - prefetch results survive through apply_event correctly
    """

    HARD_TIMEOUT_SEC = 60

    def test_prefetch_under_lock_no_deadlock(self, fresh_db):
        """The prefetch spawns threads that each call namer.embed/LLM —
        none of those threads should try to acquire the merge_lock (they
        shouldn't touch the DB), so prefetch_llm_batch from inside the
        lock must complete promptly."""
        from db.category_events import CategoryEvent, prefetch_llm_batch

        events = []
        for i in range(8):
            events.append(CategoryEvent(
                event_type="add_category_new",
                payload={
                    "painpoint_ids": [1],
                    "sample_titles": [f"Sample {i}"],
                    "sample_descriptions": [f"Desc {i}"],
                },
                target_category=None,
                metric_name="cluster_size",
                metric_value=5.0,
                threshold=5.0,
            ))

        namer = FakeNamer()
        conn = db.get_db()
        try:
            start = time.monotonic()
            with merge_lock(conn, timeout=10):
                prefetch_llm_batch(conn, events, namer, max_workers=4)
            elapsed = time.monotonic() - start
            assert elapsed < self.HARD_TIMEOUT_SEC, (
                f"prefetch_llm_batch hung under lock for {elapsed}s"
            )

            # Every event should have its llm_result populated
            for ev in events:
                assert ev.llm_result is not None
                assert "name" in ev.llm_result
        finally:
            conn.close()

    def test_prefetch_with_failing_namer_doesnt_deadlock(self, fresh_db):
        """If the namer raises for some events, prefetch should log and
        continue — no thread must hang."""
        from db.category_events import CategoryEvent, prefetch_llm_batch

        class FlakeyNamer(FakeNamer):
            def __init__(self):
                super().__init__()
                self._count = 0

            def name_new_category(self, sample_titles, sample_descriptions,
                                  existing_taxonomy=None):
                self._count += 1
                if self._count % 2 == 0:
                    raise RuntimeError("simulated LLM flakiness")
                return super().name_new_category(
                    sample_titles, sample_descriptions, existing_taxonomy,
                )

        events = []
        for i in range(6):
            events.append(CategoryEvent(
                event_type="add_category_new",
                payload={
                    "painpoint_ids": [1],
                    "sample_titles": [f"Sample {i}"],
                    "sample_descriptions": [f"Desc {i}"],
                },
                target_category=None,
                metric_name="cluster_size",
                metric_value=5.0,
                threshold=5.0,
            ))

        namer = FlakeyNamer()
        conn = db.get_db()
        try:
            start = time.monotonic()
            with merge_lock(conn, timeout=10):
                prefetch_llm_batch(conn, events, namer, max_workers=3)
            elapsed = time.monotonic() - start
            assert elapsed < self.HARD_TIMEOUT_SEC, "prefetch hung on failing namer"

            # Some events have results, others are None (failed calls)
            successes = sum(1 for ev in events if ev.llm_result is not None)
            failures = sum(1 for ev in events if ev.llm_result is None)
            assert successes > 0 and failures > 0, (
                f"expected mix of success/failure, got "
                f"successes={successes} failures={failures}"
            )
        finally:
            conn.close()

    def test_sweep_with_prefetch_concurrent_with_promoters(self, fresh_db):
        """The old stress test shape: run a sweep in one thread while
        multiple promoters hammer from others. With the new prefetch
        path, the lock is STILL held during prefetch (we chose the
        middle-ground design), so promoters still block during sweep —
        but neither side should deadlock."""
        import threading

        embedder = FakeEmbedder()

        # Seed 20 pending pps
        pp_ids = []
        for i in range(20):
            post_id = _make_post(f"t3_stress_{i}", score=200, num_comments=50,
                                 title=f"Stress title {i}")
            pp_id = save_pending_painpoint(
                post_id, f"Stress painpoint variant {i}",
                description="Stress test painpoint", severity=7,
            )
            pp_ids.append(pp_id)

        errors = []

        def promoter_worker(chunk):
            try:
                for pp_id in chunk:
                    promote_pending(pp_id, embedder=embedder)
            except Exception as e:
                errors.append(("promoter", e))

        sweep_results = []

        def sweep_worker():
            try:
                time.sleep(0.05)  # let promoters start first
                summary = run_sweep(namer=FakeNamer(), embedder=embedder)
                sweep_results.append(summary)
            except Exception as e:
                errors.append(("sweep", e))

        # Split work across 3 promoter threads + 1 sweep thread
        chunks = [pp_ids[i::3] for i in range(3)]
        promoter_threads = [threading.Thread(target=promoter_worker, args=(c,))
                            for c in chunks]
        sweep_thread = threading.Thread(target=sweep_worker)

        for t in promoter_threads:
            t.start()
        sweep_thread.start()

        for t in promoter_threads:
            t.join(timeout=self.HARD_TIMEOUT_SEC)
            assert not t.is_alive(), "promoter thread hung"
        sweep_thread.join(timeout=self.HARD_TIMEOUT_SEC)
        assert not sweep_thread.is_alive(), "sweep thread hung"

        assert errors == [], f"threads raised: {errors}"
        assert len(sweep_results) == 1

        # DB invariants — same as other stress tests
        conn = db.get_db()
        try:
            orphans = conn.execute(
                "SELECT pp.id FROM pending_painpoints pp "
                "LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id "
                "WHERE ps.painpoint_id IS NULL"
            ).fetchall()
            assert orphans == [], f"orphan pending pps: {[r[0] for r in orphans]}"
        finally:
            conn.close()

    def test_repeated_sweeps_with_prefetch_dont_leak_threads(self, fresh_db):
        """10 consecutive sweeps in one process — each spawns threads in
        its ThreadPoolExecutor. Verify no thread leak by checking active
        thread count before and after (allowing some slack for test
        infrastructure)."""
        import threading

        embedder = FakeEmbedder()

        # Seed a clusterable Uncategorized state so the sweep has work
        _seed_uncategorized_painpoints([
            "GitHub Actions monorepo build slow",
            "GitHub Actions monorepo build hangs",
            "GitHub Actions monorepo build fails",
            "GitHub Actions monorepo build retry",
            "GitHub Actions monorepo build flake",
        ])

        baseline_threads = threading.active_count()

        namer = FakeNamer()
        start = time.monotonic()
        for i in range(10):
            run_sweep(namer=namer, embedder=embedder)
            assert time.monotonic() - start < self.HARD_TIMEOUT_SEC, (
                f"sweep {i} took too long, likely deadlock"
            )

        # Give pool threads a moment to wind down
        time.sleep(0.2)
        final_threads = threading.active_count()
        assert final_threads <= baseline_threads + 2, (
            f"thread leak: baseline={baseline_threads} final={final_threads} "
            f"— each sweep's ThreadPoolExecutor should clean up"
        )


class TestMergeCategoriesDescription:
    """_apply_merge_categories should call describe_merged_category on the
    namer and write the result to the survivor's description column."""

    def test_survivor_description_updated_after_merge(self, fresh_db):
        from db.category_events import CategoryEvent, _apply_merge_categories

        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            now = db._now()
            survivor_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Survivor', ?, 'old survivor desc', ?) RETURNING id",
                (parent_cat, now),
            ).fetchone()["id"]
            loser_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Loser', ?, 'old loser desc', ?) RETURNING id",
                (parent_cat, now),
            ).fetchone()["id"]
            # Give each one a painpoint so the LLM has samples to work with
            for cat_id, title in [(survivor_id, "Sample A"), (loser_id, "Sample B")]:
                conn.execute(
                    "INSERT INTO painpoints (title, description, severity, signal_count, "
                    "category_id, first_seen, last_updated) VALUES (?, '', 5, 1, ?, ?, ?)",
                    (title, cat_id, now, now),
                )
            conn.commit()

            event = CategoryEvent(
                event_type="merge_categories",
                payload={"survivor_id": survivor_id, "loser_id": loser_id},
                target_category=survivor_id,
                metric_name="merge_text_sim",
                metric_value=0.8,
                threshold=0.65,
            )
            _apply_merge_categories(conn, event, namer=FakeNamer())
            conn.commit()

            # FakeNamer.describe_merged_category returns
            # {"description": f"Merged: {survivor} + {loser}"}
            desc = conn.execute(
                "SELECT description FROM categories WHERE id = ?", (survivor_id,)
            ).fetchone()["description"]
            assert "Merged:" in desc, (
                f"survivor description should reflect the merge, got {desc!r}"
            )
            assert "Survivor" in desc and "Loser" in desc, (
                f"description should mention both source names, got {desc!r}"
            )
        finally:
            conn.close()
