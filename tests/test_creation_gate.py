"""Tests for the hybrid-retrieval creation gate (workstream B).

Covers: the gate that stops _apply_add_category_split and
_apply_add_category_new from minting a category that's already
represented by an existing one.

Uses FakeEmbedder: its word-hash collision model means overlapping
words produce similar vectors, so we can construct near-duplicate
pairs by title overlap. FTS5 operates on real text so BM25 works
regardless of embedder choice.
"""

import pytest

import db
from db.category_events import (
    CategoryEvent,
    _apply_add_category_new,
    _apply_add_category_split,
)
from db.category_retrieval import (
    SIMILAR_CATEGORY_THRESHOLD,
    find_similar_category,
    sync_category_fts,
)
from db.embeddings import (
    FakeEmbedder,
    store_category_anchor,
    store_painpoint_embedding,
    update_category_embedding,
)
from db.llm_naming import FakeNamer


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    path = tmp_path / "gate.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    yield path


def _mk_cat(conn, name, description, parent_id, embedder):
    """Insert a category and fully wire it (anchor, vec, FTS5) so the
    gate's hybrid retrieval has something to find."""
    now = db._now()
    conn.execute(
        "INSERT INTO categories (name, parent_id, description, created_at) "
        "VALUES (?,?,?,?)",
        (name, parent_id, description, now),
    )
    cat_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    store_category_anchor(conn, cat_id, name, description, embedder)
    update_category_embedding(conn, cat_id)
    sync_category_fts(conn, cat_id, name, description)
    return cat_id


def _mk_pp(conn, title, category_id, embedder):
    now = db._now()
    conn.execute(
        "INSERT INTO painpoints "
        "(title, description, severity, signal_count, category_id, "
        "first_seen, last_updated) VALUES (?, '', 5, 1, ?, ?, ?)",
        (title, category_id, now, now),
    )
    pp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    store_painpoint_embedding(conn, pp_id, embedder.embed(title))
    return pp_id


class TestFindSimilarCategory:
    def test_keyword_overlap_surfaces_candidate(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            # Seed one obvious candidate and one decoy.
            video_id = _mk_cat(
                conn, "TestVideoHub",
                "video editing screen recording transcription hosting",
                None, embedder,
            )
            _mk_cat(
                conn, "TestNutritionHub",
                "meal planning macros vegan plant based eating cooking",
                None, embedder,
            )
            conn.commit()

            # Propose a near-duplicate of Video Tools.
            candidates = find_similar_category(
                conn,
                name="Video Editing and Clipping",
                description="video editing trimming clips for shorts",
                embedder=embedder,
            )

            assert candidates, "hybrid retrieval should surface candidates"
            top_id, top_cos, _rrf = candidates[0]
            assert top_id == video_id, (
                f"expected TestVideoHub to surface; got {top_id}"
            )
            # FakeEmbedder + shared "video editing" tokens → high cosine.
            assert top_cos >= 0.3, (
                f"expected non-trivial dense cos; got {top_cos:.3f}"
            )
        finally:
            conn.close()

    def test_uncategorized_always_excluded(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            candidates = find_similar_category(
                conn,
                name="anything",
                description="random description",
                embedder=embedder,
            )
            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            assert all(c[0] != uncat_id for c in candidates)
        finally:
            conn.close()


class TestSplitGateCollapsesDuplicates:
    def test_split_sub_matching_existing_routes_instead_of_creating(
        self, fresh_db,
    ):
        """A split pass that proposes a new sub which is essentially
        the source category under a new name must NOT create that row
        — it must reroute the sub's painpoints to the source (or
        whichever existing category scores above the threshold)."""
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            # Existing "TestVideoHub" under a parent root.
            root_id = _mk_cat(
                conn, "Creator Content Root",
                "creator content media production",
                None, embedder,
            )
            video_id = _mk_cat(
                conn, "TestVideoHub",
                "video editing video clipping video trimming video hosting video recording",
                root_id, embedder,
            )

            # Paintpoints currently living in Video Tools (the split source).
            pp1 = _mk_pp(conn, "video editing is slow", video_id, embedder)
            pp2 = _mk_pp(conn, "video clipping export fails", video_id, embedder)
            pp3 = _mk_pp(conn, "video trimming needs auto", video_id, embedder)
            conn.commit()

            # Monkey-patch the gate threshold to whatever FakeEmbedder
            # actually produces for this pair so the test is
            # deterministic across FakeEmbedder's hash distribution.
            candidates_probe = find_similar_category(
                conn,
                name="Video Editing and Clipping",
                description="video editing video clipping video trimming short video cutting",
                embedder=embedder,
            )
            assert candidates_probe
            top_cos = candidates_probe[0][1]
            # Tune threshold just below the observed cos so the gate fires.
            from db import category_events as ce
            monkey = top_cos - 0.01
            orig = ce.SIMILAR_CATEGORY_THRESHOLD
            ce.SIMILAR_CATEGORY_THRESHOLD = monkey
            try:
                event = CategoryEvent(
                    event_type="add_category_split",
                    payload={
                        "parent_category_id": root_id,
                        "source_category_id": video_id,
                        "subcategories": [
                            {
                                "name": "Video Editing and Clipping",
                                "description": (
                                    "video editing video clipping video trimming "
                                    "short video cutting"
                                ),
                                "painpoint_ids": [pp1, pp2, pp3],
                            },
                        ],
                    },
                )
                new_ids = _apply_add_category_split(
                    conn, event, FakeNamer(), embedder=embedder,
                )

                # Gate should have collapsed: zero new categories created.
                assert new_ids == [], (
                    f"gate failed: split created {new_ids} despite a "
                    f"candidate at cos={top_cos:.3f} >= threshold={monkey:.3f}"
                )

                # "Video Editing and Clipping" must NOT be a category row.
                exists = conn.execute(
                    "SELECT id FROM categories WHERE name = ?",
                    ("Video Editing and Clipping",),
                ).fetchone()
                assert exists is None, "gate created a row it should have skipped"

                # All painpoints should still be in a category (not dropped).
                rows = conn.execute(
                    "SELECT category_id FROM painpoints "
                    "WHERE id IN (?,?,?)", (pp1, pp2, pp3),
                ).fetchall()
                assert all(r["category_id"] is not None for r in rows)
            finally:
                ce.SIMILAR_CATEGORY_THRESHOLD = orig
        finally:
            conn.close()


class TestAddNewGateCollapsesDuplicates:
    def test_add_new_with_matching_existing_routes_instead_of_creating(
        self, fresh_db,
    ):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            # Existing target.
            crm_id = _mk_cat(
                conn, "CRM and Sales",
                "crm sales lead pipeline deal tracking outbound",
                None, embedder,
            )
            # Uncategorized member that the LLM would normally mint
            # a new "CRM Software Solutions" category for.
            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            pp1 = _mk_pp(conn, "crm pipeline stuck", uncat_id, embedder)
            pp2 = _mk_pp(conn, "crm deal tracking breaks", uncat_id, embedder)
            conn.commit()

            candidates_probe = find_similar_category(
                conn,
                name="CRM Software Solutions",
                description="crm sales crm pipeline crm deal tracking crm outbound tools",
                embedder=embedder,
            )
            assert candidates_probe
            top_cos = candidates_probe[0][1]

            from db import category_events as ce
            orig = ce.SIMILAR_CATEGORY_THRESHOLD
            ce.SIMILAR_CATEGORY_THRESHOLD = top_cos - 0.01
            try:
                event = CategoryEvent(
                    event_type="add_category_new",
                    payload={
                        "painpoint_ids": [pp1, pp2],
                        "sample_titles": ["crm pipeline stuck"],
                        "sample_descriptions": [""],
                    },
                    llm_result={
                        "name": "CRM Software Solutions",
                        "description": (
                            "crm sales crm pipeline crm deal tracking crm outbound tools"
                        ),
                        "parent": None,
                    },
                )
                _apply_add_category_new(
                    conn, event, FakeNamer(), embedder=embedder,
                )

                exists = conn.execute(
                    "SELECT id FROM categories WHERE name = ?",
                    ("CRM Software Solutions",),
                ).fetchone()
                assert exists is None, "gate created a duplicate"

                # Painpoints should have landed in the existing CRM bucket.
                rows = conn.execute(
                    "SELECT category_id FROM painpoints WHERE id IN (?,?)",
                    (pp1, pp2),
                ).fetchall()
                assert all(r["category_id"] == crm_id for r in rows)
            finally:
                ce.SIMILAR_CATEGORY_THRESHOLD = orig
        finally:
            conn.close()


class TestSplitReplantsUnderBetterParent:
    def test_cross_parent_replant_when_other_root_fits_better(self, fresh_db):
        """The LLM splits an AI-focused source and proposes a sub whose
        text is really about marketing. Hybrid retrieval should find
        `App Biz Root > Indie Marketing` as a better fit and replant
        the new sub under App Biz Root instead of under the AI root."""
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            ai_root = _mk_cat(
                conn, "TestAIRoot",
                "artificial intelligence llm model training inference",
                None, embedder,
            )
            # The split source lives under the AI root.
            ai_child_source = _mk_cat(
                conn, "TestAIChildSource",
                "ai model deployment inference serving performance",
                ai_root, embedder,
            )

            # Strong candidate under a DIFFERENT root whose text
            # overlaps the proposed sub's text.
            biz_root = _mk_cat(
                conn, "TestAppBizRoot",
                "app business app marketing growth indie app launch",
                None, embedder,
            )
            _mk_cat(
                conn, "TestIndieMarketing",
                "indie app marketing cold outbound growth launch strategy marketing campaign",
                biz_root, embedder,
            )

            # Paintpoints in the split source.
            pp1 = _mk_pp(
                conn, "indie app marketing cold outbound", ai_child_source, embedder,
            )
            pp2 = _mk_pp(
                conn, "marketing campaign strategy indie launch", ai_child_source, embedder,
            )
            pp3 = _mk_pp(
                conn, "growth indie marketing outbound strategy", ai_child_source, embedder,
            )
            conn.commit()

            from db import category_events as ce
            # Threshold constants are bound as local names in
            # category_events at import time, so monkey-patch them
            # there, not on the category_retrieval module.
            orig_sim = ce.SIMILAR_CATEGORY_THRESHOLD
            ce.SIMILAR_CATEGORY_THRESHOLD = 0.95
            orig_min = ce.CROSS_PARENT_REPARENT_MIN_COS
            orig_margin = ce.CROSS_PARENT_REPARENT_MARGIN
            ce.CROSS_PARENT_REPARENT_MIN_COS = 0.0
            ce.CROSS_PARENT_REPARENT_MARGIN = 0.0
            try:
                event = CategoryEvent(
                    event_type="add_category_split",
                    payload={
                        "parent_category_id": ai_root,
                        "source_category_id": ai_child_source,
                        "subcategories": [
                            {
                                "name": "Marketing and Growth Strategy",
                                "description": (
                                    "indie app marketing cold outbound growth launch "
                                    "strategy marketing campaign"
                                ),
                                "painpoint_ids": [pp1, pp2, pp3],
                            },
                        ],
                    },
                )
                new_ids = _apply_add_category_split(
                    conn, event, FakeNamer(), embedder=embedder,
                )
                assert len(new_ids) == 1, (
                    f"expected 1 new sub (replanted), got {new_ids}"
                )
                # The new sub must be parented under biz_root, NOT ai_root.
                row = conn.execute(
                    "SELECT parent_id FROM categories WHERE id = ?", (new_ids[0],),
                ).fetchone()
                assert row["parent_id"] == biz_root, (
                    f"replant failed: new sub parent={row['parent_id']}, "
                    f"expected biz_root={biz_root}"
                )
            finally:
                ce.SIMILAR_CATEGORY_THRESHOLD = orig_sim
                ce.CROSS_PARENT_REPARENT_MIN_COS = orig_min
                ce.CROSS_PARENT_REPARENT_MARGIN = orig_margin
        finally:
            conn.close()


class TestGateLowerThresholdCreatesNormally:
    def test_no_candidate_above_threshold_creates_new(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            # Only one distant decoy exists.
            _mk_cat(
                conn, "TestNutritionHub",
                "meal planning macros vegan plant based eating",
                None, embedder,
            )
            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            pp1 = _mk_pp(
                conn, "kubernetes pod eviction autoscaler", uncat_id, embedder,
            )
            conn.commit()

            event = CategoryEvent(
                event_type="add_category_new",
                payload={
                    "painpoint_ids": [pp1],
                    "sample_titles": ["kubernetes pod eviction autoscaler"],
                    "sample_descriptions": [""],
                },
                llm_result={
                    "name": "Kubernetes Cluster Autoscaling",
                    "description": (
                        "kubernetes pods eviction autoscaler node management "
                        "cluster scaling"
                    ),
                    "parent": None,
                },
            )
            _apply_add_category_new(
                conn, event, FakeNamer(), embedder=embedder,
            )

            # No similar existing category → gate stays silent, new row exists.
            exists = conn.execute(
                "SELECT id FROM categories WHERE name = ?",
                ("Kubernetes Cluster Autoscaling",),
            ).fetchone()
            assert exists is not None
        finally:
            conn.close()
