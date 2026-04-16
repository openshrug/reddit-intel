"""Tests for the category-anchor embedding approach.

Diagnosis: in the live run we saw "Generative Media" absorbing "Cold
Email Deliverability" because pure mean-of-members centroids drift
toward off-topic joiners (positive feedback loop).

Fix: store a static anchor embedding (of `name + description`) per
category and blend it with the member mean when writing `category_vec`.
These tests demonstrate the anchor's resistance to hijacking and
verify the delete/merge/split wiring removes/refreshes anchors correctly.
"""

import math

import pytest

import db
from db.embeddings import (
    ANCHOR_WEIGHT,
    EMBEDDING_DIM,
    FakeEmbedder,
    _anchor_text,
    delete_category_anchor,
    find_best_category,
    get_category_anchor,
    store_category_anchor,
    store_painpoint_embedding,
    update_category_embedding,
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    path = tmp_path / "anchor.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    yield path


def _cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _make_cat(conn, name, description, parent_id=None):
    conn.execute(
        "INSERT INTO categories (name, parent_id, description, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (name, parent_id, description),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _add_pp(conn, title, category_id, embedder):
    now = db._now()
    conn.execute(
        "INSERT INTO painpoints (title, description, severity, signal_count, "
        "category_id, first_seen, last_updated) VALUES (?, '', 5, 1, ?, ?, ?)",
        (title, category_id, now, now),
    )
    pp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    store_painpoint_embedding(conn, pp_id, embedder.embed(title))
    return pp_id


class TestAnchorStorage:
    def test_store_and_get_anchor_roundtrip(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            cid = _make_cat(conn, "Test Cat", "keyword rich description")
            store_category_anchor(conn, cid, "Test Cat", "keyword rich description", embedder)
            emb = get_category_anchor(conn, cid)
            assert emb is not None
            assert len(emb) == EMBEDDING_DIM
            # Matches what the embedder would produce for the same text.
            expected = embedder.embed(_anchor_text("Test Cat", "keyword rich description"))
            assert _cosine(emb, expected) == pytest.approx(1.0, abs=1e-6)
        finally:
            conn.close()

    def test_missing_anchor_returns_none(self, fresh_db):
        conn = db.get_db()
        try:
            cid = _make_cat(conn, "No Anchor", "whatever")
            assert get_category_anchor(conn, cid) is None
        finally:
            conn.close()

    def test_delete_anchor_removes_row(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            cid = _make_cat(conn, "Will Delete", "desc")
            store_category_anchor(conn, cid, "Will Delete", "desc", embedder)
            assert get_category_anchor(conn, cid) is not None
            delete_category_anchor(conn, cid)
            assert get_category_anchor(conn, cid) is None
        finally:
            conn.close()


class TestBlendedCentroid:
    def test_no_anchor_no_members_removes_vec_row(self, fresh_db):
        conn = db.get_db()
        try:
            cid = _make_cat(conn, "Empty", "empty")
            update_category_embedding(conn, cid)
            row = conn.execute(
                "SELECT rowid FROM category_vec WHERE rowid = ?", (cid,)
            ).fetchone()
            assert row is None
        finally:
            conn.close()

    def test_anchor_only_no_members_uses_anchor(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            cid = _make_cat(conn, "Anchor Only", "kubernetes pods eviction")
            store_category_anchor(
                conn, cid, "Anchor Only", "kubernetes pods eviction", embedder,
            )
            update_category_embedding(conn, cid)
            emb = _get_category_vec(conn, cid)
            anchor = get_category_anchor(conn, cid)
            assert emb is not None
            assert _cosine(emb, anchor) == pytest.approx(1.0, abs=1e-6)
        finally:
            conn.close()

    def test_legacy_no_anchor_with_members_uses_pure_mean(self, fresh_db):
        """Preserves old behaviour for DBs that never had an anchor."""
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            cid = _make_cat(conn, "Legacy", "doesnt matter")
            # Deliberately NOT storing an anchor.
            _add_pp(conn, "kubernetes pods eviction autoscaler", cid, embedder)
            _add_pp(conn, "kubernetes pod eviction during autoscale", cid, embedder)
            update_category_embedding(conn, cid)
            emb = _get_category_vec(conn, cid)
            assert emb is not None
            # Pure mean-of-members → highly similar to each member.
            member_emb = embedder.embed("kubernetes pods eviction autoscaler")
            assert _cosine(emb, member_emb) > 0.5
        finally:
            conn.close()

    def test_blend_is_anchor_dominant(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            cid = _make_cat(conn, "Blended", "kubernetes autoscaling pod eviction")
            store_category_anchor(
                conn, cid, "Blended", "kubernetes autoscaling pod eviction", embedder,
            )
            _add_pp(conn, "postgres bulk insert deadlock", cid, embedder)
            update_category_embedding(conn, cid)
            emb = _get_category_vec(conn, cid)
            anchor = get_category_anchor(conn, cid)
            member = embedder.embed("postgres bulk insert deadlock")
            sim_to_anchor = _cosine(emb, anchor)
            sim_to_member = _cosine(emb, member)
            # ANCHOR_WEIGHT=0.7: the blended vector must lean toward the anchor.
            assert sim_to_anchor > sim_to_member
            assert sim_to_anchor > ANCHOR_WEIGHT - 0.1  # qualitative dominance check
        finally:
            conn.close()


class TestHijackResistance:
    """The core motivation: a bad member must NOT pull the centroid far
    from the declared identity. Demonstrated by contrasting against the
    old pure-mean behaviour."""

    def test_off_topic_member_cannot_dominate_anchor(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            name = "HijackResistanceCat"
            desc = "video generation image generation AI creative visual content media"
            cid = _make_cat(conn, name, desc)
            store_category_anchor(conn, cid, name, desc, embedder)

            # One on-topic member, then a pile of off-topic ones (the
            # observed hijack pattern: N cold-email painpoints creeping
            # into a media category).
            _add_pp(conn, "stable diffusion image generation prompts", cid, embedder)
            for i in range(10):
                _add_pp(
                    conn,
                    f"cold email deliverability bounce rate leadgen campaign {i}",
                    cid,
                    embedder,
                )

            update_category_embedding(conn, cid)
            blended = _get_category_vec(conn, cid)
            anchor = get_category_anchor(conn, cid)
            off_topic = embedder.embed(
                "cold email deliverability bounce rate leadgen campaign 0"
            )

            # With the blend, the centroid still sits far closer to the
            # declared identity than to the hijackers.
            sim_anchor = _cosine(blended, anchor)
            sim_hijacker = _cosine(blended, off_topic)
            assert sim_anchor > sim_hijacker + 0.2, (
                f"blend failed hijack-resistance: anchor sim={sim_anchor:.3f}, "
                f"hijacker sim={sim_hijacker:.3f}"
            )
        finally:
            conn.close()

    def test_pure_mean_would_hijack_without_anchor(self, fresh_db):
        """Negative control: show that the OLD behaviour (no anchor)
        does drift — this is what we're protecting against."""
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            cid = _make_cat(conn, "Bare", "no anchor here")
            _add_pp(conn, "stable diffusion image generation prompts", cid, embedder)
            for i in range(10):
                _add_pp(
                    conn,
                    f"cold email deliverability bounce rate leadgen campaign {i}",
                    cid,
                    embedder,
                )
            # No anchor → pure mean-of-members path.
            update_category_embedding(conn, cid)
            emb = _get_category_vec(conn, cid)
            on_topic = embedder.embed("stable diffusion image generation prompts")
            off_topic = embedder.embed(
                "cold email deliverability bounce rate leadgen campaign 0"
            )
            # Without the anchor, the centroid is dragged toward the hijackers
            # because they outnumber the legitimate member.
            assert _cosine(emb, off_topic) > _cosine(emb, on_topic)
        finally:
            conn.close()


class TestSweepWiringPreservesAnchors:
    """Verify the sweep appliers call the anchor helpers as expected."""

    def test_add_category_new_writes_anchor(self, fresh_db):
        from db.category_events import CategoryEvent, apply_with_test
        from db.llm_naming import FakeNamer

        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            # Seed three painpoints in Uncategorized so the new category has members.
            pp_ids = []
            for i in range(3):
                pp_ids.append(
                    _add_pp(conn, f"some new pain variant {i}", uncat_id, embedder)
                )
            conn.commit()

            event = CategoryEvent(
                event_type="add_category_new",
                payload={
                    "painpoint_ids": pp_ids,
                    "sample_titles": [f"some new pain variant {i}" for i in range(3)],
                    "sample_descriptions": ["" for _ in range(3)],
                },
                target_category=uncat_id,
                metric_name="cluster_size",
                metric_value=3.0,
                threshold=3.0,
            )

            namer = FakeNamer()
            assert apply_with_test(conn, event, namer, embedder=embedder)

            new_cat_id = conn.execute(
                "SELECT id FROM categories WHERE name LIKE 'AutoCat-%' ORDER BY id DESC LIMIT 1"
            ).fetchone()["id"]
            anchor = get_category_anchor(conn, new_cat_id)
            assert anchor is not None, "new category must have an anchor after apply"
        finally:
            conn.close()

    def test_delete_category_removes_anchor(self, fresh_db):
        from db.category_events import CategoryEvent, apply_with_test
        from db.llm_naming import FakeNamer

        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            parent = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
            cid = _make_cat(conn, "Condemned", "about to die", parent_id=parent)
            store_category_anchor(conn, cid, "Condemned", "about to die", embedder)
            conn.commit()

            assert get_category_anchor(conn, cid) is not None

            event = CategoryEvent(
                event_type="delete_category",
                payload={
                    "category_id": cid,
                    "category_name": "Condemned",
                    "parent_id": parent,
                },
                target_category=cid,
            )
            assert apply_with_test(conn, event, FakeNamer(), embedder=embedder)

            assert get_category_anchor(conn, cid) is None
        finally:
            conn.close()


def _get_category_vec(conn, cat_id):
    """Helper: unpack category_vec into a list of floats."""
    import struct
    row = conn.execute(
        "SELECT embedding FROM category_vec WHERE rowid = ?", (cat_id,)
    ).fetchone()
    if row is None:
        return None
    blob = row[0]
    if len(blob) != EMBEDDING_DIM * 4:
        return None
    return list(struct.unpack(f"{EMBEDDING_DIM}f", blob))


def test_find_best_category_still_routes_correctly(fresh_db):
    """After the anchor rework, find_best_category still lands painpoints
    in the most semantically-appropriate bucket."""
    embedder = FakeEmbedder()
    conn = db.get_db()
    try:
        parent = conn.execute(
            "SELECT id FROM categories WHERE parent_id IS NULL "
            "AND name != 'Uncategorized' LIMIT 1"
        ).fetchone()["id"]
        k8s_id = _make_cat(
            conn, "K8s", "kubernetes pods eviction autoscaler", parent_id=parent,
        )
        pg_id = _make_cat(
            conn, "Postgres", "postgres bulk insert deadlock lock", parent_id=parent,
        )
        store_category_anchor(
            conn, k8s_id, "K8s", "kubernetes pods eviction autoscaler", embedder,
        )
        store_category_anchor(
            conn, pg_id, "Postgres", "postgres bulk insert deadlock lock", embedder,
        )
        update_category_embedding(conn, k8s_id)
        update_category_embedding(conn, pg_id)
        conn.commit()

        # FakeEmbedder is a word-hash → spiky vectors. Use heavy word
        # overlap with the anchor text so similarity clears
        # CATEGORY_COSINE_THRESHOLD. Production embeddings (OpenAI) are
        # much smoother; this test is about routing *correctness*, not
        # the absolute threshold.
        k8s_query = embedder.embed("kubernetes pods eviction autoscaler k8s issue")
        pg_query = embedder.embed("postgres bulk insert deadlock lock timeout")

        best_k8s = find_best_category(conn, k8s_query, embedder=embedder)
        best_pg = find_best_category(conn, pg_query, embedder=embedder)

        assert best_k8s == k8s_id
        assert best_pg == pg_id
    finally:
        conn.close()
