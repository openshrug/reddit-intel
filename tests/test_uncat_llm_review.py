"""Tests for the LLM-review of Uncategorized singletons.

Covers: keep-path (no event), create-path (new category event with
pre-populated llm_result), priority ordering, max_reviews cap.
"""

import pytest

import db
from db.embeddings import (
    FakeEmbedder,
    get_category_anchor,
    store_painpoint_embedding,
)
from db.llm_naming import FakeNamer, UncatDecision
from db.category_events import (
    apply_with_test,
    propose_uncategorized_singleton_events,
)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    path = tmp_path / "uncat.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    yield path


def _seed_uncat_pp(conn, title, severity, signal_count, embedder):
    """Create a painpoint directly in Uncategorized."""
    uncat_id = conn.execute(
        "SELECT id FROM categories WHERE name = 'Uncategorized'"
    ).fetchone()["id"]
    now = db._now()
    conn.execute(
        "INSERT INTO painpoints (title, description, severity, signal_count, "
        "category_id, first_seen, last_updated) VALUES (?, '', ?, ?, ?, ?, ?)",
        (title, severity, signal_count, uncat_id, now, now),
    )
    pp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    store_painpoint_embedding(conn, pp_id, embedder.embed(title))
    return pp_id


class _CreateAllNamer(FakeNamer):
    """Namer whose decide_uncategorized always returns create, with a
    name derived from the painpoint title."""

    def decide_uncategorized(self, title, description, signal_count, severity,
                              existing_taxonomy):
        return UncatDecision(
            action="create",
            reason="fake: always create",
            name=f"ReviewCat-{title[:15]}",
            description=f"Auto description for: {title}",
            parent=None,
        )


class _KeepEverythingNamer(FakeNamer):
    """Same as the default FakeNamer (keep), but named explicitly for
    readability in these tests."""


class TestKeepPath:
    def test_no_events_when_llm_says_keep(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            _seed_uncat_pp(conn, "random niche complaint", 5, 1, embedder)
            _seed_uncat_pp(conn, "another lonely painpoint", 5, 1, embedder)
            conn.commit()

            namer = _KeepEverythingNamer()
            events = list(propose_uncategorized_singleton_events(
                conn, namer=namer, embedder=embedder,
            ))
            assert events == []
        finally:
            conn.close()


class TestCreatePath:
    def test_create_emits_add_category_new_with_prefetched_llm_result(
        self, fresh_db,
    ):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            pp_id = _seed_uncat_pp(
                conn, "unique developer pain xyz", 8, 3, embedder,
            )
            conn.commit()

            namer = _CreateAllNamer()
            events = list(propose_uncategorized_singleton_events(
                conn, namer=namer, embedder=embedder,
            ))
            assert len(events) == 1
            ev = events[0]
            assert ev.event_type == "add_category_new"
            assert ev.payload["painpoint_ids"] == [pp_id]
            assert ev.triggering_pp == pp_id
            # Pre-populated so _apply_add_category_new skips the name call.
            assert ev.llm_result is not None
            assert ev.llm_result["name"].startswith("ReviewCat-")
            assert ev.llm_result["description"]
        finally:
            conn.close()

    def test_apply_creates_category_with_anchor(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            pp_id = _seed_uncat_pp(
                conn, "brand new developer pain xyz", 8, 3, embedder,
            )
            conn.commit()

            namer = _CreateAllNamer()
            events = list(propose_uncategorized_singleton_events(
                conn, namer=namer, embedder=embedder,
            ))
            assert len(events) == 1
            assert apply_with_test(
                conn, events[0], namer, embedder=embedder,
            )

            # The painpoint should now be out of Uncategorized and the
            # new category should have an anchor.
            row = conn.execute(
                "SELECT category_id FROM painpoints WHERE id = ?", (pp_id,)
            ).fetchone()
            new_cat_id = row["category_id"]

            uncat_id = conn.execute(
                "SELECT id FROM categories WHERE name = 'Uncategorized'"
            ).fetchone()["id"]
            assert new_cat_id != uncat_id

            anchor = get_category_anchor(conn, new_cat_id)
            assert anchor is not None, (
                "newly created uncat-review category must have an anchor"
            )
        finally:
            conn.close()


class TestPriorityAndCap:
    def test_highest_signal_reviewed_first(self, fresh_db):
        # Dissimilar titles so FakeEmbedder's word-hash doesn't cluster
        # them together (which would exclude them from singleton review).
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            low = _seed_uncat_pp(
                conn, "kubernetes pod eviction autoscaler", 2, 1, embedder,
            )   # priority = 2
            high = _seed_uncat_pp(
                conn, "stripe webhook delivery failures", 9, 5, embedder,
            )   # priority = 45
            mid = _seed_uncat_pp(
                conn, "postgres bulk insert deadlock", 6, 2, embedder,
            )   # priority = 12
            conn.commit()

            seen = []

            class _RecordingNamer(FakeNamer):
                def decide_uncategorized(inner_self, title, description,
                                          signal_count, severity,
                                          existing_taxonomy):
                    seen.append(title)
                    return UncatDecision(action="keep", reason="fake")

            list(propose_uncategorized_singleton_events(
                conn, namer=_RecordingNamer(), embedder=embedder, max_reviews=1,
            ))
            assert seen == ["stripe webhook delivery failures"]
            _ = (low, mid)
        finally:
            conn.close()

    def test_max_reviews_bounds_llm_calls(self, fresh_db):
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            # Dissimilar titles (different domains) so they stay
            # singletons and hit the LLM-review path.
            titles = [
                "react server components hydration error",
                "docker layer cache invalidation",
                "terraform state file corruption",
                "graphql n+1 query performance",
                "elasticsearch index rebuild timeout",
            ]
            for t in titles:
                _seed_uncat_pp(conn, t, 5, 1, embedder)
            conn.commit()

            call_count = 0

            class _CountingNamer(FakeNamer):
                def decide_uncategorized(inner_self, *args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    return UncatDecision(action="keep", reason="fake")

            list(propose_uncategorized_singleton_events(
                conn, namer=_CountingNamer(), embedder=embedder, max_reviews=2,
            ))
            assert call_count == 2
        finally:
            conn.close()


class TestClusteredPainpointsExcluded:
    def test_clustered_pps_are_not_reviewed(self, fresh_db):
        """If the cluster pass would pick these up (size ≥ MIN_SUB_CLUSTER_SIZE
        at cosine ≥ MERGE_COSINE_THRESHOLD), the LLM shouldn't double-review
        them."""
        embedder = FakeEmbedder()
        conn = db.get_db()
        try:
            # 3 near-identical titles to force a cluster via FakeEmbedder's
            # word-hash similarity.
            for i in range(3):
                _seed_uncat_pp(
                    conn, f"kubernetes pods eviction autoscaler {i}",
                    7, 2, embedder,
                )
            conn.commit()

            seen_titles = []

            class _RecordingNamer(FakeNamer):
                def decide_uncategorized(inner_self, title, *args, **kwargs):
                    seen_titles.append(title)
                    return UncatDecision(action="keep", reason="fake")

            list(propose_uncategorized_singleton_events(
                conn, namer=_RecordingNamer(), embedder=embedder,
            ))
            # The 3 clustered painpoints all share enough vocabulary to be
            # in the same connected component at threshold 0.60 — they
            # should have been skipped, so nothing is reviewed.
            assert seen_titles == [], (
                f"clustered pps should not be LLM-reviewed, got {seen_titles}"
            )
        finally:
            conn.close()
