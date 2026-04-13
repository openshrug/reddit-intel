"""Integration tests for the db/ package.

Runs against a temp SQLite file — never touches trends.db.
Tests are sequential: each builds on state created by earlier tests.
Run with:  pytest tests/test_db_integration.py -v
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

import db
from db.posts import upsert_post, upsert_comment, get_post_by_reddit_id, get_comments_for_post
from db.painpoints import (
    save_pending_painpoint,
    save_pending_painpoints_batch,
    get_unmerged_pending,
)


# The Layer A multi-match merging helpers (`merge_pending_into_painpoint`,
# `link_painpoint_source`, `upsert_painpoint`) were removed — the live
# pipeline promotes per-pending via embedding cosine similarity instead.
# These tiny SQL helpers reproduce just enough of their behaviour to keep
# the FK / query / provenance tests below meaningful without reviving the
# dead API surface.
def _test_upsert_painpoint(conn, title, *, description="", severity=5, category_id=None,
                           signal_count=1):
    now = db._now()
    existing = conn.execute(
        "SELECT id FROM painpoints WHERE title = ?", (title,),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE painpoints SET signal_count = signal_count + ?, "
            "last_updated = ? WHERE id = ?",
            (signal_count, now, existing["id"]),
        )
        return existing["id"]
    conn.execute(
        "INSERT INTO painpoints (title, description, severity, signal_count, "
        "category_id, first_seen, last_updated) VALUES (?,?,?,?,?,?,?)",
        (title, description, severity, signal_count, category_id, now, now),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _test_link_source(conn, painpoint_id, pending_id):
    try:
        conn.execute(
            "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
            "VALUES (?, ?)", (painpoint_id, pending_id),
        )
    except sqlite3.IntegrityError:
        pass


def _test_merge_pending(pending_ids, *, title, description="", severity=5,
                        category_id=None):
    conn = db.get_db()
    try:
        pid = _test_upsert_painpoint(
            conn, title, description=description, severity=severity,
            category_id=category_id, signal_count=len(pending_ids),
        )
        for pp in pending_ids:
            _test_link_source(conn, pid, pp)
        conn.commit()
        return pid
    finally:
        conn.close()
from db.categories import (
    get_category_by_name,
    get_category_id_by_name,
    get_category_list_flat,
    get_all_categories,
    get_root_categories,
)
from db.queries import (
    get_top_painpoints,
    get_painpoint_evidence,
    get_painpoints_by_category,
    get_painpoints_by_subreddit,
    get_subreddit_summary,
    get_stats,
    run_sql,
)

# ---------------------------------------------------------------------------
# Shared state across ordered tests
# ---------------------------------------------------------------------------

_state = {}

TABLES = ["posts", "comments", "categories", "pending_painpoints",
          "painpoints", "painpoint_sources"]

SAMPLE_POST = {
    "name": "t3_abc123",
    "subreddit": "localllama",
    "title": "Llama 3 70B stalls at 40k context on M3 Max",
    "selftext": "Has anyone else seen this? The model just hangs...",
    "url": "https://reddit.com/r/localllama/comments/abc123/...",
    "author": "testuser",
    "score": 142,
    "upvote_ratio": 0.92,
    "num_comments": 37,
    "permalink": "/r/localllama/comments/abc123/llama_3_70b_stalls/",
    "created_utc": 1712000000.0,
    "is_self": True,
    "stickied": False,
}

SAMPLE_POST_2 = {
    "name": "t3_def456",
    "subreddit": "devops",
    "title": "GitHub Actions keeps timing out on large mono-repos",
    "selftext": "Our CI takes 45 min now...",
    "url": "https://reddit.com/r/devops/comments/def456/...",
    "author": "cicd_user",
    "score": 88,
    "upvote_ratio": 0.89,
    "num_comments": 19,
    "permalink": "/r/devops/comments/def456/actions_timeout/",
    "created_utc": 1712100000.0,
    "is_self": True,
    "stickied": False,
}

COMMENT_1 = {
    "name": "t1_comment1",
    "parent_id": "t3_abc123",
    "author": "commenter1",
    "body": "I've seen the same thing, it's a memory issue",
    "score": 45,
    "controversiality": 0,
    "permalink": "/r/localllama/comments/abc123/.../comment1/",
    "created_utc": 1712001000.0,
    "depth": 0,
}

COMMENT_2 = {
    "name": "t1_comment2",
    "parent_id": "t1_comment1",
    "author": "commenter2",
    "body": "Try reducing the batch size to 1",
    "score": 12,
    "controversiality": 0,
    "permalink": "/r/localllama/comments/abc123/.../comment2/",
    "created_utc": 1712002000.0,
    "depth": 1,
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def temp_db(tmp_path_factory):
    """Point the db package at a temp file and initialise it."""
    tmp = tmp_path_factory.mktemp("db") / "test.db"
    db.DB_PATH = tmp
    db.init_db()
    yield tmp


# ===================================================================
# Test 1 — Schema init + taxonomy seeding
# ===================================================================


class TestSchemaInit:

    def test_all_tables_exist(self):
        conn = db.get_db()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        conn.close()
        table_names = {r["name"] for r in rows}
        for t in TABLES:
            assert t in table_names, f"Missing table: {t}"

    def test_categories_seeded(self):
        conn = db.get_db()
        count = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        conn.close()
        assert count > 0, "categories table should be seeded"

    def test_root_categories_have_null_parent(self):
        roots = get_root_categories()
        assert len(roots) > 0
        root_names = {r["name"] for r in roots}
        assert "AI/ML" in root_names
        assert "Developer Tools" in root_names
        for r in roots:
            assert r["parent_id"] is None

    def test_child_categories_point_to_root(self):
        cat = get_category_by_name("AI Coding Tools")
        assert cat is not None
        parent = get_category_by_name("AI/ML")
        assert cat["parent_id"] == parent["id"]

    def test_init_db_idempotent(self):
        conn = db.get_db()
        count_before = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        conn.close()

        db.init_db()

        conn = db.get_db()
        count_after = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        conn.close()
        assert count_before == count_after, "init_db called twice must not duplicate categories"


# ===================================================================
# Test 2 — Post deduplication
# ===================================================================


class TestPostDedup:

    def test_insert_and_dedup(self):
        id1 = upsert_post(SAMPLE_POST)
        id2 = upsert_post(SAMPLE_POST)
        assert id1 == id2, "Same reddit_id must return same internal id"
        _state["post_id"] = id1

    def test_single_row_in_db(self):
        conn = db.get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE reddit_id = 't3_abc123'"
        ).fetchone()[0]
        conn.close()
        assert count == 1

    def test_permalink_prefixed(self):
        post = get_post_by_reddit_id("t3_abc123")
        assert post["permalink"].startswith("https://reddit.com")

    def test_fields_stored_correctly(self):
        post = get_post_by_reddit_id("t3_abc123")
        assert post["subreddit"] == "localllama"
        assert post["title"] == SAMPLE_POST["title"]
        assert post["score"] == 142
        assert post["upvote_ratio"] == 0.92
        assert post["is_self"] == 1
        assert post["fetched_at"] is not None


# ===================================================================
# Test 3 — Comment deduplication and FK integrity
# ===================================================================


class TestCommentDedup:

    def test_insert_comments_and_dedup(self):
        post_id = _state["post_id"]
        cid1a = upsert_comment(post_id, COMMENT_1)
        cid2 = upsert_comment(post_id, COMMENT_2)
        cid1b = upsert_comment(post_id, COMMENT_1)

        assert cid1a == cid1b, "Same reddit_id must return same comment id"
        assert cid1a != cid2
        _state["cid1"] = cid1a
        _state["cid2"] = cid2

    def test_comment_count(self):
        conn = db.get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM comments WHERE post_id = ?",
            (_state["post_id"],),
        ).fetchone()[0]
        conn.close()
        assert count == 2

    def test_reply_chain(self):
        conn = db.get_db()
        row = conn.execute(
            "SELECT parent_reddit_id FROM comments WHERE reddit_id = 't1_comment2'"
        ).fetchone()
        conn.close()
        assert row["parent_reddit_id"] == "t1_comment1"

    def test_get_comments_for_post_ordered_by_score(self):
        comments = get_comments_for_post(_state["post_id"])
        assert len(comments) == 2
        assert comments[0]["score"] >= comments[1]["score"]

    def test_comment_permalink_prefixed(self):
        conn = db.get_db()
        row = conn.execute(
            "SELECT permalink FROM comments WHERE reddit_id = 't1_comment1'"
        ).fetchone()
        conn.close()
        assert row["permalink"].startswith("https://reddit.com")


# ===================================================================
# Test 4 — Pending painpoint creation with category resolution
# ===================================================================


class TestPendingPainpoints:

    def test_valid_category(self):
        pp1 = save_pending_painpoint(
            _state["post_id"],
            "LLM context window hangs on Apple Silicon",
            comment_id=_state["cid1"],
            category_name="LLM Infrastructure",
            description="Users report 70B models hanging at high context lengths on M-series Macs",
            quoted_text="The model just hangs after ~40k tokens",
            severity=8,
        )
        _state["pp1"] = pp1

        conn = db.get_db()
        row = conn.execute(
            "SELECT category_id FROM pending_painpoints WHERE id = ?", (pp1,)
        ).fetchone()
        conn.close()

        llm_infra_id = get_category_id_by_name("LLM Infrastructure")
        assert row["category_id"] == llm_infra_id
        _state["llm_infra_cat_id"] = llm_infra_id

    def test_unknown_category_resolves_to_null(self):
        pp2 = save_pending_painpoint(
            _state["post_id"],
            "Batch size config unclear",
            category_name="Nonexistent Category",
            description="Users don't know how to configure batch size",
            severity=4,
        )
        _state["pp2"] = pp2

        conn = db.get_db()
        row = conn.execute(
            "SELECT category_id FROM pending_painpoints WHERE id = ?", (pp2,)
        ).fetchone()
        conn.close()
        assert row["category_id"] is None

    def test_no_category_resolves_to_null(self):
        pp3 = save_pending_painpoint(
            _state["post_id"],
            "General frustration with inference speed",
            severity=5,
        )
        _state["pp3"] = pp3

        conn = db.get_db()
        row = conn.execute(
            "SELECT category_id FROM pending_painpoints WHERE id = ?", (pp3,)
        ).fetchone()
        conn.close()
        assert row["category_id"] is None

    def test_extracted_at_populated(self):
        conn = db.get_db()
        rows = conn.execute(
            "SELECT extracted_at FROM pending_painpoints WHERE id IN (?,?,?)",
            (_state["pp1"], _state["pp2"], _state["pp3"]),
        ).fetchall()
        conn.close()
        for r in rows:
            assert r["extracted_at"] is not None

    def test_severity_clamping_low(self):
        pp = save_pending_painpoint(
            _state["post_id"], "Severity floor test", severity=-5
        )
        conn = db.get_db()
        row = conn.execute(
            "SELECT severity FROM pending_painpoints WHERE id = ?", (pp,)
        ).fetchone()
        conn.close()
        assert row["severity"] == 1

    def test_severity_zero_clamped_to_one(self):
        """severity=0 is out-of-range; the clamp floors it to 1 (no silent
        substitution to a made-up default)."""
        pp = save_pending_painpoint(
            _state["post_id"], "Severity zero test", severity=0
        )
        conn = db.get_db()
        row = conn.execute(
            "SELECT severity FROM pending_painpoints WHERE id = ?", (pp,)
        ).fetchone()
        conn.close()
        assert row["severity"] == 1

    def test_severity_none_raises(self):
        """severity=None must raise rather than silently default to 5."""
        with pytest.raises(ValueError, match="severity"):
            save_pending_painpoint(
                _state["post_id"], "Severity none test", severity=None
            )

    def test_severity_clamping_high(self):
        pp = save_pending_painpoint(
            _state["post_id"], "Severity ceiling test", severity=15
        )
        conn = db.get_db()
        row = conn.execute(
            "SELECT severity FROM pending_painpoints WHERE id = ?", (pp,)
        ).fetchone()
        conn.close()
        assert row["severity"] == 10


# ===================================================================
# Test 5 — Batch pending painpoint insertion
# ===================================================================


class TestBatchPending:

    def test_batch_insert(self):
        post2_id = upsert_post(SAMPLE_POST_2)
        _state["post2_id"] = post2_id

        items = [
            {"post_id": post2_id, "title": "Pain A", "severity": 7,
             "category_name": "CI/CD & DevOps"},
            {"post_id": post2_id, "title": "Pain B", "severity": 3},
            {"post_id": post2_id, "title": "Pain C", "severity": 6,
             "quoted_text": "this is so broken"},
        ]
        ids = save_pending_painpoints_batch(items)
        _state["batch_ids"] = ids

        assert len(ids) == 3
        assert len(set(ids)) == 3, "All ids must be distinct"

    def test_batch_rows_exist(self):
        conn = db.get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM pending_painpoints WHERE id IN (?,?,?)",
            tuple(_state["batch_ids"]),
        ).fetchone()[0]
        conn.close()
        assert count == 3

    def test_batch_category_resolved(self):
        pain_a_id = _state["batch_ids"][0]
        conn = db.get_db()
        row = conn.execute(
            "SELECT category_id FROM pending_painpoints WHERE id = ?", (pain_a_id,)
        ).fetchone()
        conn.close()
        cicd_id = get_category_id_by_name("CI/CD & DevOps")
        assert row["category_id"] == cicd_id

    def test_batch_same_timestamp(self):
        conn = db.get_db()
        rows = conn.execute(
            "SELECT DISTINCT extracted_at FROM pending_painpoints WHERE id IN (?,?,?)",
            tuple(_state["batch_ids"]),
        ).fetchall()
        conn.close()
        assert len(rows) == 1, "All batch items should share one extracted_at"


# ===================================================================
# Test 6 — Merge workflow: pending → merged painpoint
# ===================================================================


class TestMergeWorkflow:

    def test_merge_creates_painpoint_and_sources(self):
        merged_id = _test_merge_pending(
            pending_ids=[_state["pp1"], _state["pp3"]],
            title="LLM inference hangs on Apple Silicon",
            description="Multiple reports of 70B models stalling at high context on M-series Macs",
            severity=8,
            category_id=_state["llm_infra_cat_id"],
        )
        _state["merged_id"] = merged_id

        conn = db.get_db()
        row = conn.execute(
            "SELECT * FROM painpoints WHERE id = ?", (merged_id,)
        ).fetchone()
        assert row["signal_count"] == 2
        assert row["severity"] == 8
        assert row["title"] == "LLM inference hangs on Apple Silicon"

        sources = conn.execute(
            "SELECT pending_painpoint_id FROM painpoint_sources WHERE painpoint_id = ?",
            (merged_id,),
        ).fetchall()
        conn.close()

        source_ids = {r["pending_painpoint_id"] for r in sources}
        assert source_ids == {_state["pp1"], _state["pp3"]}

    def test_duplicate_source_link_is_idempotent(self):
        """Re-linking the same pending painpoint must not create duplicate rows."""
        conn = db.get_db()
        try:
            _test_link_source(conn, _state["merged_id"], _state["pp1"])
            _test_link_source(conn, _state["merged_id"], _state["pp3"])
            conn.commit()
        finally:
            conn.close()

        conn = db.get_db()
        sources = conn.execute(
            "SELECT COUNT(*) FROM painpoint_sources WHERE painpoint_id = ?",
            (_state["merged_id"],),
        ).fetchone()[0]
        conn.close()
        assert sources == 2, "Duplicate source links must not be created"


# ===================================================================
# Test 7 — Merge into existing painpoint (signal_count bump)
# ===================================================================


class TestMergeSignalBump:

    def test_signal_count_increments(self):
        conn = db.get_db()
        before = conn.execute(
            "SELECT last_updated FROM painpoints WHERE id = ?",
            (_state["merged_id"],),
        ).fetchone()["last_updated"]
        conn.close()

        new_pp = save_pending_painpoint(
            _state["post2_id"],
            "Apple Silicon inference problems",
            category_name="LLM Infrastructure",
            severity=7,
        )
        _state["new_pp"] = new_pp

        merged_id2 = _test_merge_pending(
            pending_ids=[new_pp],
            title="LLM inference hangs on Apple Silicon",
            severity=7,
        )
        assert merged_id2 == _state["merged_id"]

        conn = db.get_db()
        row = conn.execute(
            "SELECT signal_count, last_updated FROM painpoints WHERE id = ?",
            (_state["merged_id"],),
        ).fetchone()
        conn.close()

        assert row["signal_count"] == 3
        assert row["last_updated"] >= before

    def test_three_sources_now(self):
        conn = db.get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM painpoint_sources WHERE painpoint_id = ?",
            (_state["merged_id"],),
        ).fetchone()[0]
        conn.close()
        assert count == 3


# ===================================================================
# Test 8 — Full provenance chain query
# ===================================================================


class TestProvenanceChain:

    def test_evidence_count(self):
        evidence = get_painpoint_evidence(_state["merged_id"])
        assert len(evidence) == 3

    def test_evidence_fields(self):
        evidence = get_painpoint_evidence(_state["merged_id"])
        required = {"pending_title", "quoted_text", "subreddit",
                     "post_title", "post_permalink", "category_name"}
        for row in evidence:
            for field in required:
                assert field in row, f"Missing field: {field}"

    def test_evidence_has_comment(self):
        evidence = get_painpoint_evidence(_state["merged_id"])
        with_comment = [r for r in evidence if r["comment_body"] is not None]
        without_comment = [r for r in evidence if r["comment_body"] is None]
        assert len(with_comment) >= 1, "At least one evidence row should have a comment"
        assert len(without_comment) >= 1, "At least one evidence row should be post-level"

    def test_evidence_ordered_by_post_score(self):
        evidence = get_painpoint_evidence(_state["merged_id"])
        scores = [r["post_score"] for r in evidence]
        assert scores == sorted(scores, reverse=True)


# ===================================================================
# Test 9 — Agent query: top painpoints
# ===================================================================


class TestTopPainpoints:

    def test_returns_ordered_results(self):
        results = get_top_painpoints(limit=10)
        assert len(results) >= 1
        assert results[0]["title"] == "LLM inference hangs on Apple Silicon"
        assert results[0]["signal_count"] == 3

    def test_category_field_present(self):
        results = get_top_painpoints(limit=10)
        for r in results:
            assert "category" in r

    def test_ordering(self):
        results = get_top_painpoints(limit=10)
        for i in range(len(results) - 1):
            a, b = results[i], results[i + 1]
            assert (a["signal_count"], a["severity"]) >= (b["signal_count"], b["severity"])


# ===================================================================
# Test 10 — Agent query: painpoints by category
# ===================================================================


class TestPainpointsByCategory:

    def test_filter_by_category(self):
        results = get_painpoints_by_category("LLM Infrastructure")
        assert len(results) >= 1
        for r in results:
            assert r["category"] == "LLM Infrastructure"

    def test_excludes_other_categories(self):
        results = get_painpoints_by_category("CI/CD & DevOps")
        llm_titles = {r["title"] for r in results}
        assert "LLM inference hangs on Apple Silicon" not in llm_titles

    def test_nonexistent_category_returns_empty(self):
        results = get_painpoints_by_category("Nonexistent Category")
        assert results == []


# ===================================================================
# Test 11 — Agent query: painpoints by subreddit
# ===================================================================


class TestPainpointsBySubreddit:

    def test_localllama_results(self):
        results = get_painpoints_by_subreddit("localllama")
        assert len(results) >= 1
        for r in results:
            assert "evidence_count" in r
            assert r["evidence_count"] >= 1

    def test_devops_includes_cross_sub_painpoint(self):
        results = get_painpoints_by_subreddit("devops")
        titles = {r["title"] for r in results}
        assert "LLM inference hangs on Apple Silicon" in titles, (
            "Merged painpoint with evidence from devops (new_pp) should appear"
        )

    def test_nonexistent_subreddit_empty(self):
        results = get_painpoints_by_subreddit("nonexistent_sub")
        assert results == []


# ===================================================================
# Test 12 — Agent query: subreddit summary
# ===================================================================


class TestSubredditSummary:

    def test_localllama_summary(self):
        summary = get_subreddit_summary("localllama")
        assert summary["subreddit"] == "localllama"
        assert summary["post_count"] == 1
        assert summary["comment_count"] == 2
        assert summary["painpoint_count"] >= 1
        assert isinstance(summary["top_categories"], list)

    def test_devops_summary(self):
        summary = get_subreddit_summary("devops")
        assert summary["post_count"] == 1
        assert summary["comment_count"] == 0

    def test_nonexistent_subreddit(self):
        summary = get_subreddit_summary("nonexistent_sub")
        assert summary["post_count"] == 0
        assert summary["comment_count"] == 0
        assert summary["painpoint_count"] == 0


# ===================================================================
# Test 13 — Agent query: global stats
# ===================================================================


class TestGlobalStats:

    def test_counts(self):
        stats = get_stats()
        assert stats["posts"] == 2
        assert stats["comments"] == 2
        assert stats["categories"] > 0
        assert stats["painpoints"] >= 1
        assert stats["subreddits"] == 2

    def test_unmerged_count(self):
        stats = get_stats()
        total_pending = stats["pending_painpoints"]
        merged_count = 3  # pp1, pp3, new_pp are merged
        assert stats["unmerged_pending"] == total_pending - merged_count


# ===================================================================
# Test 14 — FK cascade behaviour
# ===================================================================


class TestFKCascade:

    def test_comment_cascade_on_post_delete(self):
        """Deleting a post cascades to its comments."""
        sacrifice_post = upsert_post({
            "name": "t3_sacrifice",
            "subreddit": "test",
            "title": "Sacrifice post",
            "permalink": "/r/test/sacrifice/",
        })
        sacrifice_comment = upsert_comment(sacrifice_post, {
            "name": "t1_sacrifice_c",
            "body": "sacrifice comment",
        })

        conn = db.get_db()
        assert conn.execute(
            "SELECT COUNT(*) FROM comments WHERE id = ?", (sacrifice_comment,)
        ).fetchone()[0] == 1

        conn.execute("DELETE FROM posts WHERE id = ?", (sacrifice_post,))
        conn.commit()

        assert conn.execute(
            "SELECT COUNT(*) FROM comments WHERE id = ?", (sacrifice_comment,)
        ).fetchone()[0] == 0
        conn.close()

    def test_pending_painpoint_blocks_post_delete(self):
        """pending_painpoints.post_id has no CASCADE — delete should fail."""
        blocker_post = upsert_post({
            "name": "t3_blocker",
            "subreddit": "test",
            "title": "Blocker post",
            "permalink": "/r/test/blocker/",
        })
        save_pending_painpoint(blocker_post, "Blocks deletion", severity=5)

        conn = db.get_db()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM posts WHERE id = ?", (blocker_post,))
            conn.commit()
        conn.close()

    def test_painpoint_source_cascade(self):
        """Deleting a merged painpoint cascades to painpoint_sources."""
        pp = save_pending_painpoint(
            _state["post2_id"], "Cascade source test", severity=5
        )
        pid = _test_merge_pending(
            [pp], title="Cascade test painpoint", severity=5
        )

        conn = db.get_db()
        assert conn.execute(
            "SELECT COUNT(*) FROM painpoint_sources WHERE painpoint_id = ?", (pid,)
        ).fetchone()[0] == 1

        conn.execute("DELETE FROM painpoints WHERE id = ?", (pid,))
        conn.commit()

        assert conn.execute(
            "SELECT COUNT(*) FROM painpoint_sources WHERE painpoint_id = ?", (pid,)
        ).fetchone()[0] == 0
        conn.close()


# ===================================================================
# Test 15 — Empty DB queries don't crash
# ===================================================================


class TestEmptyDBQueries:

    @pytest.fixture(autouse=True)
    def fresh_db(self, tmp_path):
        """Separate temp DB with only seeded categories."""
        original = db.DB_PATH
        db.DB_PATH = tmp_path / "empty.db"
        db.init_db()
        yield
        db.DB_PATH = original

    def test_top_painpoints_empty(self):
        assert get_top_painpoints() == []

    def test_evidence_empty(self):
        assert get_painpoint_evidence(999) == []

    def test_by_category_empty(self):
        assert get_painpoints_by_category("LLM Infrastructure") == []

    def test_by_subreddit_empty(self):
        assert get_painpoints_by_subreddit("nonexistent") == []

    def test_subreddit_summary_empty(self):
        s = get_subreddit_summary("nonexistent")
        assert s["post_count"] == 0
        assert s["comment_count"] == 0
        assert s["painpoint_count"] == 0

    def test_stats_empty(self):
        stats = get_stats()
        assert stats["posts"] == 0
        assert stats["comments"] == 0
        assert stats["painpoints"] == 0
        assert stats["categories"] > 0

    def test_run_sql_select(self):
        result = run_sql("SELECT COUNT(*) AS cnt FROM posts")
        assert result[0]["cnt"] == 0

    def test_run_sql_rejects_mutation(self):
        result = run_sql("DELETE FROM posts")
        assert "error" in result

    def test_unmerged_empty(self):
        assert get_unmerged_pending() == []

    def test_category_helpers(self):
        assert get_category_list_flat() != []
        assert get_all_categories() != []
        assert get_root_categories() != []
        assert get_category_by_name(None) is None
        assert get_category_id_by_name("Nonexistent") is None
