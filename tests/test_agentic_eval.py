"""Hermetic tests for evaluation.agentic_eval helpers.

Currently focused on ``cross_run_pending_diff`` because it joins on
Reddit identity (post/comment permalinks) rather than synthetic row
IDs — that's the contract that the cross-run comparison story depends
on, so it earns a unit test.

The driver in ``compare_runs.py`` is a thin markdown wrapper over the
helper plus ``metrics.json`` reads; it's exercised end-to-end against
real run dirs in ``evaluation/agentic_eval/runs/`` so a hermetic test
for the markdown shape would mostly assert on cosmetics.

Run with:  pytest tests/test_agentic_eval.py -v
"""

import db
from db.painpoints import save_pending_painpoints_batch
from db.posts import upsert_comment, upsert_post
from evaluation.agentic_eval.inspect_db import cross_run_pending_diff


# ============================================================
# Builders — minimal Reddit-shaped fixtures
# ============================================================

def _post(name, subreddit, *, permalink, title="t"):
    """Return a post dict in the shape ``upsert_post`` consumes."""
    return {
        "name": name, "subreddit": subreddit, "title": title,
        "selftext": "body", "url": f"https://reddit.com{permalink}",
        "author": "u", "score": 10, "upvote_ratio": 1.0,
        "num_comments": 0, "permalink": permalink,
        "created_utc": 1.0, "is_self": True, "stickied": False,
    }


def _comment(name, post_name, *, permalink, body="c"):
    """Comment dict tagged with the post_name it belongs to (the
    ``post_name`` field is consumed by ``_seed_run`` to resolve the
    parent post's internal id; it isn't part of the Reddit shape)."""
    return {
        "post_name": post_name,
        "name": name, "parent_id": post_name, "author": "u", "body": body,
        "score": 1, "controversiality": 0, "permalink": permalink,
        "created_utc": 1.0, "depth": 0,
    }


def _seed_run(db_path, *, posts, comments, pendings):
    """Initialise a fresh DB at ``db_path`` and populate it with the
    given posts/comments/pendings."""
    db.DB_PATH = db_path
    db.init_db()
    post_ids = {p["name"]: upsert_post(p) for p in posts}
    comment_ids = {}
    for c in comments:
        # copy so multiple _seed_run calls can re-use the same fixture dict
        c = dict(c)
        post_id = post_ids[c.pop("post_name")]
        comment_ids[c["name"]] = upsert_comment(post_id, c)
    materialised = [
        {
            **pp,
            "post_id": post_ids[pp["post_name"]],
            "comment_id": (comment_ids[pp["comment_name"]]
                           if pp.get("comment_name") else None),
        }
        for pp in pendings
    ]
    save_pending_painpoints_batch(materialised, embedder=None)


# ============================================================
# Tests
# ============================================================


class TestCrossRunPendingDiff:
    def test_joins_by_reddit_permalink_not_synthetic_id(self, tmp_path):
        """Same Reddit cells must be matched across two independent runs
        even when SQLite renumbers the row IDs (which it does because
        they're per-DB AUTOINCREMENT)."""
        original = db.DB_PATH

        # Run A: 2 posts, 1 comment, 3 pendings.
        post_a = _post("t3_a", "self", permalink="/r/self/comments/a/")
        post_b = _post("t3_b", "self", permalink="/r/self/comments/b/")
        comment_x = _comment(
            "t1_x", "t3_a", permalink="/r/self/comments/a/cx/", body="cx body"
        )
        run_a = tmp_path / "run_a.db"
        try:
            _seed_run(
                run_a,
                posts=[post_a, post_b], comments=[comment_x],
                pendings=[
                    # Cell A1: post_a body  — KEPT in both runs (changed quote)
                    {"post_name": "t3_a", "title": "A1 old title",
                     "description": "d", "severity": 7,
                     "category_name": None,
                     "quoted_text": "OLD-A1-quote"},
                    # Cell A2: post_a + comment_x — DROPPED in B
                    {"post_name": "t3_a", "comment_name": "t1_x",
                     "title": "A2 dropped", "description": "d",
                     "severity": 5, "category_name": None,
                     "quoted_text": "OLD-A2-quote"},
                    # Cell A3: post_b body — KEPT in both runs (unchanged)
                    {"post_name": "t3_b", "title": "A3 same",
                     "description": "d", "severity": 4,
                     "category_name": None,
                     "quoted_text": "SAME-A3-quote"},
                ],
            )

            # Run B: same posts/comments scraped fresh (so SQLite assigns
            # different row IDs!), different pending mix.
            run_b = tmp_path / "run_b.db"
            _seed_run(
                run_b,
                posts=[post_a, post_b], comments=[comment_x],
                pendings=[
                    # Cell A1 again: same Reddit identity, different content
                    {"post_name": "t3_a", "title": "A1 new title",
                     "description": "d", "severity": 3,
                     "category_name": None,
                     "quoted_text": "NEW-A1-quote"},
                    # Cell A3 again: identical content
                    {"post_name": "t3_b", "title": "A3 same",
                     "description": "d", "severity": 4,
                     "category_name": None,
                     "quoted_text": "SAME-A3-quote"},
                    # New cell B1: post_a + comment_x with a totally
                    # different angle — only in B
                    {"post_name": "t3_a", "comment_name": "t1_x",
                     "title": "B1 added", "description": "d",
                     "severity": 6, "category_name": None,
                     "quoted_text": "NEW-B1-quote"},
                ],
            )
        finally:
            db.DB_PATH = original

        diff = cross_run_pending_diff(run_a, run_b)

        # Common cells: A1 (post_a body) + A3 (post_b body).
        # NOT comment_x — that pending was dropped in B and a different
        # pending replaced it on the same cell, so it's now in both
        # 'only_a' (the dropped one) and 'only_b' (the new one). Wait
        # no — same Reddit cell, so they should be paired in 'common'.
        common_titles = {(a["title"], b["title"]) for a, b in diff["common"]}
        assert common_titles == {
            ("A1 old title", "A1 new title"),
            ("A3 same", "A3 same"),
            ("A2 dropped", "B1 added"),
        }, common_titles

        # Both A2/B1 are on the same Reddit cell (post_a + comment_x),
        # so they pair up as a "common cell that changed", not as
        # only_a / only_b. The dropped/added story would only show
        # them as separate if Reddit identity differed.
        assert diff["only_a"] == []
        assert diff["only_b"] == []

    def test_only_a_and_only_b_buckets_when_cells_diverge(self, tmp_path):
        """Pendings on Reddit cells that one side never touched fall
        into only_a or only_b, sorted by severity descending."""
        original = db.DB_PATH

        post_a = _post("t3_aa", "s", permalink="/r/s/comments/aa/")
        post_b = _post("t3_bb", "s", permalink="/r/s/comments/bb/")
        post_c = _post("t3_cc", "s", permalink="/r/s/comments/cc/")
        try:
            run_a = tmp_path / "a.db"
            _seed_run(
                run_a, posts=[post_a, post_b], comments=[],
                pendings=[
                    {"post_name": "t3_aa", "title": "low sev only-a",
                     "description": "d", "severity": 3,
                     "category_name": None, "quoted_text": "x"},
                    {"post_name": "t3_bb", "title": "high sev only-a",
                     "description": "d", "severity": 8,
                     "category_name": None, "quoted_text": "y"},
                ],
            )
            run_b = tmp_path / "b.db"
            _seed_run(
                run_b, posts=[post_a, post_c], comments=[],
                pendings=[
                    {"post_name": "t3_cc", "title": "only-b",
                     "description": "d", "severity": 6,
                     "category_name": None, "quoted_text": "z"},
                ],
            )
        finally:
            db.DB_PATH = original

        diff = cross_run_pending_diff(run_a, run_b)

        # Only post_a was scraped on BOTH runs, but neither run had a
        # pending on it — common is empty.
        assert diff["common"] == []

        # only_a sorted by severity desc: high (8) before low (3).
        assert [r["title"] for r in diff["only_a"]] == [
            "high sev only-a", "low sev only-a",
        ]
        assert [r["title"] for r in diff["only_b"]] == ["only-b"]

    def test_post_body_and_comment_pendings_use_separate_keys(self, tmp_path):
        """A pending on a post body must NOT collide with a pending on
        a comment under the same post (the join key has to disambiguate
        the post-body bucket from each comment's permalink)."""
        original = db.DB_PATH

        post = _post("t3_pp", "s", permalink="/r/s/comments/pp/")
        comment = _comment(
            "t1_qq", "t3_pp", permalink="/r/s/comments/pp/qq/", body="qq",
        )
        try:
            run_a = tmp_path / "a.db"
            _seed_run(
                run_a, posts=[post], comments=[comment],
                pendings=[
                    {"post_name": "t3_pp", "title": "post-body pending",
                     "description": "d", "severity": 5,
                     "category_name": None, "quoted_text": "post body quote"},
                    {"post_name": "t3_pp", "comment_name": "t1_qq",
                     "title": "comment pending",
                     "description": "d", "severity": 5,
                     "category_name": None, "quoted_text": "comment quote"},
                ],
            )
            run_b = tmp_path / "b.db"
            _seed_run(
                run_b, posts=[post], comments=[comment],
                pendings=[
                    {"post_name": "t3_pp", "title": "post-body pending v2",
                     "description": "d", "severity": 5,
                     "category_name": None, "quoted_text": "different"},
                ],
            )
        finally:
            db.DB_PATH = original

        diff = cross_run_pending_diff(run_a, run_b)

        # The post-body cell pairs across runs.
        assert len(diff["common"]) == 1
        a, b = diff["common"][0]
        assert a["title"] == "post-body pending"
        assert b["title"] == "post-body pending v2"

        # The comment-side pending had no counterpart in B and must
        # NOT have been confused with the post-body pending despite
        # sharing post.permalink.
        assert [r["title"] for r in diff["only_a"]] == ["comment pending"]
        assert diff["only_b"] == []
