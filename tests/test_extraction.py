"""Hermetic tests for the painpoint_extraction package.

Covers:
- _fix_attribution: exact substring re-attribution logic (unit tests)
- extract_painpoints_from_posts: pure dict-in / dict-out contract with
  stubbed llm_call (no API key required)

The live LLM integration test lives in tests/live/test_extraction_live.py.

Run with:  pytest tests/test_extraction.py -v
"""

import asyncio

from painpoint_extraction.extractor import _fix_attribution

# ===========================================================================
# Helpers
# ===========================================================================

def _post(id=1, title="Test post", selftext="post body text", subreddit="test",
          score=10, num_comments=5):
    return {
        "id": id, "title": title, "selftext": selftext,
        "subreddit": subreddit, "score": score, "num_comments": num_comments,
    }


def _comment(id=100, body="comment body text", score=5):
    return {"id": id, "body": body, "score": score}


def _item(post_id=1, comment_id=None, quoted_text="some phrase", title="Pain",
          description="desc", severity=5, category_name="Uncategorized"):
    return {
        "post_id": post_id, "comment_id": comment_id,
        "quoted_text": quoted_text, "title": title,
        "description": description, "severity": severity,
        "category_name": category_name,
    }


# ===========================================================================
# _fix_attribution — unit tests
# ===========================================================================


class TestFixAttribution:
    def test_correct_attribution_unchanged(self):
        post = _post(selftext="the post body has some phrase in it")
        items = [_item(post_id=1, comment_id=None, quoted_text="some phrase")]
        assert _fix_attribution(items, [(post, [])]) == 0
        assert items[0]["comment_id"] is None

    def test_fixes_wrong_comment_to_post_body(self):
        post = _post(selftext="the post body has key phrase in it")
        comment = _comment(id=100, body="unrelated comment")
        items = [_item(post_id=1, comment_id=100, quoted_text="key phrase")]

        assert _fix_attribution(items, [(post, [comment])]) == 1
        assert items[0]["comment_id"] is None

    def test_fixes_post_body_to_comment(self):
        post = _post(selftext="nothing relevant here")
        comment = _comment(id=200, body="this comment has the key phrase")
        items = [_item(post_id=1, comment_id=None, quoted_text="key phrase")]

        assert _fix_attribution(items, [(post, [comment])]) == 1
        assert items[0]["comment_id"] == 200

    def test_fixes_wrong_comment_to_correct_comment(self):
        post = _post(selftext="nothing here")
        c1 = _comment(id=100, body="wrong comment")
        c2 = _comment(id=200, body="correct comment with target words")
        items = [_item(post_id=1, comment_id=100, quoted_text="target words")]

        assert _fix_attribution(items, [(post, [c1, c2])]) == 1
        assert items[0]["comment_id"] == 200

    def test_prefers_current_when_multiple_matches(self):
        post = _post(selftext="shared words appear here too")
        comment = _comment(id=100, body="shared words in comment")
        items = [_item(post_id=1, comment_id=100, quoted_text="shared words")]

        assert _fix_attribution(items, [(post, [comment])]) == 0
        assert items[0]["comment_id"] == 100

    def test_swaps_both_directions_in_same_batch(self):
        post = _post(id=1, selftext="post has alpha phrase")
        c1 = _comment(id=100, body="comment has beta phrase")
        items = [
            _item(post_id=1, comment_id=100, quoted_text="alpha phrase"),
            _item(post_id=1, comment_id=None, quoted_text="beta phrase"),
        ]

        assert _fix_attribution(items, [(post, [c1])]) == 2
        assert items[0]["comment_id"] is None
        assert items[1]["comment_id"] == 100

    def test_no_crash_on_empty_or_missing_quote(self):
        batch = [(_post(), [_comment()])]
        assert _fix_attribution([], batch) == 0
        assert _fix_attribution([_item(quoted_text="")], batch) == 0
        assert _fix_attribution([_item(quoted_text="hallucinated")], batch) == 0


# ===========================================================================
# extract_painpoints_from_posts — pure function contract
# ===========================================================================
# Direct tests for the dict-in / dict-out entry point. Pinned separately
# from the legacy extract_painpoints path so a refactor of either can't
# silently break the other's contract. Hermetic — stubs out llm_call so
# OPENAI_API_KEY is not required.


class TestExtractFromPosts:
    def _stub_llm_call(self, monkeypatch):
        """Replace llm_call with a stub that emits one painpoint per
        ``[Post N]`` marker it finds in the rendered batch text."""
        import re

        from painpoint_extraction import extractor as ext

        def fake(client, instructions, batch_text, *, response_model,
                 token_counter=None, **_):
            ids = [int(m) for m in re.findall(r"\[Post (\d+)\]", batch_text)]
            return response_model(painpoints=[
                {"title": f"Pain from post {pid}",
                 "description": "Synthetic test painpoint",
                 "severity": 5,
                 "quoted_text": "test phrase",
                 "category_name": "Uncategorized",
                 "post_id": pid,
                 "comment_id": None}
                for pid in ids
            ])

        monkeypatch.setattr(ext, "llm_call", fake)
        monkeypatch.setattr(ext, "get_client", lambda: object())

    def test_empty_input_returns_empty(self):
        from painpoint_extraction import extract_painpoints_from_posts
        items, usage = asyncio.run(extract_painpoints_from_posts([], []))
        assert items == []
        assert usage["total_tokens"] == 0

    def test_emits_one_painpoint_per_post(self, monkeypatch):
        from painpoint_extraction import extract_painpoints_from_posts
        self._stub_llm_call(monkeypatch)

        posts = [
            {"id": 1, "title": "Alpha", "selftext": "alpha body",
             "subreddit": "test", "score": 10, "num_comments": 2,
             "comments": [{"id": 10, "body": "comment body", "score": 3}]},
            {"id": 2, "title": "Beta", "selftext": "beta body",
             "subreddit": "test", "score": 8, "num_comments": 0,
             "comments": []},
        ]
        items, _ = asyncio.run(extract_painpoints_from_posts(posts, []))
        assert {it["post_id"] for it in items} == {1, 2}
        assert all(1 <= it["severity"] <= 10 for it in items)
        assert all(it["category_name"] == "Uncategorized" for it in items)

    def test_missing_comments_key_ok(self, monkeypatch):
        """Caller passes a post dict without a ``comments`` key — must
        not crash. (Shape tolerance for upstream sources like closed's
        engine_runner that may not always nest comments.)"""
        from painpoint_extraction import extract_painpoints_from_posts
        self._stub_llm_call(monkeypatch)

        posts = [{"id": 99, "title": "t", "selftext": "b", "subreddit": "s",
                  "score": 1, "num_comments": 0}]
        items, _ = asyncio.run(extract_painpoints_from_posts(posts, []))
        assert len(items) == 1
        assert items[0]["post_id"] == 99

    def test_categories_reach_prompt(self, monkeypatch):
        """Both ``{path, description}`` and ``{name, description}``
        category dicts get surfaced in the system instructions — so
        callers don't have to build the hierarchical path string
        themselves."""
        from painpoint_extraction import extract_painpoints_from_posts
        from painpoint_extraction import extractor as ext

        captured = {}
        def capturing(client, instructions, batch_text, *, response_model,
                      token_counter=None, **_):
            captured["instructions"] = instructions
            return response_model(painpoints=[])

        monkeypatch.setattr(ext, "llm_call", capturing)
        monkeypatch.setattr(ext, "get_client", lambda: object())

        posts = [{"id": 1, "title": "t", "selftext": "x", "subreddit": "s",
                  "score": 1, "num_comments": 0, "comments": []}]
        cats = [
            {"path": "AI/ML > Tools",   "description": "dev tooling"},
            {"name": "Health & Fitness", "description": "wellness"},
        ]
        asyncio.run(extract_painpoints_from_posts(posts, cats))

        assert "AI/ML > Tools" in captured["instructions"]
        assert "dev tooling" in captured["instructions"]
        assert "Health & Fitness" in captured["instructions"]
        assert "wellness" in captured["instructions"]

    def test_empty_categories_falls_back_to_uncategorized(self, monkeypatch):
        from painpoint_extraction import extract_painpoints_from_posts
        from painpoint_extraction import extractor as ext

        captured = {}
        def capturing(client, instructions, batch_text, *, response_model,
                      token_counter=None, **_):
            captured["instructions"] = instructions
            return response_model(painpoints=[])

        monkeypatch.setattr(ext, "llm_call", capturing)
        monkeypatch.setattr(ext, "get_client", lambda: object())

        posts = [{"id": 1, "title": "t", "selftext": "x", "subreddit": "s",
                  "score": 1, "num_comments": 0, "comments": []}]
        asyncio.run(extract_painpoints_from_posts(posts, []))
        assert "- Uncategorized" in captured["instructions"]
