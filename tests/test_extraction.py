"""Tests for the painpoint_extraction package.

Covers:
- _fix_attribution: exact substring re-attribution logic (unit tests)
- TokenCounter: thread-safe accumulator (unit test)
- Live LLM integration: real extraction call on synthetic posts

Run with:  pytest tests/test_extraction.py -v
"""

import os
import asyncio
import threading

import pytest

from painpoint_extraction.extractor import (
    _fix_attribution,
    _format_batch,
    _process_batch,
    ExtractionResult,
)
from llm import TokenCounter


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
# Live LLM integration test
# ===========================================================================

SYNTHETIC_POSTS = [
    (
        _post(
            id=1,
            title="Cursor keeps losing my context mid-conversation",
            selftext=(
                "Every time I paste a large file, Cursor forgets what I said "
                "earlier in the conversation. I have to re-explain the entire "
                "architecture every 3-4 messages. Really frustrating when "
                "working on complex refactors."
            ),
            subreddit="cursor",
            score=142,
            num_comments=3,
        ),
        [
            _comment(id=10, body=(
                "Same here. I started breaking my prompts into smaller chunks "
                "but then it loses the big picture. We need persistent memory "
                "across sessions."
            ), score=89),
            _comment(id=11, body=(
                "Try using @-mentions to pin files. It helped me but it's "
                "still not great for multi-file refactors."
            ), score=45),
            _comment(id=12, body=(
                "This is why I switched back to Copilot for large projects. "
                "Context window is just too small."
            ), score=23),
        ],
    ),
    (
        _post(
            id=2,
            title="Is there a way to auto-generate test files from source?",
            selftext=(
                "I write a new module, then I have to manually create the test "
                "file, import everything, write boilerplate setUp/tearDown. "
                "Every. Single. Time. Surely there's a tool that scaffolds "
                "tests automatically?"
            ),
            subreddit="Python",
            score=67,
            num_comments=2,
        ),
        [
            _comment(id=20, body=(
                "pytest-generate does some of this but it's abandoned. "
                "I ended up writing a custom cookiecutter template."
            ), score=31),
            _comment(id=21, body=(
                "Copilot can generate test stubs if you open the source file "
                "and prompt it, but the quality is hit-or-miss."
            ), score=18),
        ],
    ),
]

has_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@has_api_key
class TestLiveExtraction:
    """Integration test: real LLM call on synthetic posts."""

    @pytest.fixture(autouse=True)
    def _load_env(self):
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass

    def test_extraction_on_synthetic_posts(self):
        from llm import get_client
        from painpoint_extraction.extractor import (
            MODEL, REASONING_EFFORT, _build_instructions,
        )

        client = get_client()
        instructions = _build_instructions()
        counter = TokenCounter()
        sem = asyncio.Semaphore(1)

        items = asyncio.run(
            _process_batch(0, 1, SYNTHETIC_POSTS, client, instructions, sem, counter)
        )

        assert len(items) >= 2, (
            f"expected at least 2 painpoints from 2 rich posts, got {len(items)}"
        )

        for item in items:
            assert item["post_id"] in (1, 2)
            assert 1 <= item["severity"] <= 10
            assert len(item["title"]) > 0
            assert len(item["description"]) > 0
            assert len(item["quoted_text"]) > 0

        batch_text = _format_batch(SYNTHETIC_POSTS)
        all_source = batch_text.lower()
        verbatim = sum(
            1 for it in items
            if it["quoted_text"].lower().strip() in all_source
        )
        verbatim_pct = verbatim / len(items)
        assert verbatim_pct >= 0.5, (
            f"verbatim quote rate {verbatim_pct:.0%} is below 50% — "
            f"prompt may have regressed"
        )

        for item in items:
            if item["comment_id"] is not None:
                assert item["comment_id"] in (10, 11, 12, 20, 21), (
                    f"comment_id {item['comment_id']} is not a valid ID from "
                    f"the synthetic data"
                )

        usage = counter.as_dict()
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0


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
        from painpoint_extraction import extractor as ext
        from painpoint_extraction import extract_painpoints_from_posts

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
        from painpoint_extraction import extractor as ext
        from painpoint_extraction import extract_painpoints_from_posts

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
