"""Hermetic tests for the painpoint_extraction package.

Covers:
- postprocess_painpoints: quote faithfulness + attribution repair (unit tests)
- extract_painpoints_from_posts: pure dict-in / dict-out contract with
  stubbed llm_call (no API key required)

The live LLM integration test lives in tests/live/test_extraction_live.py.

Run with:  pytest tests/test_extraction.py -v
"""

import asyncio
from typing import get_args

from painpoint_extraction.evidence_filter import (
    ALLOWED_EVIDENCE_TYPES,
    DROP_EVIDENCE_TYPES,
    EvidenceType,
    filter_non_pain_items,
)
from painpoint_extraction.postprocess import postprocess_painpoints

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
          description="desc", severity=5, category_name="Uncategorized",
          evidence_type=None):
    item = {
        "post_id": post_id, "comment_id": comment_id,
        "quoted_text": quoted_text, "title": title,
        "description": description, "severity": severity,
        "category_name": category_name,
    }
    if evidence_type is not None:
        item["evidence_type"] = evidence_type
    return item


# ===========================================================================
# evidence_filter — unit tests
# ===========================================================================


class TestEvidenceFilter:
    def test_label_set_is_consistent(self):
        """ALLOWED + DROP tuples are the source of truth; the EvidenceType
        Literal is derived from them. Asserts the derivation hasn't drifted
        and that the two buckets are disjoint."""
        allowed = set(ALLOWED_EVIDENCE_TYPES)
        drop = set(DROP_EVIDENCE_TYPES)
        assert allowed.isdisjoint(drop), (
            f"label appears in both buckets: {allowed & drop}"
        )
        assert set(get_args(EvidenceType)) == allowed | drop

    def test_keeps_only_explicit_pain_labels_and_strips_internal_field(self):
        items = [
            _item(title="Direct", evidence_type="direct_complaint"),
            _item(title="Missing", evidence_type="missing_feature"),
            _item(title="Broken", evidence_type="broken_experience"),
            _item(title="Need", evidence_type="unmet_need"),
            _item(title="Joke", evidence_type="joke_meme_sarcasm"),
            _item(title="Pitch", evidence_type="self_promotion_showcase"),
        ]

        kept, stats = filter_non_pain_items(items)

        assert [item["title"] for item in kept] == [
            "Direct", "Missing", "Broken", "Need",
        ]
        assert all("evidence_type" not in item for item in kept)
        assert stats["kept"] == 4
        assert stats["dropped"] == 2
        assert stats["dropped_by_evidence_type"] == {
            "joke_meme_sarcasm": 1,
            "self_promotion_showcase": 1,
        }
        assert stats["dropped_items"] == [
            {
                "evidence_type": "joke_meme_sarcasm",
                "post_id": 1,
                "comment_id": None,
                "title": "Joke",
                "quoted_text": "some phrase",
            },
            {
                "evidence_type": "self_promotion_showcase",
                "post_id": 1,
                "comment_id": None,
                "title": "Pitch",
                "quoted_text": "some phrase",
            },
        ]

    def test_missing_or_unknown_evidence_type_fails_closed(self):
        items = [
            _item(title="Missing"),
            _item(title="Unknown", evidence_type="maybe_pain"),
        ]

        kept, stats = filter_non_pain_items(items)

        assert kept == []
        assert stats["kept"] == 0
        assert stats["dropped"] == 2
        assert stats["dropped_by_evidence_type"] == {"unknown": 2}
        assert [item["title"] for item in stats["dropped_items"]] == [
            "Missing", "Unknown",
        ]


# ===========================================================================
# postprocess_painpoints — unit tests
# ===========================================================================


class TestPostprocessPainpoints:
    def test_exact_keep_allows_long_quote(self):
        quote = "When I try to enter a value in a cell"
        post = _post(selftext=f"Notion is slow. {quote}, I wait.")
        items = [_item(post_id=1, comment_id=None, quoted_text=quote)]

        kept, stats = postprocess_painpoints(items, [(post, [])])

        assert kept[0]["quoted_text"] == quote
        assert kept[0]["comment_id"] is None
        assert stats["kept"] == 1
        assert stats["dropped"] == 0

    def test_repairs_comment_attribution(self):
        post = _post(selftext="nothing relevant here")
        c1 = _comment(id=100, body="wrong comment")
        c2 = _comment(id=200, body="correct comment with target words")
        items = [_item(post_id=1, comment_id=100, quoted_text="target words")]

        kept, stats = postprocess_painpoints(items, [(post, [c1, c2])])

        assert kept[0]["comment_id"] == 200
        assert stats["attribution_fixed"] == 1

    def test_fuzzy_repairs_quote_and_numeric_value_from_source(self):
        source = "no single client should be more than 25% of revenue"
        post = _post(selftext=source)
        items = [_item(
            post_id=1,
            quoted_text="no single client should be more than 30% of revenue",
        )]

        kept, stats = postprocess_painpoints(items, [(post, [])])

        assert kept[0]["quoted_text"] == source
        assert stats["fuzzy_repaired"] == 1
        assert stats["numeric_entity_repaired"] == 1

    def test_drops_invalid_or_stitched_quote(self):
        post = _post(selftext="alpha phrase is here. omega phrase is elsewhere.")
        items = [_item(post_id=1, quoted_text="alpha phrase... omega phrase")]

        kept, stats = postprocess_painpoints(items, [(post, [])])

        assert kept == []
        assert stats["dropped"] == 1

    def test_drops_ambiguous_fuzzy_match(self):
        post = _post(
            selftext=(
                "alpha beta gamma meta. "
                "alpha beta gamma beta."
            )
        )
        items = [_item(post_id=1, quoted_text="alpha beta gamma zeta")]

        kept, stats = postprocess_painpoints(items, [(post, [])])

        assert kept == []
        assert stats["dropped"] == 1

    def test_reports_batch_counters(self):
        post = _post(selftext="source quote one. source quote three")
        c1 = _comment(id=100, body="source quote two")
        items = [
            _item(post_id=1, quoted_text="source quote one"),
            _item(post_id=1, comment_id=None, quoted_text="source quote two"),
            _item(post_id=1, quoted_text="missing quote"),
        ]

        kept, stats = postprocess_painpoints(items, [(post, [c1])])

        assert len(kept) == 2
        assert stats == {
            "kept": 2,
            "attribution_fixed": 1,
            "fuzzy_repaired": 0,
            "numeric_entity_repaired": 0,
            "dropped": 1,
        }


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
                 "evidence_type": "direct_complaint",
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
            {"id": 1, "title": "Alpha", "selftext": "alpha body test phrase",
             "subreddit": "test", "score": 10, "num_comments": 2,
             "comments": [{"id": 10, "body": "comment body", "score": 3}]},
            {"id": 2, "title": "Beta", "selftext": "beta body test phrase",
             "subreddit": "test", "score": 8, "num_comments": 0,
             "comments": []},
        ]
        items, _ = asyncio.run(extract_painpoints_from_posts(posts, []))
        assert {it["post_id"] for it in items} == {1, 2}
        assert all(1 <= it["severity"] <= 10 for it in items)
        assert all(it["category_name"] == "Uncategorized" for it in items)
        assert all("evidence_type" not in it for it in items)

    def test_missing_comments_key_ok(self, monkeypatch):
        """Caller passes a post dict without a ``comments`` key — must
        not crash. (Shape tolerance for upstream sources like closed's
        engine_runner that may not always nest comments.)"""
        from painpoint_extraction import extract_painpoints_from_posts
        self._stub_llm_call(monkeypatch)

        posts = [{"id": 99, "title": "t", "selftext": "test phrase",
                  "subreddit": "s",
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
