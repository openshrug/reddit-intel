"""Live LLM integration test for painpoint extraction.

Makes a real OpenAI call on synthetic Reddit posts to verify the
extraction prompt + structured output contract hasn't regressed.

Extracted from tests/test_extraction.py so the fast hermetic unit tests
there can run without an API key. Requires OPENAI_API_KEY in .env.

Run with:  pytest tests/live/test_extraction_live.py -v
"""

import asyncio
import os

import pytest

from llm import TokenCounter
from painpoint_extraction.extractor import _format_batch, _process_batch

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


def _post(id=1, title="Test post", selftext="post body text", subreddit="test",
          score=10, num_comments=5):
    return {
        "id": id, "title": title, "selftext": selftext,
        "subreddit": subreddit, "score": score, "num_comments": num_comments,
    }


def _comment(id=100, body="comment body text", score=5):
    return {"id": id, "body": body, "score": score}


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

NON_PAIN_CONTROL_POST_IDS = {101, 102, 103, 104}
EXPLICIT_FRICTION_CONTROL_POST_IDS = {201, 202}

MIXED_EVIDENCE_POSTS = [
    (
        _post(
            id=101,
            title="Venmo me 1k refund lol",
            selftext=(
                "If my side project fails I'm just going to post 'Venmo me "
                "1k refund' and call it customer success. This is a joke thread."
            ),
            subreddit="indiehackers",
            score=88,
            num_comments=0,
        ),
        [],
    ),
    (
        _post(
            id=102,
            title="Show: Tenant Management Portal for small landlords",
            selftext=(
                "I built a Tenant Management Portal with rent reminders, "
                "maintenance tickets, and lease uploads. Happy to demo it for "
                "anyone looking for property software."
            ),
            subreddit="sideproject",
            score=44,
            num_comments=0,
        ),
        [],
    ),
    (
        _post(
            id=103,
            title="Cursor appreciation post",
            selftext=(
                "Cursor has been amazing for my workflow. The autocomplete is "
                "fast, the chat is helpful, and I love using it every day."
            ),
            subreddit="cursor",
            score=91,
            num_comments=0,
        ),
        [],
    ),
    (
        _post(
            id=104,
            title="Most startups treat sales as an afterthought",
            selftext=(
                "Founders should take sales seriously from day one. The lesson "
                "is simple: distribution matters more than clever features."
            ),
            subreddit="entrepreneur",
            score=73,
            num_comments=0,
        ),
        [],
    ),
    (
        _post(
            id=201,
            title="Show: I built a receipt parser because expense tracking hurt",
            selftext=(
                "I built a receipt parser because I kept losing reimbursable "
                "expenses. Every month I spent hours digging through email and "
                "photos to reconstruct missing receipt trails for my accountant."
            ),
            subreddit="sideproject",
            score=64,
            num_comments=0,
        ),
        [],
    ),
    (
        _post(
            id=202,
            title="Cloudflare bot rules broke my self-hosted app",
            selftext=(
                "After the policy change, Cloudflare started blocking my app's "
                "legitimate crawler callbacks. This broke my deployment workflow "
                "and I now have to manually unblock requests after every release."
            ),
            subreddit="selfhosted",
            score=112,
            num_comments=0,
        ),
        [],
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
        from painpoint_extraction.extractor import _build_instructions

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

    def test_filters_non_pain_controls_without_losing_explicit_friction(self):
        from llm import get_client
        from painpoint_extraction.extractor import _build_instructions

        client = get_client()
        instructions = _build_instructions()
        counter = TokenCounter()
        sem = asyncio.Semaphore(1)

        items = asyncio.run(
            _process_batch(
                0, 1, MIXED_EVIDENCE_POSTS, client, instructions, sem, counter,
            )
        )

        kept_post_ids = {item["post_id"] for item in items}
        leaked_non_pain_ids = kept_post_ids & NON_PAIN_CONTROL_POST_IDS
        assert leaked_non_pain_ids == set(), (
            "expected clear non-pain controls to be filtered, but kept "
            f"post IDs {sorted(leaked_non_pain_ids)} with items {items}"
        )

        kept_friction_ids = kept_post_ids & EXPLICIT_FRICTION_CONTROL_POST_IDS
        assert kept_friction_ids, (
            "expected at least one explicit-friction control to survive, "
            f"got items {items}"
        )

        for item in items:
            assert "evidence_type" not in item

        usage = counter.as_dict()
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0
