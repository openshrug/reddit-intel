"""Tests for reddit_scraper.py — async httpx scraper.

All HTTP is mocked via httpx.MockTransport. No real Reddit API calls.

Run with:  pytest tests/test_scraper.py -v
"""

import asyncio

import httpx
import pytest

from reddit_scraper import (
    BACKOFF_BASE,
    COMMENT_FIELDS,
    MAX_RETRIES,
    POST_FIELDS,
    _dedup_and_rank,
    _parse_comment,
    _parse_post,
    _request,
    scrape_comments,
    scrape_subreddit,
    scrape_subreddit_full,
)

# ===================================================================
# Helpers — fake Reddit API responses
# ===================================================================

def _fake_post(name, *, score=100, num_comments=20, subreddit="test",
               title=None, stickied=False, is_self=True):
    """Build a Reddit API post child object."""
    return {
        "kind": "t3",
        "data": {
            "name": name,
            "title": title or f"Post {name}",
            "score": score,
            "num_comments": num_comments,
            "subreddit": subreddit,
            "url": f"https://reddit.com/r/{subreddit}/{name}",
            "selftext": f"Body of {name}",
            "permalink": f"/r/{subreddit}/comments/{name.replace('t3_', '')}/post/",
            "author": "testuser",
            "upvote_ratio": 0.95,
            "created_utc": 1700000000.0,
            "is_self": is_self,
            "link_flair_text": "Discussion",
            "stickied": stickied,
        },
    }


def _listing_response(children, after=None):
    """Build a Reddit listing JSON response."""
    return {"data": {"children": children, "after": after}}


def _fake_comment(name, *, body="A comment", score=10, depth=0):
    """Build a Reddit API comment child object."""
    return {
        "kind": "t1",
        "data": {
            "name": name,
            "parent_id": "t3_abc",
            "body": body,
            "score": score,
            "author": "commenter",
            "created_utc": 1700000100.0,
            "depth": depth,
            "controversiality": 0,
            "permalink": f"/r/test/comments/abc/post/{name.replace('t1_', '')}/",
        },
    }


def _comments_response(comments):
    """Build a Reddit comments-page JSON response (listing pair)."""
    return [
        {"data": {"children": [_fake_post("t3_abc")]}},
        {"data": {"children": comments}},
    ]


def _noop_sem():
    return asyncio.Semaphore(100)


async def _fake_sleep(_seconds):
    """Drop-in for asyncio.sleep that returns immediately."""
    pass


# ===================================================================
# _parse_post / _parse_comment — dict shape
# ===================================================================

class TestParsePostShape:
    def test_all_fields_present(self):
        raw = _fake_post("t3_1")["data"]
        parsed = _parse_post(raw, "test")
        assert POST_FIELDS == set(parsed.keys())

    def test_permalink_normalised(self):
        raw = _fake_post("t3_1")["data"]
        parsed = _parse_post(raw, "test")
        assert parsed["permalink"].startswith("https://reddit.com")

    def test_selftext_truncated(self):
        raw = _fake_post("t3_1")["data"]
        raw["selftext"] = "x" * 20_000
        parsed = _parse_post(raw, "test")
        assert len(parsed["selftext"]) == 10_000

    def test_stickied_is_bool(self):
        parsed = _parse_post(_fake_post("t3_1", stickied=True)["data"], "test")
        assert parsed["stickied"] is True


class TestParseCommentShape:
    def test_all_fields_present(self):
        raw = _fake_comment("t1_a")["data"]
        parsed = _parse_comment(raw)
        assert COMMENT_FIELDS == set(parsed.keys())

    def test_permalink_normalised(self):
        raw = _fake_comment("t1_a")["data"]
        parsed = _parse_comment(raw)
        assert parsed["permalink"].startswith("https://reddit.com")

    def test_body_truncated(self):
        raw = _fake_comment("t1_a")["data"]
        raw["body"] = "x" * 10_000
        parsed = _parse_comment(raw)
        assert len(parsed["body"]) == 5_000


# ===================================================================
# _dedup_and_rank
# ===================================================================

class TestDedupAndRank:
    def test_removes_duplicates(self):
        post_a = _parse_post(_fake_post("t3_1", score=50)["data"], "test")
        post_a_dup = _parse_post(_fake_post("t3_1", score=50)["data"], "test")
        post_b = _parse_post(_fake_post("t3_2", score=100)["data"], "test")
        result = _dedup_and_rank([[post_a, post_b], [post_a_dup]])
        assert len(result) == 2

    def test_ordered_by_engagement(self):
        low = _parse_post(_fake_post("t3_1", score=10, num_comments=5)["data"], "test")
        high = _parse_post(_fake_post("t3_2", score=200, num_comments=50)["data"], "test")
        result = _dedup_and_rank([[low, high]])
        assert result[0]["name"] == "t3_2"
        assert result[1]["name"] == "t3_1"

    def test_empty_batches(self):
        assert _dedup_and_rank([[], []]) == []

    def test_preserves_first_seen(self):
        """When the same post appears in multiple batches, keep the first."""
        a1 = _parse_post(_fake_post("t3_1", score=10)["data"], "test")
        a2 = _parse_post(_fake_post("t3_1", score=999)["data"], "test")
        result = _dedup_and_rank([[a1], [a2]])
        assert len(result) == 1
        assert result[0]["score"] == 10


# ===================================================================
# scrape_subreddit (async, mocked HTTP)
# ===================================================================

class TestScrapeSubreddit:
    @pytest.mark.asyncio
    async def test_basic_listing(self):
        children = [_fake_post("t3_1"), _fake_post("t3_2")]
        body = _listing_response(children)

        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=body)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            posts = await scrape_subreddit(
                client, _noop_sem(), "test",
            )
        assert len(posts) == 2
        assert posts[0]["name"] == "t3_1"

    @pytest.mark.asyncio
    async def test_stickied_filtered(self):
        children = [
            _fake_post("t3_sticky", stickied=True),
            _fake_post("t3_normal"),
        ]
        body = _listing_response(children)
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=body)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            posts = await scrape_subreddit(
                client, _noop_sem(), "test",
            )
        assert len(posts) == 1
        assert posts[0]["name"] == "t3_normal"

    @pytest.mark.asyncio
    async def test_empty_listing(self):
        body = _listing_response([])
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=body)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            posts = await scrape_subreddit(
                client, _noop_sem(), "test",
            )
        assert posts == []

    @pytest.mark.asyncio
    async def test_404_raises(self):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(404)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RuntimeError, match="404"):
                await scrape_subreddit(
                    client, _noop_sem(), "nosuchsub",
                )

    @pytest.mark.asyncio
    async def test_429_retry(self, monkeypatch):
        monkeypatch.setattr("reddit_scraper.asyncio.sleep", _fake_sleep)
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429)
            return httpx.Response(200, json=_listing_response(
                [_fake_post("t3_1")]
            ))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            posts = await scrape_subreddit(
                client, _noop_sem(), "test",
            )
        assert len(posts) == 1
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_pagination(self):
        page1 = _listing_response([_fake_post("t3_1")], after="cursor_abc")
        page2 = _listing_response([_fake_post("t3_2")], after=None)

        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            if "after" not in str(req.url):
                return httpx.Response(200, json=page1)
            return httpx.Response(200, json=page2)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            posts = await scrape_subreddit(
                client, _noop_sem(), "test", pages=2,
            )
        assert len(posts) == 2
        assert call_count == 2


# ===================================================================
# scrape_comments (async, mocked HTTP)
# ===================================================================

class TestScrapeComments:
    @pytest.mark.asyncio
    async def test_basic(self):
        body = _comments_response([
            _fake_comment("t1_a", score=20),
            _fake_comment("t1_b", score=5),
        ])
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=body)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            comments = await scrape_comments(
                client, _noop_sem(),
                "https://reddit.com/r/test/comments/abc/post/",
            )
        assert len(comments) == 2
        assert comments[0]["score"] >= comments[1]["score"]

    @pytest.mark.asyncio
    async def test_non_200_returns_empty(self):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(403)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            comments = await scrape_comments(
                client, _noop_sem(),
                "https://reddit.com/r/test/comments/abc/post/",
            )
        assert comments == []

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        many = [_fake_comment(f"t1_{i}", score=100 - i) for i in range(20)]
        body = _comments_response(many)
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=body)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            comments = await scrape_comments(
                client, _noop_sem(),
                "https://reddit.com/r/test/comments/abc/post/",
                limit=5,
            )
        assert len(comments) == 5


# ===================================================================
# scrape_subreddit_full (async, mocked HTTP)
# ===================================================================

def _build_full_transport(posts_per_window=3, comments_per_post=2):
    """Build a mock transport that serves listing + comment requests."""
    all_posts = {}
    for w_idx, window in enumerate(["week", "month", "year"]):
        for i in range(posts_per_window):
            name = f"t3_{window}_{i}"
            all_posts.setdefault(name, _fake_post(
                name, score=100 * (w_idx + 1) + i * 10,
                num_comments=10 + i,
            ))
    # Week and month share some overlap with year
    overlap_name = "t3_overlap"
    all_posts[overlap_name] = _fake_post(overlap_name, score=500, num_comments=80)

    def handler(req):
        url = str(req.url)

        # Listing request
        if "/top" in url or "/hot" in url:
            for window in ["week", "month", "year"]:
                if f"t={window}" in url:
                    children = [
                        all_posts[k] for k in all_posts
                        if k.startswith(f"t3_{window}_") or k == overlap_name
                    ]
                    return httpx.Response(200, json=_listing_response(children))
            return httpx.Response(200, json=_listing_response([]))

        # Comment request
        if "/comments/" in url:
            coms = [_fake_comment(f"t1_{i}") for i in range(comments_per_post)]
            return httpx.Response(200, json=_comments_response(coms))

        return httpx.Response(404)

    return handler, all_posts


class TestScrapeSubredditFull:
    @pytest.mark.asyncio
    async def test_dedup_across_windows(self, monkeypatch):
        handler, _ = _build_full_transport(posts_per_window=3)
        monkeypatch.setattr(
            "reddit_scraper._oauth_headers",
            lambda: {"Authorization": "Bearer fake"},
        )
        transport = httpx.MockTransport(handler)
        posts = await scrape_subreddit_full(
            "test", posts_with_comments=2, _transport=transport,
        )
        names = [p["name"] for p in posts]
        assert len(names) == len(set(names)), "duplicates found"

    @pytest.mark.asyncio
    async def test_posts_with_comments_respected(self, monkeypatch):
        handler, _ = _build_full_transport(posts_per_window=5)

        comment_fetches = 0

        def counting_handler(req):
            nonlocal comment_fetches
            if "/comments/" in str(req.url):
                comment_fetches += 1
            return handler(req)

        monkeypatch.setattr(
            "reddit_scraper._oauth_headers",
            lambda: {"Authorization": "Bearer fake"},
        )
        transport = httpx.MockTransport(counting_handler)
        budget = 3
        await scrape_subreddit_full(
            "test", posts_with_comments=budget, _transport=transport,
        )
        assert comment_fetches == budget

    @pytest.mark.asyncio
    async def test_min_score_filter(self, monkeypatch):
        handler, _ = _build_full_transport(posts_per_window=5)

        comment_urls = []

        def tracking_handler(req):
            if "/comments/" in str(req.url):
                comment_urls.append(str(req.url))
            return handler(req)

        monkeypatch.setattr(
            "reddit_scraper._oauth_headers",
            lambda: {"Authorization": "Bearer fake"},
        )
        transport = httpx.MockTransport(tracking_handler)
        posts = await scrape_subreddit_full(
            "test", posts_with_comments=100, min_score=9999,
            _transport=transport,
        )
        assert len(comment_urls) == 0 or all(
            p["score"] >= 9999 for p in posts if "comments" in p
        )

    @pytest.mark.asyncio
    async def test_404_subreddit_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            "reddit_scraper._oauth_headers",
            lambda: {"Authorization": "Bearer fake"},
        )
        transport = httpx.MockTransport(
            lambda req: httpx.Response(404)
        )
        posts = await scrape_subreddit_full("nosuchsub", _transport=transport)
        assert posts == []

    @pytest.mark.asyncio
    async def test_partial_window_failure(self, monkeypatch):
        """One window fails, others succeed — still returns posts."""
        monkeypatch.setattr(
            "reddit_scraper._oauth_headers",
            lambda: {"Authorization": "Bearer fake"},
        )
        monkeypatch.setattr("reddit_scraper.asyncio.sleep", _fake_sleep)
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            url = str(req.url)
            if "t=week" in url:
                return httpx.Response(500)
            return httpx.Response(200, json=_listing_response(
                [_fake_post("t3_1", score=99)]
            ))

        transport = httpx.MockTransport(handler)
        posts = await scrape_subreddit_full(
            "test", posts_with_comments=0, _transport=transport,
        )
        assert len(posts) >= 1

    @pytest.mark.asyncio
    async def test_empty_subreddit(self, monkeypatch):
        monkeypatch.setattr(
            "reddit_scraper._oauth_headers",
            lambda: {"Authorization": "Bearer fake"},
        )
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=_listing_response([]))
        )
        posts = await scrape_subreddit_full("empty", _transport=transport)
        assert posts == []


# ===================================================================
# _request retry logic
# ===================================================================

class TestRequestRetry:
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await _request(client, _noop_sem(), "GET", "https://example.com")
        assert resp.status_code == 200
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_429_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("reddit_scraper.asyncio.sleep", _fake_sleep)
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(429)
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await _request(client, _noop_sem(), "GET", "https://example.com")
        assert resp.status_code == 200
        assert call_count == 3  # 2 retries + 1 success

    @pytest.mark.asyncio
    async def test_5xx_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("reddit_scraper.asyncio.sleep", _fake_sleep)
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await _request(client, _noop_sem(), "GET", "https://example.com")
        assert resp.status_code == 200
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries(self, monkeypatch):
        monkeypatch.setattr("reddit_scraper.asyncio.sleep", _fake_sleep)
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            return httpx.Response(429)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await _request(client, _noop_sem(), "GET", "https://example.com")
        assert resp.status_code == 429
        assert call_count == 1 + MAX_RETRIES

    @pytest.mark.asyncio
    async def test_404_raises_immediately(self):
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(RuntimeError, match="404"):
                await _request(client, _noop_sem(), "GET", "https://example.com/x")
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_4xx_returns_immediately(self):
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            return httpx.Response(403)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await _request(client, _noop_sem(), "GET", "https://example.com")
        assert resp.status_code == 403
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_after_header_respected(self, monkeypatch):
        delays = []
        async def tracking_sleep(seconds):
            delays.append(seconds)

        monkeypatch.setattr("reddit_scraper.asyncio.sleep", tracking_sleep)
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, headers={"Retry-After": "7"})
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await _request(client, _noop_sem(), "GET", "https://example.com")
        assert resp.status_code == 200
        assert delays == [7]

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self, monkeypatch):
        delays = []
        async def tracking_sleep(seconds):
            delays.append(seconds)

        monkeypatch.setattr("reddit_scraper.asyncio.sleep", tracking_sleep)

        def handler(req):
            return httpx.Response(500)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await _request(client, _noop_sem(), "GET", "https://example.com")
        assert len(delays) == MAX_RETRIES
        for i, d in enumerate(delays):
            assert d == BACKOFF_BASE * (2 ** i)

    @pytest.mark.asyncio
    async def test_comments_retry_on_503(self, monkeypatch):
        """scrape_comments uses _request, so 503 gets retried."""
        monkeypatch.setattr("reddit_scraper.asyncio.sleep", _fake_sleep)
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(503)
            return httpx.Response(200, json=_comments_response(
                [_fake_comment("t1_a")]
            ))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            comments = await scrape_comments(
                client, _noop_sem(),
                "https://reddit.com/r/test/comments/abc/post/",
            )
        assert len(comments) == 1
        assert call_count == 2


