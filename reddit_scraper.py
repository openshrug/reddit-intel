"""
Async Reddit scraper using OAuth (script app). 60 req/min limit.

All public functions are async coroutines using httpx. Callers that need
a sync entry point should use ``asyncio.run()``.

Low-level API client + subreddit-full scrape (three time windows,
dedup, configurable comment budget).
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent / "config.json"

# --- Tunables ---
# scrape_subreddit_full tunables
POSTS_PER_WINDOW = 100 # Reddit returns up to 100 posts per request
POSTS_WITH_COMMENTS = 60
# one post comments retrieval tunables
COMMENTS_PER_POST = 25
COMMENT_DEPTH = 2 # 1 = top-level only, 2 = top-level + direct replies, etc.
# request concurrency tunables
CONCURRENT_REQUESTS = 10
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds: 2, 4, 8
# post/comment body length tunables
MAX_POST_BODY_LEN = 10_000
MAX_COMMENT_BODY_LEN = 5_000

_token_cache = {"token": None, "expires_at": 0}


# ============================================================
# OAuth (sync — called once, result cached)
# ============================================================

def _get_token():
    """Get or refresh OAuth token (sync, cached)."""
    if _token_cache["token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["token"]

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "RedditPulse/1.0")

    if not client_id or not client_secret:
        return None

    resp = httpx.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": user_agent},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600) - 60
    return _token_cache["token"]


def _oauth_headers():
    token = _get_token()
    if not token:
        raise RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set in .env")
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": os.getenv("REDDIT_USER_AGENT", "RedditPulse/1.0"),
    }


# ============================================================
# Config helpers (sync, pure file I/O)
# ============================================================

def _load_subreddits():
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
        return [s["name"] for s in config.get("reddit", {}).get("subreddits", [])]
    return ["programming", "technology", "SideProject", "webdev",
            "MachineLearning", "artificial", "startups"]


def _load_min_score():
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
        return config.get("reddit", {}).get("min_score", 50)
    return 50


# ============================================================
# Post / comment dict builders
# ============================================================

POST_FIELDS = frozenset({
    "name", "subreddit", "title", "selftext", "url", "author", "score",
    "upvote_ratio", "num_comments", "permalink", "created_utc", "is_self",
    "link_flair_text", "stickied",
})

COMMENT_FIELDS = frozenset({
    "name", "parent_id", "body", "score", "author", "created_utc",
    "depth", "controversiality", "permalink",
})


def _parse_post(post_data: dict, subreddit: str) -> dict:
    """Normalise a Reddit API post object into our canonical dict shape."""
    permalink = post_data.get("permalink", "")
    if permalink and not permalink.startswith("http"):
        permalink = f"https://reddit.com{permalink}"
    return {
        "name": post_data.get("name", ""),
        "subreddit": subreddit or post_data.get("subreddit", ""),
        "title": post_data.get("title", ""),
        "selftext": (post_data.get("selftext", "") or "")[:MAX_POST_BODY_LEN],
        "url": post_data.get("url", ""),
        "author": post_data.get("author", ""),
        "score": post_data.get("score", 0),
        "upvote_ratio": post_data.get("upvote_ratio"),
        "num_comments": post_data.get("num_comments", 0),
        "permalink": permalink,
        "created_utc": post_data.get("created_utc"),
        "is_self": bool(post_data.get("is_self")),
        "link_flair_text": post_data.get("link_flair_text") or "",
        "stickied": bool(post_data.get("stickied")),
    }


def _parse_comment(comment_data: dict) -> dict:
    """Normalise a Reddit API comment object into our canonical dict shape."""
    permalink = comment_data.get("permalink", "")
    if permalink and not permalink.startswith("http"):
        permalink = f"https://reddit.com{permalink}"
    return {
        "name": comment_data.get("name", ""),
        "parent_id": comment_data.get("parent_id", ""),
        "body": (comment_data.get("body", "") or "")[:MAX_COMMENT_BODY_LEN],
        "score": comment_data.get("score", 0),
        "author": comment_data.get("author", ""),
        "created_utc": comment_data.get("created_utc"),
        "depth": comment_data.get("depth", 0),
        "controversiality": comment_data.get("controversiality", 0),
        "permalink": permalink,
    }


# ============================================================
# Retry with backoff
# ============================================================

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


async def _request(client, sem, method, url, **kwargs):
    """HTTP request with semaphore gating, retry on 429/5xx, exponential
    backoff. Returns the httpx.Response on success or after exhausting
    retries. Raises RuntimeError on 404."""
    last_resp = None
    for attempt in range(1 + MAX_RETRIES):
        async with sem:
            last_resp = await client.request(method, url, **kwargs)

        if last_resp.status_code == 404:
            raise RuntimeError(f"404 Not Found: {url}")

        if last_resp.status_code not in RETRYABLE_STATUS_CODES:
            return last_resp

        if attempt < MAX_RETRIES:
            retry_after = last_resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = int(retry_after)
            else:
                delay = BACKOFF_BASE * (2 ** attempt)
            log.warning(
                "HTTP %d from %s — retrying in %.1fs (attempt %d/%d)",
                last_resp.status_code, url, delay, attempt + 1, MAX_RETRIES,
            )
            await asyncio.sleep(delay)

    return last_resp


# ============================================================
# Low-level async API calls
# ============================================================

async def scrape_subreddit(client, sem, subreddit, sort="hot",
                           limit=25, time_filter="week", pages=1):
    """Fetch a subreddit listing. Returns list of post dicts."""
    url = f"https://oauth.reddit.com/r/{subreddit}/{sort}"
    posts = []
    after = None

    for _ in range(pages):
        params = {"limit": min(limit, 100), "t": time_filter, "raw_json": 1}
        if after:
            params["after"] = after

        resp = await _request(client, sem, "GET", url, params=params)
        resp.raise_for_status()

        data = resp.json().get("data", {})
        children = data.get("children", [])
        if not children:
            break

        for child in children:
            post = child.get("data", {})
            if post.get("stickied"):
                continue
            posts.append(_parse_post(post, subreddit))

        after = data.get("after")
        if not after:
            break

    return posts


async def scrape_comments(client, sem, permalink, limit=10, depth=1):
    """Fetch top comments for a post. Returns list of comment dicts."""
    path = permalink.replace("https://reddit.com", "").replace("https://www.reddit.com", "")
    url = f"https://oauth.reddit.com{path}"

    resp = await _request(
        client, sem, "GET", url,
        params={"limit": limit, "sort": "top", "depth": depth, "raw_json": 1},
    )

    if resp.status_code != 200:
        return []

    data = resp.json()
    if not isinstance(data, list) or len(data) < 2:
        return []

    comments = []
    for child in data[1].get("data", {}).get("children", []):
        if child.get("kind") != "t1":
            continue
        comments.append(_parse_comment(child.get("data", {})))

    return sorted(comments, key=lambda x: x["score"], reverse=True)[:limit]


async def search_reddit(client, sem, query, sort="relevance",
                        time_filter="month", limit=100, subreddit=None):
    """Search Reddit. Returns list of post dicts."""
    if subreddit:
        url = f"https://oauth.reddit.com/r/{subreddit}/search"
        params = {"q": query, "restrict_sr": "on", "sort": sort,
                  "t": time_filter, "limit": min(limit, 100), "raw_json": 1}
    else:
        url = "https://oauth.reddit.com/search"
        params = {"q": query, "sort": sort, "t": time_filter,
                  "limit": min(limit, 100), "raw_json": 1}

    resp = await _request(client, sem, "GET", url, params=params)

    if resp.status_code != 200:
        return []

    posts = []
    for child in resp.json().get("data", {}).get("children", []):
        post = child.get("data", {})
        if post.get("stickied"):
            continue
        posts.append(_parse_post(post, subreddit or ""))
    return posts


# ============================================================
# Dedup + ranking helpers (pure, sync)
# ============================================================

def _dedup_and_rank(listing_batches):
    """Merge multiple listing results, dedup by reddit fullname, sort by
    engagement (score + num_comments) descending."""
    seen = {}
    for batch in listing_batches:
        for post in batch:
            key = post.get("name") or post["permalink"]
            if key not in seen:
                seen[key] = post
    ranked = sorted(seen.values(),
                    key=lambda p: p["score"] + p["num_comments"],
                    reverse=True)
    return ranked


# ============================================================
# High-level: full subreddit scrape
# ============================================================

async def scrape_subreddit_full(subreddit, *, posts_per_window=POSTS_PER_WINDOW,
                                posts_with_comments=POSTS_WITH_COMMENTS, min_score=None,
                                comments_per_post=COMMENTS_PER_POST,
                                comment_depth=COMMENT_DEPTH,
                                _transport=None):
    """Scrape a subreddit across week/month/year, dedup, and fetch
    comments for the top posts by engagement.

    Args:
        subreddit: Subreddit name (without ``r/``).
        posts_per_window: Max posts per time window listing (max 100).
        posts_with_comments: How many top posts get their comments fetched.
        min_score: If set, only posts with ``score >= min_score`` are
            eligible for comment fetching.
        comments_per_post: Max comments to fetch per post.
        comment_depth: How deep into reply chains the API should return
            (1 = top-level only, 2 = top-level + direct replies, etc.).
        _transport: Override httpx transport (for testing with MockTransport).

    Returns:
        List of post dicts (sorted by engagement desc). Posts that had
        comments fetched carry a ``"comments"`` key with a list of
        comment dicts.
    """
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

    client_kwargs = {"headers": _oauth_headers(), "timeout": 15.0}
    if _transport is not None:
        client_kwargs["transport"] = _transport

    async with httpx.AsyncClient(**client_kwargs) as client:
        # Phase 1 — listings (parallel, errors logged per window)
        windows = ["week", "month", "year"]

        async def _fetch_window(window):
            try:
                return await scrape_subreddit(
                    client, sem, subreddit,
                    sort="top", limit=posts_per_window,
                    time_filter=window,
                )
            except Exception as exc:
                log.warning("r/%s top/%s failed: %s", subreddit, window, exc)
                return []

        listings = await asyncio.gather(*(_fetch_window(w) for w in windows))

        # Phase 2 — dedup + rank
        unique = _dedup_and_rank(listings)
        log.info("r/%s: %d unique posts from %d raw across %s",
                 subreddit, len(unique),
                 sum(len(b) for b in listings), windows)

        if not unique:
            return []

        # Phase 3 — pick comment targets
        if min_score is not None:
            targets = [p for p in unique if p["score"] >= min_score]
        else:
            targets = list(unique)
        targets = targets[:posts_with_comments]

        # Phase 4 — fetch comments (parallel, errors logged per post)
        async def _attach_comments(post):
            try:
                post["comments"] = await scrape_comments(
                    client, sem, post["permalink"],
                    limit=comments_per_post,
                    depth=comment_depth,
                )
            except Exception as exc:
                log.warning("Comments failed for %s: %s", post["permalink"], exc)
                post["comments"] = []

        await asyncio.gather(*(_attach_comments(p) for p in targets))
        log.info("r/%s: fetched comments for %d/%d posts",
                 subreddit, len(targets), len(unique))

    return unique


# ============================================================
# Multi-subreddit helpers
# ============================================================

SEED_SUBS = [
    "programming", "webdev", "devops", "ExperiencedDevs",
    "artificial", "MachineLearning", "LocalLLaMA",
    "ChatGPT", "ClaudeAI", "SideProject", "startups",
    "technology", "sysadmin",
]

SEED_QUERIES = [
    '"frustrated with" OR "I hate" OR "why is it so hard"',
    '"switched from" OR "moved away from" OR "gave up on"',
    '"wish there was" OR "someone should build"',
    '"just launched" OR "just shipped" OR "I built"',
    '"open source alternative" OR "free alternative"',
]


def _load_deep_subs():
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
        subs = [s["name"] for s in config.get("reddit", {}).get("subreddits", [])]
        if subs:
            return subs
    return SEED_SUBS


async def scrape_all_subreddits(subreddits=None, **kwargs):
    """Run scrape_subreddit_full for each subreddit sequentially
    (they share the same rate-limit budget)."""
    subreddits = subreddits or _load_deep_subs()
    all_results = {}
    for sub in subreddits:
        try:
            posts = await scrape_subreddit_full(sub, **kwargs)
            all_results[sub] = posts
            log.info("r/%s: %d posts", sub, len(posts))
        except Exception as exc:
            log.warning("r/%s: failed (%s)", sub, exc)
            all_results[sub] = []
    return all_results


# ============================================================
# CLI smoke test
# ============================================================

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    sub = sys.argv[1] if len(sys.argv) > 1 else "programming"
    posts = asyncio.run(scrape_subreddit_full(sub, posts_with_comments=5))
    print(f"\nTotal: {len(posts)} unique posts")
    for p in posts[:10]:
        n_comments = len(p.get("comments", []))
        print(f"  [{p['score']:>5}] {p['title'][:60]}"
              + (f"  ({n_comments} comments)" if n_comments else ""))
