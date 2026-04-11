"""
Reddit scraper using OAuth (script app). 60 req/min limit.

Low-level API client + advanced scraping strategies (deep scrape,
LLM-driven search queries, subreddit discovery).
"""

import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

CONFIG_FILE = Path(__file__).parent / "config.json"

_token_cache = {"token": None, "expires_at": 0}


# ============================================================
# LOW-LEVEL: OAuth + single-resource API calls
# ============================================================

def _get_token():
    """Get or refresh OAuth token."""
    if _token_cache["token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["token"]

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "RedditPulse/1.0")

    if not client_id or not client_secret:
        return None

    resp = requests.post(
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
        raise Exception("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set in .env")
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": os.getenv("REDDIT_USER_AGENT", "RedditPulse/1.0"),
    }


def _load_subreddits():
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
        return [s["name"] for s in config.get("reddit", {}).get("subreddits", [])]
    return ["programming", "technology", "SideProject", "webdev", "MachineLearning", "artificial", "startups"]


def _load_min_score():
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
        return config.get("reddit", {}).get("min_score", 50)
    return 50


def scrape_subreddit(subreddit, sort="hot", limit=25, time_filter="week",
                     pages=1, use_cursor=False):
    """
    Scrape a subreddit via Reddit OAuth. Supports pagination via `after` cursor.

    Args:
        limit: Posts per page (capped at 100 by Reddit).
        pages: Number of pages to fetch. Total posts ~ limit x pages.
        use_cursor: If True, persist the pagination cursor across runs so
                    subsequent calls continue where the last one left off.
                    When the cursor is exhausted, automatically resets.
    """
    url = f"https://oauth.reddit.com/r/{subreddit}/{sort}"
    posts = []

    cursor_key = f"{subreddit}|{sort}|{time_filter}"
    if use_cursor:
        import database as db
        after = db.get_cursor("reddit", cursor_key)
    else:
        after = None

    for _ in range(pages):
        params = {"limit": min(limit, 100), "t": time_filter, "raw_json": 1}
        if after:
            params["after"] = after

        resp = requests.get(url, headers=_oauth_headers(), params=params, timeout=15)
        if resp.status_code == 404:
            raise Exception("404 Not Found")
        if resp.status_code == 429:
            time.sleep(5)
            resp = requests.get(url, headers=_oauth_headers(), params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", {})

        children = data.get("children", [])
        if not children:
            break

        for child in children:
            post = child.get("data", {})
            if post.get("stickied"):
                continue
            posts.append({
                "title": post.get("title", ""),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "subreddit": subreddit,
                "url": post.get("url", ""),
                "selftext": (post.get("selftext", "") or "")[:1000],
                "permalink": f"https://reddit.com{post.get('permalink', '')}",
                "author": post.get("author", ""),
                "source": "reddit",
            })

        after = data.get("after")
        if not after:
            break

    if use_cursor:
        import database as db
        db.save_cursor("reddit", cursor_key, after)

    return posts


def scrape_comments(permalink, limit=10):
    """Scrape top comments for a post."""
    path = permalink.replace("https://reddit.com", "").replace("https://www.reddit.com", "")
    url = f"https://oauth.reddit.com{path}"

    resp = requests.get(
        url,
        headers=_oauth_headers(),
        params={"limit": limit, "sort": "top", "depth": 1, "raw_json": 1},
        timeout=15,
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
        c = child.get("data", {})
        comments.append({
            "text": (c.get("body", "") or "")[:500],
            "score": c.get("score", 0),
            "author": c.get("author", ""),
        })
    return sorted(comments, key=lambda x: x["score"], reverse=True)[:limit]


def search_reddit(query, sort="relevance", time_filter="month", limit=100, subreddit=None):
    """Search Reddit via OAuth API."""
    if subreddit:
        url = f"https://oauth.reddit.com/r/{subreddit}/search"
        params = {"q": query, "restrict_sr": "on", "sort": sort, "t": time_filter,
                  "limit": min(limit, 100), "raw_json": 1}
    else:
        url = "https://oauth.reddit.com/search"
        params = {"q": query, "sort": sort, "t": time_filter,
                  "limit": min(limit, 100), "raw_json": 1}

    resp = requests.get(url, headers=_oauth_headers(), params=params, timeout=15)
    if resp.status_code != 200:
        return []

    data = resp.json()
    posts = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        if post.get("stickied"):
            continue
        posts.append({
            "title": post.get("title", ""),
            "score": post.get("score", 0),
            "num_comments": post.get("num_comments", 0),
            "subreddit": post.get("subreddit", ""),
            "url": post.get("url", ""),
            "selftext": (post.get("selftext", "") or "")[:1000],
            "permalink": f"https://reddit.com{post.get('permalink', '')}",
            "source": "reddit_search",
            "search_query": query,
        })
    return posts


def scrape_all_tech_subreddits(limit=10, min_score=None, sort="hot",
                                time_filter="week", pages=1, use_cursor=False):
    """Scrape all configured tech subreddits in parallel via OAuth."""
    subreddits = _load_subreddits()
    if min_score is None:
        min_score = _load_min_score()

    def _scrape_one(sub):
        try:
            posts = scrape_subreddit(sub, sort=sort, limit=limit,
                                     time_filter=time_filter, pages=pages,
                                     use_cursor=use_cursor)
            kept = [p for p in posts if p["score"] >= min_score]
            return sub, kept, None
        except Exception as e:
            return sub, [], e

    all_posts = []
    dead_subs = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_scrape_one, sub): sub for sub in subreddits}
        for future in as_completed(futures):
            sub, posts, error = future.result()
            if error:
                if "404" in str(error):
                    dead_subs.append(sub)
                    print(f"  r/{sub}: NOT FOUND — removing")
                else:
                    print(f"  r/{sub}: failed ({error})")
            else:
                all_posts.extend(posts)
                print(f"  r/{sub}: {len(posts)} posts")

    if dead_subs and CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
        config["reddit"]["subreddits"] = [
            s for s in config["reddit"]["subreddits"]
            if s["name"] not in dead_subs
        ]
        CONFIG_FILE.write_text(json.dumps(config, indent=2))

    return all_posts


# ============================================================
# ADVANCED: LLM-driven queries, discovery, deep scrape
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


QUERY_GEN_PROMPT = """Generate Reddit search queries to find user painpoints, product feedback, and trends.
Based on the DB context, generate queries that find NEW data we don't have yet.

Return JSON:
{"queries": [{"q": "query string", "type": "pain|product|trend|gap", "why": "..."}]}

Rules:
- Use Reddit search syntax: quotes for exact match, OR/AND
- Be specific: "cursor AI tab complete wrong imports" not "AI bad"
- Dig deeper on existing DB painpoints
- Max 15 queries"""


def generate_search_queries(openai_api_key):
    """LLM generates search queries based on DB state."""
    try:
        import database as db
        from openai import OpenAI

        stats = db.get_stats()
        if stats["painpoints"] == 0 and stats["products"] == 0:
            return SEED_QUERIES

        context = "## Current DB:\n"
        painpoints = db.get_top_painpoints(limit=15)
        if painpoints:
            context += "\n### Top painpoints:\n"
            for pp in painpoints:
                context += f"- [{pp['signal_count']}x, sev {pp['severity']}] {pp['title']}: {(pp.get('description') or '')[:100]}\n"

        conn = db.get_db()
        products = [dict(r) for r in conn.execute(
            "SELECT name, description FROM products ORDER BY last_updated DESC LIMIT 15"
        ).fetchall()]
        conn.close()
        if products:
            context += "\n### Known products:\n"
            for p in products:
                context += f"- {p['name']}: {(p.get('description') or '')[:80]}\n"

        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            max_tokens=1500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": QUERY_GEN_PROMPT},
                {"role": "user", "content": context},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        queries = [q["q"] for q in data.get("queries", [])]
        return queries or SEED_QUERIES

    except Exception as e:
        print(f"  [WARN] Query generation failed ({e}), using seeds")
        return SEED_QUERIES


DISCOVER_SUBS_PROMPT = """Generate Reddit search queries to find relevant tech subreddits.

Return JSON:
{
  "search_queries": ["query1", ...],
  "blocklist": ["subreddit_slug_to_ignore", ...]
}

Rules for queries (5-8):
- Target communities we DON'T already scrape
- Tech/product/builder communities only
- Be specific

Rules for blocklist:
- ANY entertainment, memes, politics, news, deals, career advice, resumes, general discussion
- When in doubt, BLOCK IT"""


def discover_subreddits(openai_api_key=None, min_subscribers=5000):
    """LLM-driven subreddit discovery via parallel Reddit search."""
    search_queries = ["I built a developer tool", "open source alternative",
                      "frustrated with coding", "startup launched product"]
    blocklist = set()

    if openai_api_key:
        try:
            import database as db
            from openai import OpenAI

            context = "## Current subreddits:\n"
            if CONFIG_FILE.exists():
                config = json.loads(CONFIG_FILE.read_text())
                for s in config.get("reddit", {}).get("subreddits", []):
                    context += f"- r/{s['name']}\n"

            client = OpenAI(api_key=openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                max_tokens=1000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": DISCOVER_SUBS_PROMPT},
                    {"role": "user", "content": context},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            search_queries = data.get("search_queries", search_queries)
            blocklist = {s.lower() for s in data.get("blocklist", [])}
        except Exception as e:
            print(f"  [WARN] LLM discovery failed ({e}), using defaults")

    found_subs = {}

    def _search_one(query):
        try:
            return search_reddit(query, sort="relevance", time_filter="year", limit=100)
        except Exception:
            return []

    print(f"  Searching {len(search_queries)} queries in parallel...")
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_search_one, q): q for q in search_queries}
        for future in as_completed(futures):
            for p in future.result():
                sub = p.get("subreddit", "")
                if sub and sub.lower() not in blocklist:
                    found_subs[sub] = found_subs.get(sub, 0) + 1

    ranked = sorted(found_subs.items(), key=lambda x: x[1], reverse=True)
    verified = []
    for sub_name, count in ranked[:30]:
        try:
            resp = requests.get(
                f"https://oauth.reddit.com/r/{sub_name}/about",
                headers=_oauth_headers(), timeout=10
            )
            if resp.status_code != 200:
                continue
            about = resp.json().get("data", {})
            subscribers = about.get("subscribers", 0)
            if subscribers < min_subscribers:
                continue
            verified.append({
                "name": sub_name,
                "subscribers": subscribers,
                "description": (about.get("public_description", "") or "")[:200],
                "relevance_score": count,
            })
        except Exception:
            continue

    return verified


def _load_deep_subs():
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
        subs = [s["name"] for s in config.get("reddit", {}).get("subreddits", [])]
        if subs:
            return subs
    return SEED_SUBS


def deep_scrape_reddit(subreddits=None):
    """Deep Reddit scrape via OAuth — parallel threads."""
    subreddits = subreddits or _load_deep_subs()

    def _scrape_one(sub):
        try:
            hot = scrape_subreddit(sub, sort="hot", limit=100)
            top = scrape_subreddit(sub, sort="top", limit=50, time_filter="week")
            seen = {p["permalink"] for p in hot}
            combined = hot + [p for p in top if p["permalink"] not in seen]
            return sub, combined, None
        except Exception as e:
            return sub, [], e

    all_posts = []
    print(f"  Scraping {len(subreddits)} subs in parallel...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_scrape_one, sub): sub for sub in subreddits}
        for future in as_completed(futures):
            sub, posts, error = future.result()
            if error:
                print(f"  r/{sub}: failed ({error})")
            else:
                all_posts.extend(posts)
                print(f"  r/{sub}: {len(posts)} posts")

    top_posts = sorted(all_posts, key=lambda x: x["score"], reverse=True)[:30]

    def _fetch_comments(post):
        try:
            post["top_comments"] = scrape_comments(post["permalink"], limit=5)
        except Exception:
            post["top_comments"] = []
        return post

    print(f"  Fetching comments for top 30 posts...")
    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(_fetch_comments, top_posts))

    all_posts.sort(key=lambda x: x["score"], reverse=True)
    comments_total = sum(len(p.get("top_comments", [])) for p in all_posts)
    print(f"  Reddit total: {len(all_posts)} posts, {comments_total} comments\n")
    return all_posts


def deep_search_painpoints(queries=None):
    """Search Reddit for pain/complaints via OAuth in parallel."""
    queries = queries or SEED_QUERIES

    def _search_one(query):
        try:
            return search_reddit(query, sort="top", time_filter="month", limit=100)
        except Exception as e:
            print(f"    \"{query[:30]}\" failed: {e}")
            return []

    all_posts = []
    seen = set()
    print(f"  Running {len(queries)} searches in parallel...")
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_search_one, q): q for q in queries}
        for future in as_completed(futures):
            for p in future.result():
                key = p.get("permalink", p.get("url", ""))
                if key not in seen:
                    seen.add(key)
                    all_posts.append(p)

    all_posts.sort(key=lambda x: x["score"], reverse=True)
    print(f"  Search total: {len(all_posts)} posts\n")
    return all_posts


def deep_scrape_all(discover_new_subs=True, openai_api_key=None):
    """Run deep scrape across Reddit. All phases parallelized."""
    print("\n🔍 DEEP SCRAPE\n")

    print("  🧠 Generating search queries...")
    search_queries = generate_search_queries(openai_api_key) if openai_api_key else SEED_QUERIES

    if discover_new_subs and openai_api_key:
        print("\n  📡 Discovering subreddits...")
        new_subs = discover_subreddits(openai_api_key=openai_api_key)
        if new_subs:
            print(f"  Found {len(new_subs)} candidate subs (not auto-adding)")

    print("\n  📡 Deep Reddit scrape...")
    reddit_posts = deep_scrape_reddit()

    print("\n  📡 Reddit keyword search...")
    search_posts = deep_search_painpoints(queries=search_queries)

    results = {
        "reddit": reddit_posts,
        "reddit_search": search_posts,
    }
    total = sum(len(v) for v in results.values())
    print(f"\n  DEEP SCRAPE DONE: {total} total posts\n")
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing Reddit OAuth...")
    token = _get_token()
    print(f"Token: {token[:20]}..." if token else "NO TOKEN")

    posts = scrape_all_tech_subreddits(limit=5)
    print(f"\nTotal: {len(posts)} posts")
    for p in posts[:5]:
        print(f"  [{p['score']}] r/{p['subreddit']}: {p['title'][:60]}")
