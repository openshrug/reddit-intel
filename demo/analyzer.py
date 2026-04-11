"""
Scoped single-subreddit analyzer for the public demo.

This is a lighter companion to `ingest.analyze()`:
  - Scrapes exactly one subreddit (hot + top + comments)
  - Saves signals into trends.db like the main pipeline
  - Runs filter + extract passes that see the FULL DB context
    (taxonomy, hot categories, market gaps) so per-sub insights are
    grounded in everything else the system already knows
  - Writes painpoints / products / funding / links to trends.db via
    the same db.upsert_* helpers the ingest pipeline uses
  - Caches the display payload as JSON per subreddit so web re-renders
    don't re-run the LLM

Progress events are streamed via an `on_event` callback so the web UI
can show a live feed while the pipeline runs.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import database as db
from llm import get_client, llm_call, execute_sql_queries, web_search, DB_SCHEMA
from reddit_scraper import scrape_subreddit, scrape_comments


CACHE_DIR = Path(__file__).parent.parent / "demo_cache"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24h


FILTER_PROMPT = """You are filtering posts from a single subreddit to find real signal.

Keep only posts that reveal:
- A specific, actionable painpoint (not vague whining)
- A product, tool, or service people actually use (with traction or failure signal)
- A concrete technical discussion, comparison, or how-to problem
- A genuine market observation (not generic hype)

Drop: memes, news reposts, career advice, self-promo without substance, "look what I made" with no users.

Return JSON:
{
  "kept": [{"idx": <index>, "why": "why this matters", "importance": 1-10}],
  "dropped_count": N,
  "theme_summary": "one sentence describing the dominant themes of this community"
}

importance 8+: clear signal. 5-7: useful context. Below 5: drop it."""


EXTRACT_PROMPT = f"""You are extracting market intelligence from posts in ONE subreddit.

Your output feeds both:
  1. A public dashboard for this subreddit (be specific, quotable, visual)
  2. A cross-community trend intelligence database (be rigorous, linkable)

You have SQL access (SELECT only) and web search. You see the full taxonomy,
current hot categories, and existing market gaps. USE THIS CONTEXT:
  - Before creating a painpoint, check the DB for near-duplicates — reuse titles
    that already exist rather than making new ones whenever the issue is the same.
  - Be honest about product effectiveness. If a tool solves the pain, say so.
    Only flag a gap if the post evidence shows a real failure.
  - Pull the taxonomy slug from the provided list — don't invent slugs.

Each response must be JSON:

- To research:
  {{"done": false, "sql_queries": [...], "web_searches": [...]}}

- To commit:
  {{
    "done": true,
    "community_vibe": "one sentence about who this community is and what they care about",
    "themes": [{{"name": "...", "description": "...", "post_count": N}}],
    "hot_takes": [{{"claim": "...", "context": "why it stands out"}}],
    "painpoints": [
      {{
        "title": "...",
        "description": "...",
        "severity": 1-10,
        "category_slugs": ["..."],
        "quotes": ["verbatim user quote", "..."]
      }}
    ],
    "products": [
      {{
        "name": "...",
        "description": "...",
        "builder": "...",
        "tech_complexity": "LOW|MEDIUM|HIGH",
        "mention_count": N,
        "sentiment": "positive|mixed|negative",
        "context": "how people are using or complaining about it",
        "category_slugs": ["..."]
      }}
    ],
    "funding": [
      {{"company": "...", "amount": "...", "round_type": "...",
        "investors": ["..."], "what_they_build": "...",
        "painpoint_solved": "...", "why_funded": "...", "category_slugs": ["..."]}}
    ],
    "product_painpoint_links": [
      {{"product": "...", "painpoint": "...",
        "relationship": "addresses|fails_at|partial",
        "effectiveness": 1-10,
        "gap_description": "what's still missing",
        "gap_type": "pricing|performance|ux|scope|reliability|integration"}}
    ],
    "proposed_categories": [{{"name": "...", "parent_slug": "...", "reason": "..."}}]
  }}

Be SPECIFIC. "Performance issues" is bad. "Llama 3 70B stalls at 40k context on M3 Max" is good.

Return fewer high-quality items rather than padding with junk. If nothing fits a category, return [].

DB schema: {DB_SCHEMA}"""


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _normalize_name(subreddit):
    return subreddit.strip().removeprefix("r/").removeprefix("/r/").strip("/").lower()


def _cache_path(subreddit):
    return CACHE_DIR / f"{_normalize_name(subreddit)}.json"


def _load_cache(subreddit):
    """Return cached data only if still fresh (TTL honored)."""
    path = _cache_path(subreddit)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        scraped = data.get("scraped_at_unix", 0)
        if time.time() - scraped > CACHE_TTL_SECONDS:
            return None
        return data
    except Exception:
        return None


def load_cache_any(subreddit):
    """Return cached data regardless of TTL. Used by the ideas polling
    endpoint and the insights page so a slightly-stale analysis still
    renders instead of 404-ing on the user."""
    path = _cache_path(subreddit)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_cache(subreddit, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(subreddit).write_text(json.dumps(data, indent=2, default=str))


def _emit(on_event, stage, status, detail=""):
    if on_event:
        try:
            on_event({"stage": stage, "status": status, "detail": detail})
        except Exception:
            pass


@contextmanager
def _heartbeat(on_event, stage, label, interval=2.0):
    """Emit 'tick' events at a steady cadence while a long operation runs,
    so the progress UI never shows a stale line. The frontend replaces the
    most recent tick line in place so the log doesn't flood.
    """
    stop = threading.Event()

    def _ticker():
        t0 = time.time()
        while not stop.wait(interval):
            elapsed = time.time() - t0
            try:
                on_event({
                    "stage": stage,
                    "status": "tick",
                    "detail": f"{label} · {elapsed:0.0f}s",
                })
            except Exception:
                pass

    thread = None
    if on_event:
        thread = threading.Thread(target=_ticker, daemon=True)
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread:
            thread.join(timeout=0.1)


def _as_dicts(value):
    """LLMs occasionally hand back bare strings where a list of dicts was
    requested (e.g. `themes: ["ai", "rust"]`). Return only the dict entries
    so downstream `.get(...)` calls don't explode."""
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict)]


def _sanitize_extracted(data):
    """Coerce every LLM-returned list into a list-of-dicts so the rest of
    the pipeline can safely call `.get()` on its items."""
    if not isinstance(data, dict):
        return {}
    keys = (
        "themes",
        "hot_takes",
        "painpoints",
        "products",
        "funding",
        "product_painpoint_links",
        "proposed_categories",
        "sql_queries",
        "web_searches",
    )
    for k in keys:
        if k in data:
            data[k] = _as_dicts(data.get(k))
    return data


# ============================================================
# SCRAPE
# ============================================================

def _scrape(subreddit, on_event=None):
    """Fetch hot + top posts and top comments for the highest-engagement posts."""
    _emit(on_event, "scrape", "running", f"Fetching hot posts from r/{subreddit}...")
    with _heartbeat(on_event, "scrape", "fetching hot posts"):
        hot = scrape_subreddit(subreddit, sort="hot", limit=100, pages=1)
    _emit(on_event, "scrape", "done", f"got {len(hot)} hot posts")

    _emit(on_event, "scrape", "running", f"Fetching top-of-week from r/{subreddit}...")
    with _heartbeat(on_event, "scrape", "fetching top-of-week"):
        top = scrape_subreddit(subreddit, sort="top", limit=100, time_filter="week", pages=1)
    _emit(on_event, "scrape", "done", f"got {len(top)} top-of-week posts")

    # Dedupe by permalink
    seen = set()
    posts = []
    for p in hot + top:
        key = p.get("permalink") or p.get("url")
        if key and key not in seen:
            seen.add(key)
            posts.append(p)

    _emit(on_event, "scrape", "done", f"deduped to {len(posts)} unique posts")

    # Show the user a taste of what was actually pulled
    top_by_score = sorted(posts, key=lambda p: p.get("score", 0), reverse=True)
    for p in top_by_score[:5]:
        title = (p.get("title") or "").strip()
        if title:
            _emit(on_event, "scrape", "done",
                  f"▲ {p.get('score', 0)} · {title[:120]}")

    # Fetch comments for the most engaged posts
    top_by_engagement = sorted(
        posts,
        key=lambda p: p.get("num_comments", 0) + p.get("score", 0) // 5,
        reverse=True,
    )[:25]

    _emit(on_event, "comments", "running",
          f"Fetching comments on {len(top_by_engagement)} top threads...")

    def _fetch_comments(post):
        try:
            return post, scrape_comments(post["permalink"], limit=8)
        except Exception:
            return post, []

    with _heartbeat(on_event, "comments", "fetching comment threads"):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_fetch_comments, p) for p in top_by_engagement]
            for f in as_completed(futures):
                post, comments = f.result()
                post["comments"] = comments

    total_comments = sum(len(p.get("comments") or []) for p in posts)
    _emit(on_event, "comments", "done", f"got {total_comments} comments")

    # Spoil a couple of real comment snippets
    for post in top_by_engagement[:3]:
        for c in (post.get("comments") or [])[:1]:
            text = (c.get("text") or "").strip().replace("\n", " ")
            if text:
                _emit(on_event, "comments", "done", f"💬 {text[:120]}")

    return posts


# ============================================================
# FORMATTING
# ============================================================

def _format_posts_for_llm(posts, include_comments=True, limit=80):
    """Format posts as numbered items for the LLM filter/extract passes."""
    lines = []
    for idx, p in enumerate(posts[:limit]):
        lines.append(
            f"[{idx}] ({p.get('score', 0)}↑, {p.get('num_comments', 0)}💬) {p.get('title', '')}"
        )
        if p.get("selftext"):
            lines.append(f"    {p['selftext'][:300]}")
        if include_comments and p.get("comments"):
            for c in p["comments"][:3]:
                lines.append(f"    > ({c.get('score', 0)}↑) {c.get('text', '')[:200]}")
    return "\n".join(lines)


def _format_db_context():
    """Build the lightweight DB context block passed to the extract LLM."""
    taxonomy = "\n".join(
        f"- `{c['slug']}` — {c['path']}: {c.get('description') or ''}"
        for c in db.get_category_list_flat()
    )
    stats = db.get_stats()
    hot = db.get_hot_categories(limit=12)
    gaps = db.get_market_gaps(limit=10)

    parts = [
        f"## Taxonomy:\n{taxonomy}",
        f"\n## DB totals: {stats['products']} products, {stats['painpoints']} painpoints, "
        f"{stats['funding_rounds']} funding rounds, {stats['quotes']} quotes",
    ]
    if hot:
        parts.append("\n## Hot categories:")
        for h in hot:
            parts.append(
                f"- {h['category']} ({h['slug']}): {h['painpoints']}pp, {h['products']}prod, {h['total_signals']}sig"
            )
    if gaps:
        parts.append("\n## Market gaps:")
        for g in gaps:
            r = f"{g['pain_to_product_ratio']}:1" if g["pain_to_product_ratio"] < 100 else "∞"
            parts.append(
                f"- {g['category']}: {g['painpoint_count']}pp, {g['product_count']}prod ({r})"
            )
    return "\n".join(parts)


# ============================================================
# LLM PASSES
# ============================================================

def _filter_chunk(client, chunk_posts, chunk_id, on_event):
    """Run the filter prompt on one chunk of posts. Returns (kept_items, theme_summary)."""
    text_lines = []
    for local_idx, p in enumerate(chunk_posts):
        text_lines.append(
            f"[{local_idx}] ({p.get('score', 0)}↑, {p.get('num_comments', 0)}💬) {p.get('title', '')}"
        )
        if p.get("selftext"):
            text_lines.append(f"    {p['selftext'][:300]}")
    text = "\n".join(text_lines)

    messages = [
        {"role": "system", "content": FILTER_PROMPT},
        {"role": "user", "content": f"Posts to filter (chunk {chunk_id}):\n\n{text}"},
    ]

    try:
        with _heartbeat(on_event, "filter", f"chunk {chunk_id} thinking"):
            data = json.loads(llm_call(client, messages, max_tokens=2000))
    except Exception as e:
        _emit(on_event, "filter", "error", f"chunk {chunk_id}: {e}")
        raise RuntimeError(f"Filter chunk {chunk_id} failed: {e}") from e

    if not isinstance(data, dict):
        raise RuntimeError(f"Filter chunk {chunk_id} returned non-object")

    kept_items = _as_dicts(data.get("kept"))
    dropped = data.get("dropped_count", len(chunk_posts) - len(kept_items))
    theme_summary = data.get("theme_summary") if isinstance(data.get("theme_summary"), str) else ""

    _emit(on_event, "filter", "done",
          f"chunk {chunk_id}: kept {len(kept_items)}, dropped {dropped}")

    # Surface the most interesting reasoning from this chunk
    for item in sorted(kept_items, key=lambda x: x.get("importance", 0) or 0, reverse=True)[:2]:
        why = item.get("why") if isinstance(item.get("why"), str) else ""
        importance = item.get("importance", "?")
        if why:
            _emit(on_event, "filter", "done", f"[{importance}/10] {why[:140]}")

    return kept_items, theme_summary


def _filter_pass(client, posts, on_event=None):
    """Parallel filter pass — splits posts into chunks so users see progress
    and we burn multiple LLM calls concurrently instead of one slow blocking one."""
    CHUNK_SIZE = 25
    chunks = [posts[i:i + CHUNK_SIZE] for i in range(0, len(posts), CHUNK_SIZE)]

    _emit(on_event, "filter", "running",
          f"filtering {len(posts)} posts in {len(chunks)} parallel chunks...")

    # Track (chunk_id, chunk_posts) → (kept_items, theme_summary)
    kept_all = []  # list of (global_post, item_meta)
    themes = []

    with ThreadPoolExecutor(max_workers=min(6, len(chunks))) as pool:
        future_to_chunk = {
            pool.submit(_filter_chunk, client, chunk, i + 1, on_event): (i, chunk)
            for i, chunk in enumerate(chunks)
        }
        for f in as_completed(future_to_chunk):
            chunk_idx, chunk_posts = future_to_chunk[f]
            kept_items, theme_summary = f.result()
            if theme_summary:
                themes.append(theme_summary)
            for item in kept_items:
                local_idx = item.get("idx")
                if isinstance(local_idx, int) and 0 <= local_idx < len(chunk_posts):
                    kept_all.append((chunk_posts[local_idx], item))

    kept = [p for p, _ in kept_all]
    merged_theme = " · ".join(themes[:3])

    _emit(on_event, "filter", "done",
          f"total: kept {len(kept)} / {len(posts)}")
    if merged_theme:
        _emit(on_event, "filter", "done", f"theme · {merged_theme[:200]}")

    return kept, merged_theme


def _extract_pass(client, subreddit, posts, theme_summary, on_event=None, max_iterations=4):
    """LLM research loop that extracts insights with full DB context.
    Returns the final committed payload (JSON dict)."""
    _emit(on_event, "extract", "running",
          "LLM extracting insights (using DB context)...")

    db_context = _format_db_context()
    posts_text = _format_posts_for_llm(posts, include_comments=True, limit=60)

    user_content = (
        f"Analyzing subreddit: r/{subreddit}\n"
        f"{('Filter theme summary: ' + theme_summary) if theme_summary else ''}\n\n"
        f"{db_context}\n\n"
        f"## Posts from r/{subreddit}:\n\n{posts_text}\n\n"
        "Research with SQL or web if you need to, then commit (done:true)."
    )

    messages = [
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": user_content},
    ]

    data = {}
    for iteration in range(max_iterations):
        try:
            with _heartbeat(on_event, "extract", f"iter {iteration + 1} thinking"):
                raw = json.loads(llm_call(client, messages))
        except Exception as e:
            _emit(on_event, "extract", "error", f"Extract failed: {e}")
            raise RuntimeError(f"Extract pass failed: {e}") from e
        data = _sanitize_extracted(raw)

        sql_queries = data.get("sql_queries", [])
        web_searches = data.get("web_searches", [])
        is_done = data.get("done", True) or (
            not sql_queries
            and not web_searches
            and any(data.get(k) for k in ("painpoints", "products", "funding"))
        )

        if is_done:
            pp_list = data.get("painpoints", [])
            pr_list = data.get("products", [])
            th_list = data.get("themes", [])
            _emit(on_event, "extract", "done",
                  f"{len(pp_list)} painpoints, {len(pr_list)} products, {len(th_list)} themes")
            # Spoil a few specific findings in the progress log
            for pp in pp_list[:3]:
                title = pp.get("title") if isinstance(pp, dict) else None
                if title:
                    _emit(on_event, "extract", "done", f"pain → {title}")
            for pr in pr_list[:3]:
                name = pr.get("name") if isinstance(pr, dict) else None
                if name:
                    _emit(on_event, "extract", "done", f"product → {name}")
            return data

        research_parts = []
        if sql_queries:
            _emit(on_event, "extract", "running",
                  f"iter {iteration + 1}: model is running {len(sql_queries)} SQL queries")
            for sq in sql_queries[:4]:
                reason = sq.get("reason", "") if isinstance(sq, dict) else ""
                if reason:
                    _emit(on_event, "extract", "running", f"sql · {reason}")
            research_parts.append(
                "SQL results:\n\n" + "\n\n".join(execute_sql_queries(sql_queries))
            )
        if web_searches:
            _emit(on_event, "extract", "running",
                  f"iter {iteration + 1}: model wants {len(web_searches)} web searches")
            for ws in web_searches[:3]:
                q = ws.get("query", "") if isinstance(ws, dict) else ""
                reason = ws.get("reason", "") if isinstance(ws, dict) else ""
                _emit(on_event, "extract", "running", f"web · {reason or q}")
                with _heartbeat(on_event, "extract", f"searching · {(reason or q)[:60]}"):
                    result = web_search(client, q)
                research_parts.append(f"Web search ({reason}): {q}\nResult:\n{result[:1500]}")

        messages.append({"role": "assistant", "content": json.dumps(data)})
        messages.append({"role": "user", "content": "\n\n".join(research_parts)})

    _emit(on_event, "extract", "done", "Reached max iterations")
    return data if isinstance(data, dict) else {}


# ============================================================
# DB WRITES
# ============================================================

def _save_to_db(subreddit, raw_posts, filtered_posts, extracted, run_id):
    """Write the LLM output into trends.db, tracking which IDs came from this run."""
    painpoint_ids = []
    product_ids = []
    funding_ids = []

    for p in extracted.get("products", []) or []:
        if not p.get("name"):
            continue
        pid = db.upsert_product(
            p["name"],
            description=p.get("description"),
            builder=p.get("builder"),
            tech_complexity=p.get("tech_complexity"),
            viral_trigger=p.get("viral_trigger"),
            why_viral=p.get("why_viral"),
        )
        if pid:
            product_ids.append(pid)
            for slug in p.get("category_slugs", []) or []:
                db.link_product_category(pid, slug)

    for pp in extracted.get("painpoints", []) or []:
        if not pp.get("title"):
            continue
        pid = db.upsert_painpoint(
            pp["title"],
            description=pp.get("description", ""),
            severity=pp.get("severity", 5),
        )
        if pid:
            painpoint_ids.append(pid)
            for slug in pp.get("category_slugs", []) or []:
                db.link_painpoint_category(pid, slug)
            for q in pp.get("quotes", []) or []:
                if q and len(q.strip()) > 10:
                    db.add_quote(pid, q, f"reddit_demo:{subreddit}")

    for f in extracted.get("funding", []) or []:
        if not f.get("company"):
            continue
        rid = db.save_funding_round(
            f["company"],
            amount=f.get("amount", ""),
            round_type=f.get("round_type", ""),
            investor_names=f.get("investors"),
            what_they_build=f.get("what_they_build", ""),
            painpoint_solved=f.get("painpoint_solved", ""),
            why_funded=f.get("why_funded", ""),
        )
        if rid:
            funding_ids.append(rid)
            for slug in f.get("category_slugs", []) or []:
                db.link_funding_category(rid, slug)

    for link in extracted.get("product_painpoint_links", []) or []:
        if link.get("product") and link.get("painpoint"):
            db.link_product_painpoint(
                link["product"],
                link["painpoint"],
                relationship=link.get("relationship", "addresses"),
                effectiveness=link.get("effectiveness"),
                gap_description=link.get("gap_description", ""),
                gap_type=link.get("gap_type", ""),
                notes=link.get("notes", ""),
            )

    for cat in extracted.get("proposed_categories", []) or []:
        if cat.get("name"):
            db.propose_category(cat["name"], cat.get("parent_slug"))

    return {
        "painpoint_ids": painpoint_ids,
        "product_ids": product_ids,
        "funding_ids": funding_ids,
    }


# ============================================================
# CROSS-REFERENCE
# ============================================================

def _cross_reference(painpoint_ids, product_ids):
    """Look up cross-community context for the painpoints/products extracted
    from this subreddit. Shows how this community's concerns connect to the
    broader intelligence DB."""
    if not painpoint_ids and not product_ids:
        return {}

    conn = db.get_db()
    cross = {"similar_painpoints": [], "funded_solutions": [], "failing_products": []}

    if painpoint_ids:
        placeholders = ",".join("?" * len(painpoint_ids))
        rows = conn.execute(
            f"""
            SELECT pp.id, pp.title, pp.signal_count, pp.severity,
                   GROUP_CONCAT(DISTINCT c.name) AS categories
            FROM painpoints pp
            LEFT JOIN painpoint_categories pc ON pc.painpoint_id = pp.id
            LEFT JOIN categories c ON c.id = pc.category_id
            WHERE pp.id IN ({placeholders})
            GROUP BY pp.id
            ORDER BY pp.signal_count DESC
            """,
            painpoint_ids,
        ).fetchall()
        cross["similar_painpoints"] = [dict(r) for r in rows]

        # Funded solutions that address these painpoints
        rows = conn.execute(
            f"""
            SELECT DISTINCT
                COALESCE(p.name, '(unknown)') AS company,
                fr.amount,
                fr.what_they_build
            FROM funding_rounds fr
            LEFT JOIN products p ON p.id = fr.product_id
            JOIN round_painpoints rp ON rp.round_id = fr.id
            WHERE rp.painpoint_id IN ({placeholders})
            LIMIT 10
            """,
            painpoint_ids,
        ).fetchall()
        cross["funded_solutions"] = [dict(r) for r in rows]

        # Products that FAIL at these painpoints
        rows = conn.execute(
            f"""
            SELECT p.name, pp.title AS painpoint, ppp.effectiveness, ppp.gap_description, ppp.gap_type
            FROM product_painpoints ppp
            JOIN products p ON p.id = ppp.product_id
            JOIN painpoints pp ON pp.id = ppp.painpoint_id
            WHERE ppp.painpoint_id IN ({placeholders})
              AND ppp.relationship IN ('fails_at', 'partial')
            ORDER BY ppp.effectiveness ASC
            LIMIT 10
            """,
            painpoint_ids,
        ).fetchall()
        cross["failing_products"] = [dict(r) for r in rows]

    conn.close()
    return cross


# ============================================================
# STATS
# ============================================================

def _compute_stats(raw_posts, filtered_posts):
    """Aggregate numeric stats for the dashboard hero strip."""
    authors = {
        p.get("author")
        for p in raw_posts
        if p.get("author") and p.get("author") != "[deleted]"
    }
    total_comments = sum(len(p.get("comments") or []) for p in raw_posts)
    total_score = sum(p.get("score", 0) for p in raw_posts)
    top_posts = sorted(raw_posts, key=lambda p: p.get("score", 0), reverse=True)[:10]
    return {
        "posts_scraped": len(raw_posts),
        "comments_scraped": total_comments,
        "unique_users": len(authors),
        "signal_kept": len(filtered_posts),
        "total_score": total_score,
        "top_posts": [
            {
                "title": p.get("title", ""),
                "score": p.get("score", 0),
                "num_comments": p.get("num_comments", 0),
                "permalink": p.get("permalink", ""),
                "author": p.get("author", ""),
            }
            for p in top_posts
        ],
    }


# ============================================================
# PUBLIC API
# ============================================================

def analyze_subreddit(subreddit, on_event=None, force_refresh=False):
    """
    Scrape and analyze a single subreddit end-to-end.

    Writes to trends.db like the main pipeline, but also caches the
    display payload as JSON so web re-renders don't re-run the LLM.
    """
    subreddit = _normalize_name(subreddit)
    if not subreddit:
        raise ValueError("subreddit name cannot be empty")

    if not force_refresh:
        cached = _load_cache(subreddit)
        if cached:
            _emit(on_event, "cache", "done", "Loaded from cache")
            _emit(on_event, "done", "done", "Analysis complete (cached)")
            cached["cached"] = True
            return cached

    client = get_client()
    _emit(on_event, "start", "running", f"Analyzing r/{subreddit}...")

    # 1. Scrape
    raw_posts = _scrape(subreddit, on_event=on_event)
    if not raw_posts:
        _emit(on_event, "error", "error",
              "No posts found — subreddit may be private or empty")
        raise RuntimeError(f"No posts found for r/{subreddit}")

    # 2. Save raw signals to DB (lets them show up in future intelligence runs)
    run_id = db.start_run([f"reddit_demo:{subreddit}"])
    db.save_signals(run_id, f"reddit_demo:{subreddit}", raw_posts)
    db.compute_percentiles(run_id)

    # 3. Filter
    filtered, theme_summary = _filter_pass(client, raw_posts, on_event=on_event)
    if not filtered:
        filtered = sorted(raw_posts, key=lambda p: p.get("score", 0), reverse=True)[:30]

    # 4. Extract (LLM sees DB context + scoped posts; writes back to DB)
    extracted = _extract_pass(
        client, subreddit, filtered, theme_summary, on_event=on_event
    )

    saved_ids = _save_to_db(subreddit, raw_posts, filtered, extracted, run_id)
    db.finish_run(run_id, len(raw_posts), len(filtered))

    # 5. Cross-reference against the rest of the DB
    _emit(on_event, "cross_ref", "running", "Cross-referencing with broader database...")
    cross = _cross_reference(saved_ids["painpoint_ids"], saved_ids["product_ids"])
    _emit(on_event, "cross_ref", "done",
          f"Found {len(cross.get('funded_solutions', []))} related funded startups")

    # 6. Pack display payload
    stats = _compute_stats(raw_posts, filtered)

    result = {
        "subreddit": subreddit,
        "scraped_at": _now_iso(),
        "scraped_at_unix": time.time(),
        "run_id": run_id,
        "stats": stats,
        "theme_summary": theme_summary,
        "community_vibe": extracted.get("community_vibe", ""),
        "themes": extracted.get("themes", []),
        "painpoints": extracted.get("painpoints", []),
        "products": extracted.get("products", []),
        "hot_takes": extracted.get("hot_takes", []),
        "funding": extracted.get("funding", []),
        "cross_reference": cross,
        "saved_ids": saved_ids,
        "cached": False,
    }

    _save_cache(subreddit, result)
    _emit(on_event, "done", "done", "Analysis complete")
    return result


def list_cached():
    """Return a list of cached subreddit analyses for the landing page grid."""
    if not CACHE_DIR.exists():
        return []
    results = []
    for path in sorted(CACHE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            results.append(
                {
                    "subreddit": data.get("subreddit", path.stem),
                    "scraped_at": data.get("scraped_at", ""),
                    "posts": data.get("stats", {}).get("posts_scraped", 0),
                    "painpoints": len(data.get("painpoints", [])),
                    "products": len(data.get("products", [])),
                    "community_vibe": data.get("community_vibe", ""),
                }
            )
        except Exception:
            continue
    return sorted(results, key=lambda r: r["scraped_at"], reverse=True)


# ============================================================
# IDEA GENERATION (runs in background while user reads painpoints)
# ============================================================

# Live in-memory log of idea-generation events, keyed by normalized subreddit.
# Written by the background thread, read by the /api polling endpoint.
_IDEAS_LOGS: dict = {}
_IDEAS_LOGS_LOCK = threading.Lock()


def get_ideas_log(subreddit):
    """Return the current event log for this subreddit. Returns the live
    in-memory log while generation is running, or the persisted log from
    cache after it finishes."""
    subreddit = _normalize_name(subreddit)
    with _IDEAS_LOGS_LOCK:
        live = _IDEAS_LOGS.get(subreddit)
        if live is not None:
            return list(live)
    cache = load_cache_any(subreddit)
    return (cache or {}).get("ideas_log") or []


def mark_ideas_pending(subreddit):
    """Flip the cache file into 'ideas are running' state so the UI can show
    a generating placeholder while the background thread works."""
    cache = load_cache_any(subreddit)
    if not cache:
        return False
    cache["ideas_status"] = "running"
    cache["ideas"] = cache.get("ideas") or []
    cache["ideas_error"] = None
    cache["ideas_started_at"] = _now_iso()
    _save_cache(subreddit, cache)
    return True


def generate_ideas_for_cached(subreddit, count=3):
    """Load the cached analysis for a subreddit, run propose_idea scoped to
    its painpoint IDs, write the results back into the cache. Meant to be
    called from a background thread right after analyze_subreddit() finishes
    so that by the time the user opens /r/{sub} the ideas are already baking.
    """
    # Lazy import to avoid circular dependency with ideas.py at module load
    from ideas import propose_idea

    subreddit = _normalize_name(subreddit)
    cache = load_cache_any(subreddit)
    if not cache:
        return []

    painpoint_ids = (cache.get("saved_ids") or {}).get("painpoint_ids") or []
    if not painpoint_ids:
        cache["ideas_status"] = "error"
        cache["ideas_error"] = "No painpoint IDs were captured from this run."
        _save_cache(subreddit, cache)
        return []

    # Reset the live event log for this run
    with _IDEAS_LOGS_LOCK:
        _IDEAS_LOGS[subreddit] = []

    def collect(ev):
        with _IDEAS_LOGS_LOCK:
            log = _IDEAS_LOGS.get(subreddit)
            if log is not None:
                log.append({
                    "stage": ev.get("stage", ""),
                    "status": ev.get("status", ""),
                    "detail": ev.get("detail", ""),
                })

    try:
        ideas = propose_idea(
            painpoint_ids=painpoint_ids,
            count=count,
            check_competition=False,  # slow + noisy for a web demo
            max_iterations=3,
            on_event=collect,
        )
    except Exception as e:
        with _IDEAS_LOGS_LOCK:
            final_log = list(_IDEAS_LOGS.get(subreddit) or [])
            _IDEAS_LOGS.pop(subreddit, None)
        cache = load_cache_any(subreddit) or cache
        cache["ideas_status"] = "error"
        cache["ideas_error"] = str(e)
        cache["ideas_log"] = final_log
        _save_cache(subreddit, cache)
        return []

    # Persist the final log and clear the live store
    with _IDEAS_LOGS_LOCK:
        final_log = list(_IDEAS_LOGS.get(subreddit) or [])
        _IDEAS_LOGS.pop(subreddit, None)

    cache = load_cache_any(subreddit) or cache
    cache["ideas"] = ideas
    cache["ideas_status"] = "done"
    cache["ideas_error"] = None
    cache["ideas_generated_at"] = _now_iso()
    cache["ideas_log"] = final_log
    _save_cache(subreddit, cache)
    return ideas
