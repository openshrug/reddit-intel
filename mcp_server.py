"""
Reddit Intel MCP Server.

Exposes the reddit-intel database (painpoints, categories, posts, comments)
and scraping capabilities as MCP tools + resources for any MCP-compatible
agent (OpenClaw, Claude Code, Cursor, etc.).

    reddit-intel-mcp              # installed console script (preferred)
    python mcp_server.py          # equivalent direct invocation
    python -m mcp_server          # equivalent module invocation
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

import db
import reddit_scraper
import subreddit_pipeline
import subriff_scraper
from db import queries
from db.categories import get_category_list_flat
from db.posts import get_comments_for_post, get_posts_by_ids

load_dotenv()

log = logging.getLogger(__name__)

mcp = FastMCP(
    "reddit-intel",
    instructions=(
        "Reddit painpoint intelligence DB.\n"
        "Default flow:\n"
        "  1. Read reddit-intel://stats + reddit-intel://taxonomy to ground yourself.\n"
        "  2. Browse with get_top_painpoints / get_painpoint / get_painpoint_evidence.\n"
        "  3. Call scrape_subreddit ONLY for fresh data (slow, costs API quota).\n"
        "Escape hatch: run_sql for ad-hoc SELECTs — read reddit-intel://schema first."
    ),
)


# ============================================================
# Read tools — Core
# ============================================================


@mcp.tool
def get_stats() -> dict:
    """Global DB counts (posts, comments, painpoints, categories, subreddits,
    unmerged pending). Use mid-task when you need an up-to-the-second count
    (e.g. right after scrape_subreddit). For passive session-start context,
    prefer the reddit-intel://stats resource — same data, no tool call cost."""
    return queries.get_stats()


@mcp.tool
def list_categories() -> list[dict]:
    """Full taxonomy as {path, name, description} entries. Use mid-task right
    before calling get_top_painpoints(category=...) so you pass a real name.
    For passive session-start grounding, prefer the reddit-intel://taxonomy
    resource — same data, no tool call cost."""
    return get_category_list_flat()


@mcp.tool
def get_top_painpoints(
    limit: int = 20,
    category: str | None = None,
    subreddit: str | None = None,
) -> list[dict]:
    """LIST VIEW of painpoints, ranked by signal_count then severity. Start
    broad (no filters), then narrow with category or subreddit. Use this when
    the user wants individual painpoints they can browse or drill into. For the
    aggregate shape of a subreddit (counts, top categories), call
    get_subreddit_summary instead."""
    if category:
        return queries.get_painpoints_by_category(category, limit)
    if subreddit:
        return queries.get_painpoints_by_subreddit(subreddit, limit)
    return queries.get_top_painpoints(limit)


@mcp.tool
def get_painpoint(painpoint_id: int) -> dict | None:
    """Single merged painpoint row, no evidence. Cheap lookup when you already
    have an ID and just need to remember what it is."""
    conn = db.get_db()
    try:
        row = conn.execute(
            "SELECT p.*, c.name AS category "
            "FROM painpoints p LEFT JOIN categories c ON c.id = p.category_id "
            "WHERE p.id = ?",
            (painpoint_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


@mcp.tool
def get_painpoint_evidence(painpoint_id: int, limit: int = 50) -> dict:
    """Raw Reddit posts/comments backing a painpoint (post titles, comment
    bodies, scores, permalinks). Always limited (default 50) — use a small
    limit (5-10) when you only need representative quotes."""
    rows = queries.get_painpoint_evidence(painpoint_id)
    return {
        "total_count": len(rows),
        "evidence": rows[:limit],
        "truncated": len(rows) > limit,
    }


@mcp.tool
def get_subreddit_summary(subreddit: str) -> dict:
    """AGGREGATE VIEW of a subreddit: post count, comment count, painpoint
    count, top 10 categories. Returns one summary dict, never individual
    painpoints. Use whenever the user wants the shape of r/X or as a pre-check
    before scrape_subreddit. For the actual list of painpoints in r/X, call
    get_top_painpoints(subreddit='X') instead."""
    return queries.get_subreddit_summary(subreddit)


# ============================================================
# Read tools — Helpful
# ============================================================


@mcp.tool
def get_post(post_id: int) -> dict | None:
    """Full post + its comments. Use when evidence surfaces a thread worth
    drilling into."""
    posts = get_posts_by_ids([post_id])
    if post_id not in posts:
        return None
    return {**posts[post_id], "comments": get_comments_for_post(post_id)}


# ============================================================
# Read tools — Escape hatch
# ============================================================


@mcp.tool
def run_sql(query: str) -> list[dict] | dict:
    """LAST-RESORT escape hatch for ad-hoc SELECT queries. Prefer the typed
    tools (list_categories, get_top_painpoints, get_painpoint,
    get_painpoint_evidence, get_subreddit_summary, get_post) whenever they
    cover the question — they are cleaner, faster, and cheaper. Read the
    reddit-intel://schema resource before writing a query. SELECT only;
    non-SELECT returns an error."""
    return queries.run_sql(query)


# ============================================================
# Write tools (async)
# ============================================================


@mcp.tool
async def scrape_subreddit(
    subreddit: str, min_score: int | None = None
) -> dict:
    """SLOW (30s-5min): scrape -> persist -> LLM extract -> promote. Costs
    Reddit + OpenAI quota. Use only when fresh data is needed. Idempotent:
    already-extracted posts are skipped."""
    return await subreddit_pipeline.analyze(subreddit, min_score=min_score)


@mcp.tool
async def search_reddit(
    query: str,
    subreddit: str | None = None,
    sort: str = "relevance",
    time_filter: str = "month",
    limit: int = 25,
) -> list[dict]:
    """Reddit search; does NOT persist to DB. Cheap discovery tool — pair with
    scrape_subreddit when you find an unscraped sub worth ingesting."""
    sem = asyncio.Semaphore(reddit_scraper.CONCURRENT_REQUESTS)
    async with httpx.AsyncClient(
        headers=reddit_scraper._oauth_headers(), timeout=15.0
    ) as client:
        return await reddit_scraper.search_reddit(
            client,
            sem,
            query,
            sort=sort,
            time_filter=time_filter,
            limit=limit,
            subreddit=subreddit,
        )


@mcp.tool
def find_trending_subreddits(
    min_size: int = 1000, limit: int = 20, mode: str = "growing"
) -> list[dict]:
    """Discover fast-growing subreddits via Subriff. mode='growing' sorts by
    weekly subscriber growth; mode='new' surfaces recently-created subs by
    daily growth %. Use when sourcing candidates for scrape_subreddit."""
    if mode == "new":
        return subriff_scraper.scrape_new_and_growing(
            min_size=min_size, limit=limit
        )
    return subriff_scraper.scrape_fastest_growing(
        min_size=min_size, limit=limit
    )


# ============================================================
# Resources
# ============================================================


@mcp.resource("reddit-intel://schema")
def db_schema() -> str:
    """DB schema. Read before writing run_sql queries."""
    return (Path(__file__).parent / "db" / "schema.sql").read_text()


@mcp.resource("reddit-intel://stats")
def db_stats() -> dict:
    """Current DB stats snapshot — cheap warm-check."""
    return queries.get_stats()


@mcp.resource("reddit-intel://taxonomy")
def db_taxonomy() -> str:
    """Live category taxonomy as 'Parent > Child: description' lines."""
    cats = get_category_list_flat()
    return (
        "\n".join(
            f"- {c['path']}: {c['description'] or '(no description)'}"
            for c in cats
        )
        or "- Uncategorized"
    )


# ============================================================
# Startup helpers
# ============================================================


def _check_credentials():
    if not (os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")):
        log.warning(
            "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set — "
            "scrape_subreddit and search_reddit will fail at call time. "
            "Create a Reddit script app at https://www.reddit.com/prefs/apps "
            "and set the env vars."
        )
    if not os.getenv("OPENAI_API_KEY"):
        log.warning(
            "OPENAI_API_KEY not set — scrape_subreddit's LLM extraction "
            "and embedding-promotion stages will fail."
        )


def run():
    """Entry point for the `reddit-intel-mcp` console script.

    Initializes logging + DB, warns on missing creds, then starts the MCP
    server on stdio transport. Safe to call from any cwd after
    `pip install -e .[mcp]`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    db.init_db()
    _check_credentials()
    mcp.run()


if __name__ == "__main__":
    run()
