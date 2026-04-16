"""
Subreddit analysis pipeline.

Scrape one subreddit, persist posts+comments to DB, run LLM extraction
to identify painpoints, then promote the pending painpoints into the
merged table via embedding cosine similarity.

    await analyze("ExperiencedDevs")
"""

import logging

import db
from db.posts import upsert_comment, upsert_post
from painpoint_extraction import extract_painpoints
from reddit_scraper import scrape_subreddit_full

log = logging.getLogger(__name__)


async def analyze(subreddit, *, min_score=None):
    """Full pipeline for one subreddit: scrape -> persist -> extract -> promote.

    Returns a summary dict with counts for each stage. If the promote
    stage fails (e.g. OpenAI embeddings API is down), the pending rows
    stay in `pending_painpoints` for the next promoter run — no crash,
    no data loss.
    """
    db.init_db()

    log.info("pipeline: scraping r/%s", subreddit)
    posts = await scrape_subreddit_full(subreddit, min_score=min_score)
    log.info("pipeline: scraped %d posts from r/%s", len(posts), subreddit)

    log.info("pipeline: persisting to DB")
    post_id_map = _persist_scrape(posts)
    comments_count = sum(len(p.get("comments", [])) for p in posts)
    log.info("pipeline: persisted %d posts, %d comments",
             len(post_id_map), comments_count)

    pending_ids, token_usage = await extract_painpoints(list(post_id_map.values()))
    log.info("pipeline: extracted %d pending painpoints", len(pending_ids))

    # Stage 4: promote pending painpoints into the merged table via
    # embedding cosine similarity. Any failure (API error, network, etc.)
    # is caught — the pending rows stay in the queue for the next
    # promoter run to pick up.
    promoted = {"processed": 0, "linked": 0, "error": None}
    try:
        from db.embeddings import OpenAIEmbedder
        from promoter import run_once

        log.info("pipeline: promoting pending painpoints")
        promoted = run_once(embedder=OpenAIEmbedder())
        promoted["error"] = None
        log.info("pipeline: promoted %d (of %d processed)",
                 promoted["linked"], promoted["processed"])
    except Exception as e:
        log.warning(
            "pipeline: promotion stage failed (%s: %s) — pending painpoints "
            "stay in queue for next promoter run",
            type(e).__name__, e,
        )
        promoted["error"] = f"{type(e).__name__}: {e}"

    return {
        "subreddit": subreddit,
        "posts_scraped": len(posts),
        "posts_persisted": len(post_id_map),
        "comments_persisted": comments_count,
        "painpoints_extracted": len(pending_ids),
        "painpoints_linked": promoted["linked"],
        "promote_error": promoted["error"],
        "token_usage": token_usage,
    }


def _persist_scrape(posts):
    """Write scraped posts and their comments into the DB.

    Returns:
        Dict mapping Reddit fullname (e.g. "t3_abc") to the internal
        posts.id for every post that was persisted.
    """
    id_map = {}
    for post in posts:
        post_id = upsert_post(post)
        reddit_name = post.get("name", "")
        id_map[reddit_name] = post_id

        for comment in post.get("comments", []):
            upsert_comment(post_id, comment)

    return id_map
