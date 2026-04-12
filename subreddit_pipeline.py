"""
Subreddit analysis pipeline.

Scrape one subreddit, persist posts+comments to DB, then run LLM
extraction to identify painpoints.

    await analyze("ExperiencedDevs")
"""

import logging

import db
from db.posts import upsert_post, upsert_comment
from extractor import extract_painpoints
from reddit_scraper import scrape_subreddit_full

log = logging.getLogger(__name__)


async def analyze(subreddit, *, min_score=None):
    """Full pipeline for one subreddit: scrape -> persist -> extract.

    Returns a summary dict with counts for each stage.
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

    pending_ids = await extract_painpoints(list(post_id_map.values()))

    return {
        "subreddit": subreddit,
        "posts_scraped": len(posts),
        "posts_persisted": len(post_id_map),
        "comments_persisted": comments_count,
        "painpoints_extracted": len(pending_ids),
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
