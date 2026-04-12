"""
Subreddit analysis pipeline.

Scrape one subreddit, persist posts+comments to DB, then run LLM
extraction (stubbed for now).

    await analyze("ExperiencedDevs")
"""

import logging

import db
from db.posts import upsert_post, upsert_comment
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

    pending_ids = extract_painpoints(post_id_map)

    return {
        "subreddit": subreddit,
        "posts_scraped": len(posts),
        "posts_persisted": len(post_id_map),
        "comments_persisted": comments_count,
        "painpoints_extracted": len(pending_ids),
    }


def extract_painpoints(post_id_map):
    """Extract painpoints from persisted posts via LLM.

    Stub — returns an empty list.  The real implementation will read
    posts+comments from the DB, batch them, call the LLM, and write
    to pending_painpoints.
    """
    log.info("extract_painpoints: not implemented yet, skipping")
    return []


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
