import json
import sqlite3

from . import _now, get_db


def upsert_post(post_data):
    """Insert a Reddit post if not already stored (dedup by reddit_id).

    Args:
        post_data: dict with keys from the Reddit API (title, score, subreddit,
                   url, selftext, permalink, author, etc.).

    Returns:
        Internal post id (int).
    """
    conn = get_db()
    now = _now()

    reddit_id = post_data.get("name") or post_data.get("reddit_id", "")
    permalink = post_data.get("permalink", "")
    if permalink and not permalink.startswith("http"):
        permalink = f"https://reddit.com{permalink}"

    extra_fields = {
        k: v
        for k, v in post_data.items()
        if k not in {
            "name", "reddit_id", "subreddit", "title", "selftext", "url",
            "author", "score", "upvote_ratio", "num_comments", "permalink",
            "created_utc", "is_self", "link_flair_text", "link_flair", "stickied",
        }
    }

    try:
        conn.execute(
            """INSERT INTO posts
               (reddit_id, subreddit, title, selftext, url, author, score,
                upvote_ratio, num_comments, permalink, created_utc, is_self,
                link_flair, stickied, extra, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                reddit_id,
                post_data.get("subreddit", ""),
                post_data.get("title", ""),
                (post_data.get("selftext", "") or "")[:10000],
                post_data.get("url", ""),
                post_data.get("author", ""),
                post_data.get("score", 0),
                post_data.get("upvote_ratio"),
                post_data.get("num_comments", 0),
                permalink,
                post_data.get("created_utc"),
                1 if post_data.get("is_self") else 0,
                post_data.get("link_flair_text") or post_data.get("link_flair", ""),
                1 if post_data.get("stickied") else 0,
                json.dumps(extra_fields) if extra_fields else None,
                now,
            ),
        )
        pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    except sqlite3.IntegrityError:
        row = conn.execute(
            "SELECT id FROM posts WHERE reddit_id = ?", (reddit_id,)
        ).fetchone()
        pid = row["id"]
    conn.commit()
    conn.close()
    return pid


def upsert_comment(post_id, comment_data):
    """Insert a Reddit comment if not already stored (dedup by reddit_id).

    Args:
        post_id: Internal post id (FK to posts.id).
        comment_data: dict with keys from the Reddit API (body, score, etc.).

    Returns:
        Internal comment id (int).
    """
    conn = get_db()
    now = _now()

    reddit_id = comment_data.get("name") or comment_data.get("reddit_id", "")
    permalink = comment_data.get("permalink", "")
    if permalink and not permalink.startswith("http"):
        permalink = f"https://reddit.com{permalink}"

    try:
        conn.execute(
            """INSERT INTO comments
               (reddit_id, post_id, parent_reddit_id, author, body, score,
                controversiality, permalink, created_utc, depth, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                reddit_id,
                post_id,
                comment_data.get("parent_id") or comment_data.get("parent_reddit_id"),
                comment_data.get("author", ""),
                comment_data.get("body", ""),
                comment_data.get("score", 0),
                comment_data.get("controversiality", 0),
                permalink,
                comment_data.get("created_utc"),
                comment_data.get("depth", 0),
                now,
            ),
        )
        cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    except sqlite3.IntegrityError:
        row = conn.execute(
            "SELECT id FROM comments WHERE reddit_id = ?", (reddit_id,)
        ).fetchone()
        cid = row["id"]
    conn.commit()
    conn.close()
    return cid


def get_post_by_reddit_id(reddit_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM posts WHERE reddit_id = ?", (reddit_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_posts_by_ids(post_ids):
    """Fetch multiple posts by internal ID in one query.

    Returns a dict mapping post_id -> post dict, preserving only IDs that
    exist in the database.
    """
    if not post_ids:
        return {}
    conn = get_db()
    placeholders = ",".join("?" * len(post_ids))
    rows = conn.execute(
        f"SELECT * FROM posts WHERE id IN ({placeholders})"
        " ORDER BY (score + num_comments) DESC",
        list(post_ids),
    ).fetchall()
    conn.close()
    return {r["id"]: dict(r) for r in rows}


def get_comments_for_post(post_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM comments WHERE post_id = ? ORDER BY score DESC",
        (post_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
