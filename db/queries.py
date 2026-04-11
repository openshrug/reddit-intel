"""Read-only agent-facing queries.

Every function here returns plain dicts (never sqlite3.Row) and never
mutates the database.
"""

from . import get_db


def get_top_painpoints(limit=20):
    """Merged painpoints ranked by signal_count, with category name."""
    conn = get_db()
    rows = conn.execute("""
        SELECT p.*, c.name AS category
        FROM painpoints p
        LEFT JOIN categories c ON c.id = p.category_id
        ORDER BY p.signal_count DESC, p.severity DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_painpoint_evidence(painpoint_id):
    """All raw observations backing a merged painpoint, with post/comment details.

    Returns a list of dicts, each containing the pending painpoint fields
    plus the originating post title, subreddit, permalink, and optionally
    the comment body and permalink.
    """
    conn = get_db()
    rows = conn.execute("""
        SELECT
            pp.id               AS pending_id,
            pp.title            AS pending_title,
            pp.description      AS pending_description,
            pp.quoted_text,
            pp.severity         AS pending_severity,
            pp.extracted_at,
            p.subreddit,
            p.title             AS post_title,
            p.selftext          AS post_body,
            p.score             AS post_score,
            p.permalink         AS post_permalink,
            p.created_utc       AS post_created_utc,
            c.body              AS comment_body,
            c.score             AS comment_score,
            c.permalink         AS comment_permalink,
            cat.name            AS category_name
        FROM painpoint_sources ps
        JOIN pending_painpoints pp ON pp.id = ps.pending_painpoint_id
        JOIN posts p ON p.id = pp.post_id
        LEFT JOIN comments c ON c.id = pp.comment_id
        LEFT JOIN categories cat ON cat.id = pp.category_id
        WHERE ps.painpoint_id = ?
        ORDER BY p.score DESC
    """, (painpoint_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_painpoints_by_category(category_name, limit=20):
    """Merged painpoints in a specific category."""
    conn = get_db()
    rows = conn.execute("""
        SELECT p.*, c.name AS category
        FROM painpoints p
        JOIN categories c ON c.id = p.category_id AND c.name = ?
        ORDER BY p.signal_count DESC, p.severity DESC
        LIMIT ?
    """, (category_name, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_painpoints_by_subreddit(subreddit, limit=20):
    """Merged painpoints that have evidence from a specific subreddit."""
    conn = get_db()
    rows = conn.execute("""
        SELECT p.*, c.name AS category,
               COUNT(DISTINCT pp.id) AS evidence_count
        FROM painpoints p
        JOIN painpoint_sources ps ON ps.painpoint_id = p.id
        JOIN pending_painpoints pp ON pp.id = ps.pending_painpoint_id
        JOIN posts po ON po.id = pp.post_id
        LEFT JOIN categories c ON c.id = p.category_id
        WHERE po.subreddit = ?
        GROUP BY p.id
        ORDER BY evidence_count DESC, p.signal_count DESC
        LIMIT ?
    """, (subreddit, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_subreddit_summary(subreddit):
    """High-level stats for a subreddit: post count, comment count,
    painpoint count, top categories."""
    conn = get_db()

    counts = conn.execute("""
        SELECT
            COUNT(DISTINCT p.id) AS post_count,
            COUNT(DISTINCT cm.id) AS comment_count
        FROM posts p
        LEFT JOIN comments cm ON cm.post_id = p.id
        WHERE p.subreddit = ?
    """, (subreddit,)).fetchone()

    painpoint_count = conn.execute("""
        SELECT COUNT(DISTINCT ps.painpoint_id) AS cnt
        FROM painpoint_sources ps
        JOIN pending_painpoints pp ON pp.id = ps.pending_painpoint_id
        JOIN posts po ON po.id = pp.post_id
        WHERE po.subreddit = ?
    """, (subreddit,)).fetchone()

    top_cats = conn.execute("""
        SELECT c.name, COUNT(DISTINCT ps.painpoint_id) AS painpoint_count
        FROM painpoint_sources ps
        JOIN pending_painpoints pp ON pp.id = ps.pending_painpoint_id
        JOIN posts po ON po.id = pp.post_id
        LEFT JOIN categories c ON c.id = pp.category_id
        WHERE po.subreddit = ? AND c.id IS NOT NULL
        GROUP BY c.id
        ORDER BY painpoint_count DESC
        LIMIT 10
    """, (subreddit,)).fetchall()

    conn.close()
    return {
        "subreddit": subreddit,
        "post_count": counts["post_count"],
        "comment_count": counts["comment_count"],
        "painpoint_count": painpoint_count["cnt"],
        "top_categories": [dict(r) for r in top_cats],
    }


def get_stats():
    """Global DB stats."""
    conn = get_db()
    stats = {}
    for table in ["posts", "comments", "pending_painpoints", "painpoints", "categories"]:
        stats[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    stats["unmerged_pending"] = conn.execute(
        "SELECT COUNT(*) FROM pending_painpoints pp "
        "LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id "
        "WHERE ps.painpoint_id IS NULL"
    ).fetchone()[0]
    stats["subreddits"] = conn.execute(
        "SELECT COUNT(DISTINCT subreddit) FROM posts"
    ).fetchone()[0]
    conn.close()
    return stats


def run_sql(query, params=None):
    """Execute arbitrary read-only SQL for agent research.

    Only SELECT statements are allowed.
    """
    query = query.strip()
    if not query.upper().startswith("SELECT"):
        return {"error": "Only SELECT queries allowed"}
    conn = get_db()
    try:
        rows = conn.execute(query, params or ()).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()
