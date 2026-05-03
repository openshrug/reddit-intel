"""Read-only opportunity evidence queries.

This module owns the database semantics for opportunity discovery: which
painpoints are local to a subreddit, which evidence comes from adjacent
communities, and which permalink should be used for a quoted source.
"""

from . import get_db


def get_opportunity_evidence_rows(subreddit, *, limit=10, category=None):
    """Return canonical painpoints with local/cross-subreddit evidence.

    The returned rows are plain dicts so callers outside MCP can reuse the
    same evidence semantics in tests, docs, or future export surfaces.
    """
    subreddit = _normalize_subreddit(subreddit)
    if not subreddit:
        return []

    conn = get_db()
    try:
        candidates = _fetch_candidates(
            conn, subreddit, limit=max(1, int(limit)), category=category,
        )
        for candidate in candidates:
            evidence = _fetch_primary_quote_evidence(conn, candidate["painpoint_id"])
            local, cross = _split_evidence(evidence, subreddit)
            candidate["local_evidence"] = local
            candidate["cross_subreddit_evidence"] = cross
        return candidates
    finally:
        conn.close()


def _fetch_candidates(conn, subreddit, *, limit, category=None):
    params = [subreddit]
    category_filter = ""
    if category:
        category_filter = "AND c.name = ?"
        params.append(category)
    params.append(limit)

    rows = conn.execute(
        f"""
        WITH source_rows AS (
            SELECT
                ps.painpoint_id,
                ps.pending_painpoint_id,
                src.post_id,
                src.comment_id,
                po.subreddit,
                pp.severity
            FROM painpoint_sources ps
            JOIN pending_painpoint_all_sources src
              ON src.pending_painpoint_id = ps.pending_painpoint_id
            JOIN posts po ON po.id = src.post_id
            JOIN pending_painpoints pp ON pp.id = ps.pending_painpoint_id
        )
        SELECT
            p.id AS painpoint_id,
            p.title,
            p.description,
            p.severity,
            c.name AS category,
            COUNT(*) AS local_signal_count,
            (
                SELECT COUNT(*)
                FROM source_rows sr_all
                WHERE sr_all.painpoint_id = p.id
            ) AS global_signal_count,
            (
                SELECT MIN(sr_all.severity)
                FROM source_rows sr_all
                WHERE sr_all.painpoint_id = p.id
            ) AS severity_min,
            (
                SELECT MAX(sr_all.severity)
                FROM source_rows sr_all
                WHERE sr_all.painpoint_id = p.id
            ) AS severity_max,
            (
                SELECT GROUP_CONCAT(DISTINCT sr_all.subreddit)
                FROM source_rows sr_all
                WHERE sr_all.painpoint_id = p.id
            ) AS subreddits_seen_csv
        FROM painpoints p
        JOIN source_rows sr ON sr.painpoint_id = p.id
        LEFT JOIN categories c ON c.id = p.category_id
        WHERE sr.subreddit = ?
        {category_filter}
        GROUP BY p.id
        ORDER BY
            local_signal_count DESC,
            global_signal_count DESC,
            p.signal_count DESC,
            p.severity DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    return [_candidate_dict(row, subreddit) for row in rows]


def _fetch_primary_quote_evidence(conn, painpoint_id):
    rows = conn.execute(
        """
        SELECT
            pp.id AS pending_id,
            pp.title AS pending_title,
            pp.description AS pending_description,
            pp.quoted_text AS quote,
            pp.severity AS pending_severity,
            pp.extracted_at,
            po.subreddit,
            po.title AS post_title,
            po.selftext AS post_body,
            po.score AS post_score,
            po.permalink AS post_permalink,
            po.created_utc AS post_created_utc,
            cm.body AS comment_body,
            cm.score AS comment_score,
            cm.permalink AS comment_permalink
        FROM painpoint_sources ps
        JOIN pending_painpoints pp ON pp.id = ps.pending_painpoint_id
        JOIN posts po ON po.id = pp.post_id
        LEFT JOIN comments cm ON cm.id = pp.comment_id
        WHERE ps.painpoint_id = ?
          AND pp.quoted_text IS NOT NULL
          AND TRIM(pp.quoted_text) != ''
        ORDER BY po.score DESC, COALESCE(cm.score, -1) DESC
        """,
        (painpoint_id,),
    ).fetchall()

    evidence = []
    for row in rows:
        item = dict(row)
        source_permalink = _source_permalink(item)
        if not source_permalink:
            continue
        item["source_type"] = (
            "comment" if item.get("comment_permalink") else "post"
        )
        item["source_score"] = (
            item["comment_score"]
            if item["source_type"] == "comment"
            else item["post_score"]
        )
        item["source_permalink"] = source_permalink
        evidence.append(item)
    return evidence


def _candidate_dict(row, requested_subreddit):
    data = dict(row)
    subreddits = _split_csv(data.pop("subreddits_seen_csv"))
    return {
        **data,
        "requested_subreddit": requested_subreddit,
        "subreddits_seen": subreddits,
    }


def _split_evidence(evidence, requested_subreddit):
    local = []
    cross = []
    for item in evidence:
        if _normalize_subreddit(item.get("subreddit")) == requested_subreddit:
            local.append(item)
        else:
            cross.append(item)
    return local, cross


def _source_permalink(row):
    permalink = row.get("comment_permalink") or row.get("post_permalink") or ""
    return _normalize_permalink(permalink)


def _normalize_subreddit(value):
    return (value or "").strip().removeprefix("r/").removeprefix("/r/").strip("/")


def _normalize_permalink(value):
    value = (value or "").strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return f"https://reddit.com{value}"
    return f"https://reddit.com/{value}"


def _split_csv(value):
    if not value:
        return []
    return sorted({part for part in value.split(",") if part})
