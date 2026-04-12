"""Painpoint relevance computation (§2 of docs/PAINPOINT_INGEST_PLAN.md).

A painpoint has one or more `(post, optional comment)` sources (the multi-
source case comes from batched LLM extraction). Per-source relevance is
`traction × recency × severity_mult`, and a painpoint's overall relevance
is the **max** over its sources (§2.3 — max not sum/mean to avoid double-
counting evidence with signal_count and to keep "is at least one source
still hot?" the operative question).
"""

from datetime import datetime, timezone
from math import log1p

from . import get_db, _now

# Tunables — see §10 of the plan. Hardcoded for now; lift to config later.
RELEVANCE_HALF_LIFE_DAYS = 14.0
MIN_RELEVANCE_TO_PROMOTE = 0.5
RELEVANCE_CACHE_TTL_SECONDS = 24 * 3600


def _parse_iso(ts):
    """Parse an ISO-8601 timestamp string to a UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    # SQLite stores ISO strings; the trailing 'Z' / '+00:00' depends on writer.
    # SQLite's datetime('now') returns naive strings like "2026-04-12 10:30:00"
    # with no timezone indicator — treat those as UTC.
    s = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def per_source_relevance(post, comment, severity, now=None):
    """Compute relevance for one (post, optional comment) source.

    `post` and `comment` are sqlite3.Row or dict-like objects with the same
    column names as the `posts` / `comments` tables. `comment` may be None
    for post-rooted painpoints.

    Returns a positive real number; higher = more relevant.
    """
    now = now or datetime.now(timezone.utc)

    # Time decay — exponential half-life. Use the comment's timestamp if
    # available (more specific), otherwise the post's.
    source_created_utc = (
        comment["created_utc"] if comment is not None and comment["created_utc"]
        else post["created_utc"]
    )
    if source_created_utc is None:
        recency = 1.0
    else:
        # posts.created_utc is stored as REAL epoch seconds in this codebase
        age_seconds = now.timestamp() - float(source_created_utc)
        age_days = max(0.0, age_seconds / 86400.0)
        recency = 0.5 ** (age_days / RELEVANCE_HALF_LIFE_DAYS)

    # Traction — computed from raw Reddit engagement stats.
    score = post["score"] or 0
    num_comments = post["num_comments"] or 0
    traction = log1p(max(0, score)) * 0.5 + log1p(max(0, num_comments)) * 0.8

    if comment is not None:
        # Comment-rooted painpoints inherit BOTH the post's traction AND the
        # comment's own engagement.
        c_score = comment["score"] or 0
        traction += log1p(max(0, c_score)) * 0.5

    # Severity — LLM's 1-10 claim, normalized. Severity comes pre-clamped
    # from db.painpoints.save_pending_painpoint, so we trust the value here.
    severity = max(1, min(10, int(severity or 5)))
    severity_mult = 0.5 + 0.1 * severity   # 1 → 0.6, 10 → 1.5

    return traction * recency * severity_mult


def _iter_pending_sources(conn, pending_id):
    """Yield (post_row, comment_row_or_None) tuples for every source attached
    to a pending painpoint, via the §7.5 view that unions primary + extras."""
    rows = conn.execute(
        "SELECT post_id, comment_id FROM pending_painpoint_all_sources "
        "WHERE pending_painpoint_id = ?",
        (pending_id,),
    ).fetchall()
    for r in rows:
        post = conn.execute("SELECT * FROM posts WHERE id = ?", (r["post_id"],)).fetchone()
        if post is None:
            continue
        comment = None
        if r["comment_id"] is not None:
            comment = conn.execute(
                "SELECT * FROM comments WHERE id = ?", (r["comment_id"],)
            ).fetchone()
        yield post, comment


def _iter_painpoint_sources(conn, painpoint_id):
    """Yield (post_row, comment_row_or_None) for every (post, comment) tuple
    backing a merged painpoint.

    Walks `painpoint_sources → pending_painpoint_all_sources` to find every
    contributing source across every pending pp that's been merged in.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT pps.post_id, pps.comment_id
        FROM painpoint_sources ps
        JOIN pending_painpoint_all_sources pps
          ON pps.pending_painpoint_id = ps.pending_painpoint_id
        WHERE ps.painpoint_id = ?
        """,
        (painpoint_id,),
    ).fetchall()
    for r in rows:
        post = conn.execute("SELECT * FROM posts WHERE id = ?", (r["post_id"],)).fetchone()
        if post is None:
            continue
        comment = None
        if r["comment_id"] is not None:
            comment = conn.execute(
                "SELECT * FROM comments WHERE id = ?", (r["comment_id"],)
            ).fetchone()
        yield post, comment


def compute_pending_relevance(pending_id, conn=None, now=None):
    """Relevance of a pending painpoint, taking max over its source set.

    The pending pp must already exist in `pending_painpoints`. Severity is
    read from that row.
    """
    own_conn = conn is None
    conn = conn or get_db()
    try:
        pp = conn.execute(
            "SELECT severity FROM pending_painpoints WHERE id = ?", (pending_id,)
        ).fetchone()
        if pp is None:
            raise ValueError(f"pending_painpoint {pending_id} not found")
        severity = pp["severity"]
        sources = list(_iter_pending_sources(conn, pending_id))
        if not sources:
            return 0.0
        return max(per_source_relevance(p, c, severity, now=now) for p, c in sources)
    finally:
        if own_conn:
            conn.close()


def compute_painpoint_relevance(painpoint_id, conn=None, now=None):
    """Relevance of a merged painpoint, taking max over its full source set.

    Walks `painpoint_sources → pending_painpoint_all_sources` to find every
    `(post, comment)` tuple that ever contributed to this painpoint.
    """
    own_conn = conn is None
    conn = conn or get_db()
    try:
        pp = conn.execute(
            "SELECT severity FROM painpoints WHERE id = ?", (painpoint_id,)
        ).fetchone()
        if pp is None:
            raise ValueError(f"painpoint {painpoint_id} not found")
        severity = pp["severity"]
        sources = list(_iter_painpoint_sources(conn, painpoint_id))
        if not sources:
            return 0.0
        return max(per_source_relevance(p, c, severity, now=now) for p, c in sources)
    finally:
        if own_conn:
            conn.close()


def cache_painpoint_relevance(painpoint_id, conn=None, now=None):
    """Compute and write `painpoints.relevance` + `relevance_updated_at`."""
    own_conn = conn is None
    conn = conn or get_db()
    try:
        rel = compute_painpoint_relevance(painpoint_id, conn=conn, now=now)
        conn.execute(
            "UPDATE painpoints SET relevance = ?, relevance_updated_at = ? WHERE id = ?",
            (rel, _now(), painpoint_id),
        )
        if own_conn:
            conn.commit()
        return rel
    finally:
        if own_conn:
            conn.close()


def get_or_compute_painpoint_relevance(painpoint_id, conn=None, now=None):
    """Read the cached relevance if it's fresh enough, otherwise recompute.

    "Fresh enough" = relevance_updated_at within RELEVANCE_CACHE_TTL_SECONDS
    of now. The cache exists so callers don't pay the source-walk cost on
    every read; the staleness window is the §2.3 "anything older than ~24h"
    rule.
    """
    own_conn = conn is None
    conn = conn or get_db()
    try:
        row = conn.execute(
            "SELECT relevance, relevance_updated_at FROM painpoints WHERE id = ?",
            (painpoint_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"painpoint {painpoint_id} not found")

        if row["relevance"] is not None and row["relevance_updated_at"]:
            cached_at = _parse_iso(row["relevance_updated_at"])
            now_dt = now or datetime.now(timezone.utc)
            age_seconds = (now_dt - cached_at).total_seconds()
            if age_seconds < RELEVANCE_CACHE_TTL_SECONDS:
                return row["relevance"]

        return cache_painpoint_relevance(painpoint_id, conn=conn, now=now)
    finally:
        if own_conn:
            conn.close()
