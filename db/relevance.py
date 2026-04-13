"""Painpoint relevance computation.

Single helper: `per_source_relevance(post, comment, severity)` computes
`traction × recency` for one source. Comment-rooted sources use
comment traction alone — the post's popularity doesn't validate a
random comment on it.

`severity` is accepted for API compatibility but does not affect the
result (the LLM's 1-10 claim was too subjective to weight on).

`_parse_iso` lives here because the category worker's staleness check
(`_category_last_activity`) imports it.
"""

from datetime import datetime, timezone
from math import log1p

RELEVANCE_HALF_LIFE_DAYS = 14.0


def _parse_iso(ts):
    """Parse an ISO-8601 timestamp string to a UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
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

    Returns a non-negative real; higher = more relevant.
    """
    del severity  # retained for caller compatibility
    now = now or datetime.now(timezone.utc)

    source_created_utc = (
        comment["created_utc"] if comment is not None and comment["created_utc"]
        else post["created_utc"]
    )
    if source_created_utc is None:
        recency = 1.0
    else:
        age_seconds = now.timestamp() - float(source_created_utc)
        age_days = max(0.0, age_seconds / 86400.0)
        recency = 0.5 ** (age_days / RELEVANCE_HALF_LIFE_DAYS)

    if comment is not None:
        c_score = comment["score"] or 0
        traction = log1p(max(0, c_score)) * 0.5
    else:
        score = post["score"] or 0
        num_comments = post["num_comments"] or 0
        traction = log1p(max(0, score)) * 0.5 + log1p(max(0, num_comments)) * 0.8

    return traction * recency
