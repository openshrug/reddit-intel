"""Agent-ready opportunity evidence packs.

reddit-intel stays evidence-first: this module packages ranked painpoint
evidence for opportunity discovery. Synthesis instructions, fields, and
layout rules live in `opportunity_briefs/AGENTS.md`,
`opportunity_briefs/SYNTHESIS_TEMPLATE.md`, and
`opportunity_briefs/BRIEF_LAYOUT.md`, exposed via MCP resources by
`mcp_server.py`.
"""

from typing import Literal

from pydantic import BaseModel, Field

from db import normalize_subreddit
from db.opportunity_queries import get_opportunity_evidence_rows

BRIEF_EVIDENCE_LIMIT = 30
"""Default painpoint count for opportunity-brief synthesis.

Single source of truth referenced by `mcp_server.opportunity_brief` (the
launcher prompt), `opportunity_briefs/AGENTS.md` (step 5 of the workflow),
and `tests/test_mcp_brief.py` (the prompt-body assertion). Update here to
re-tune the brief workflow."""

DEFAULT_EVIDENCE_LIMIT = 10
"""Per-side cap on quotes returned in each evidence pack (local + cross).

Sized so the synthesis template's `3-6 quotes per opportunity` ask sees a
generous pool to pick from rather than being pinned at the lower bound."""


class EvidenceQuote(BaseModel):
    quote: str
    source_permalink: str
    source_type: Literal["post", "comment"]
    subreddit: str
    post_title: str | None = None
    post_permalink: str | None = None
    comment_permalink: str | None = None
    post_score: int | None = None
    comment_score: int | None = None
    source_score: int | None = None
    post_created_utc: float | None = None


class EvidenceStrength(BaseModel):
    local_signal_count: int
    global_signal_count: int
    severity_min: int | None = None
    severity_max: int | None = None
    local_quote_count: int
    cross_subreddit_quote_count: int


class OpportunityEvidencePack(BaseModel):
    painpoint_id: int
    title: str
    description: str | None = None
    category: str | None = None
    requested_subreddit: str
    subreddits_seen: list[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrength
    local_evidence: list[EvidenceQuote] = Field(default_factory=list)
    cross_subreddit_evidence: list[EvidenceQuote] = Field(default_factory=list)


class OpportunityEvidenceResponse(BaseModel):
    requested_subreddit: str
    category: str | None = None
    total_painpoints: int
    painpoints: list[OpportunityEvidencePack]


def get_opportunity_evidence(subreddit, *, limit=BRIEF_EVIDENCE_LIMIT, category=None):
    """Return ranked evidence packs for opportunity synthesis.

    `limit` is a post-filter contract: the upstream SQL already requires each
    candidate painpoint to have at least one source with a non-empty quote,
    so the caller asking for N packs gets up to N packs back (capped by the
    actual number of quoted painpoints in the subreddit)."""
    subreddit = normalize_subreddit(subreddit)
    rows = get_opportunity_evidence_rows(
        subreddit, limit=_clamp_limit(limit), category=category,
    )
    packs = [_build_pack(row) for row in rows]
    response = OpportunityEvidenceResponse(
        requested_subreddit=subreddit,
        category=category,
        total_painpoints=len(packs),
        painpoints=packs,
    )
    return response.model_dump()


def _build_pack(row):
    local = [_quote(item) for item in row.get("local_evidence", [])]
    cross = [_quote(item) for item in row.get("cross_subreddit_evidence", [])]
    local = local[:DEFAULT_EVIDENCE_LIMIT]
    cross = cross[:DEFAULT_EVIDENCE_LIMIT]
    return OpportunityEvidencePack(
        painpoint_id=row["painpoint_id"],
        title=row["title"],
        description=row.get("description"),
        category=row.get("category"),
        requested_subreddit=row["requested_subreddit"],
        subreddits_seen=row.get("subreddits_seen", []),
        evidence_strength=EvidenceStrength(
            local_signal_count=row.get("local_signal_count") or 0,
            global_signal_count=row.get("global_signal_count") or 0,
            severity_min=row.get("severity_min"),
            severity_max=row.get("severity_max"),
            local_quote_count=len(local),
            cross_subreddit_quote_count=len(cross),
        ),
        local_evidence=local,
        cross_subreddit_evidence=cross,
    )


def _quote(item):
    return EvidenceQuote(
        quote=item["quote"],
        source_permalink=item["source_permalink"],
        source_type=item["source_type"],
        subreddit=item["subreddit"],
        post_title=item.get("post_title"),
        post_permalink=item.get("post_permalink"),
        comment_permalink=item.get("comment_permalink"),
        post_score=item.get("post_score"),
        comment_score=item.get("comment_score"),
        source_score=item.get("source_score"),
        post_created_utc=item.get("post_created_utc"),
    )


def _clamp_limit(limit):
    return max(1, min(int(limit), 50))
