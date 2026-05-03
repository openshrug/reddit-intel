"""Agent-ready opportunity evidence packs.

reddit-intel stays evidence-first: this module packages ranked painpoint
evidence and synthesis instructions, but it does not generate product ideas.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from db.opportunity_queries import get_opportunity_evidence_rows

DEFAULT_LIMIT = 10
DEFAULT_EVIDENCE_LIMIT = 4
SYNTHESIS_TEMPLATE = Path(__file__).parent / "opportunity_briefs" / "SYNTHESIS_TEMPLATE.md"

CORE_RULES = [
    "Start from evidence packs; do not invent opportunities from general market knowledge.",
    "Use an evidence-first, fit-second flow: identify the strongest opportunities before applying builder preferences.",
    "If the user does not provide builder preferences, do not invent defaults and omit builder fit.",
    "Render every evidence quote as a Markdown hyperlink using source_permalink.",
    "Treat signal_count and source counts as repetition signals, not market-size estimates.",
    "Treat categories as rough navigation, not authoritative market structure.",
]


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
    caveats: list[str] = Field(default_factory=list)


class AgentGuidelines(BaseModel):
    core_rules: list[str]
    output_fields: list[str]
    synthesis_prompt: str


class OpportunityEvidenceResponse(BaseModel):
    requested_subreddit: str
    category: str | None = None
    total_painpoints: int
    painpoints: list[OpportunityEvidencePack]
    agent_guidelines: AgentGuidelines
    caveats: list[str]


def get_opportunity_evidence(subreddit, *, limit=DEFAULT_LIMIT, category=None):
    """Return agent-ready evidence packs for opportunity synthesis."""
    subreddit = _normalize_subreddit(subreddit)
    rows = get_opportunity_evidence_rows(
        subreddit, limit=_clamp_limit(limit), category=category,
    )
    packs = [_build_pack(row) for row in rows]
    packs = [pack for pack in packs if pack.local_evidence or pack.cross_subreddit_evidence]
    response = OpportunityEvidenceResponse(
        requested_subreddit=subreddit,
        category=category,
        total_painpoints=len(packs),
        painpoints=packs,
        agent_guidelines=_agent_guidelines(subreddit),
        caveats=[
            "Reddit evidence is qualitative signal, not proof of demand or willingness to pay.",
            "Cross-subreddit evidence is adjacent support; local evidence should lead subreddit-specific briefs.",
            "Signal counts can include deduped sources that do not have separate stored quotes.",
            "Category labels are navigation aids and may drift.",
        ],
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
        caveats=[
            "Use local evidence first for this subreddit.",
            "Use cross-subreddit evidence only as clearly labeled adjacent support.",
            "Signal counts can exceed quote counts when deduped sources lack separate stored quotes.",
        ],
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


def _agent_guidelines(subreddit):
    return AgentGuidelines(
        core_rules=CORE_RULES,
        output_fields=[
            "Opportunity title",
            "Problem statement",
            "Who seems affected, marked as inferred when needed",
            "Evidence strength",
            "Builder fit, only when explicit preferences are known",
            "Evidence-vs-fit tradeoff, only when explicit preferences are known",
            "Clickable evidence quotes",
            "MVP wedge",
            "Why existing solutions may fail, or unknown",
            "Why this may extend beyond one community",
            "Validation questions",
            "Caveats",
        ],
        synthesis_prompt=_load_synthesis_template().format(subreddit=subreddit),
    )


def _load_synthesis_template():
    return SYNTHESIS_TEMPLATE.read_text()


def _clamp_limit(limit):
    return max(1, min(int(limit), 50))


def _normalize_subreddit(value):
    return (value or "").strip().removeprefix("r/").removeprefix("/r/").strip("/")
