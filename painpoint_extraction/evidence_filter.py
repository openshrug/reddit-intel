"""Filter extracted rows by the evidence behind each candidate painpoint."""

from typing import Literal

# Single source of truth for the label set. `EvidenceType` (used in the
# pydantic schema) is derived from these tuples so a label rename / addition
# only happens in one place. The matching test
# `tests/test_extraction.py::TestEvidenceFilter::test_label_set_is_consistent`
# asserts the derived Literal stays in lockstep.
ALLOWED_EVIDENCE_TYPES: tuple[str, ...] = (
    "direct_complaint",
    "missing_feature",
    "broken_experience",
    "unmet_need",
)

DROP_EVIDENCE_TYPES: tuple[str, ...] = (
    "joke_meme_sarcasm",
    "self_promotion_showcase",
    "praise_only",
    "thought_leadership",
    "policy_news_grief_inferred",
    "abstract_opinion",
)

EvidenceType = Literal[*ALLOWED_EVIDENCE_TYPES, *DROP_EVIDENCE_TYPES]

UNKNOWN_EVIDENCE_TYPE = "unknown"


def filter_non_pain_items(items):
    """Drop rows whose evidence label is not an explicit pain signal."""
    kept = []
    stats = _empty_filter_stats()

    for item in items:
        evidence_type = _normalize_evidence_type(item.get("evidence_type"))
        if evidence_type in ALLOWED_EVIDENCE_TYPES:
            kept.append(_strip_evidence_type(item))
            stats["kept"] += 1
            continue

        drop_type = (
            evidence_type
            if evidence_type in DROP_EVIDENCE_TYPES
            else UNKNOWN_EVIDENCE_TYPE
        )
        stats["dropped"] += 1
        stats["dropped_by_evidence_type"][drop_type] = (
            stats["dropped_by_evidence_type"].get(drop_type, 0) + 1
        )
        stats["dropped_items"].append(_compact_dropped_item(item, drop_type))

    return kept, stats


def _strip_evidence_type(item):
    out = dict(item)
    out.pop("evidence_type", None)
    return out


def _normalize_evidence_type(value):
    if not isinstance(value, str):
        return UNKNOWN_EVIDENCE_TYPE
    return value.strip().lower() or UNKNOWN_EVIDENCE_TYPE


def _compact_dropped_item(item, evidence_type):
    return {
        "evidence_type": evidence_type,
        "post_id": item.get("post_id"),
        "comment_id": item.get("comment_id"),
        "title": _compact_text(item.get("title")),
        "quoted_text": _compact_text(item.get("quoted_text")),
    }


def _compact_text(value, *, limit=120):
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def _empty_filter_stats():
    return {
        "kept": 0,
        "dropped": 0,
        "dropped_by_evidence_type": {},
        "dropped_items": [],
    }
