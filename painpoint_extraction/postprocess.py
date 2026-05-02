"""
Painpoint quote postprocessing.

Main flow:
1. Build a per-post source index from the batch's title/selftext/comments.
2. For each LLM-emitted painpoint, try to map `quoted_text` to one real,
   contiguous source span.
3. Prefer exact normalized matches and use them to fix `comment_id`.
4. If exact matching fails, try conservative order-preserving fuzzy matching
   against bounded source windows.
5. Store the exact source span after any repair; discard the painpoint when
   no unique trustworthy span is found.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

FUZZY_UNIQUE_MARGIN = 0.03
SPAN_LENGTH_MIN_RATIO = 0.6
SPAN_LENGTH_MAX_RATIO = 1.6


def postprocess_painpoints(items, batch):
    """Repair source attribution and discard rows with unverifiable quotes."""
    source_lookup = _build_source_lookup(batch)
    kept = []
    stats = _empty_postprocess_stats()
    for item in items:
        fixed, item_stats = postprocess_painpoint(item, source_lookup)
        _merge_postprocess_stats(stats, item_stats)
        if fixed is not None:
            kept.append(fixed)
    return kept, stats


def postprocess_painpoint(item, source_lookup):
    """Return a source-faithful painpoint row, or None when quote repair fails."""
    stats = _empty_postprocess_stats()
    sources = source_lookup.get(item.get("post_id"))
    quote = (item.get("quoted_text") or "").strip()
    if not quote or sources is None:
        stats["dropped"] += 1
        return None, stats

    match = _find_quote_match(quote, sources, item.get("comment_id"))
    if match is None:
        stats["dropped"] += 1
        return None, stats

    fixed = dict(item)
    if _fix_attribution(fixed, match):
        stats["attribution_fixed"] += 1
    if fixed.get("quoted_text") != match.text:
        fixed["quoted_text"] = match.text
        if match.fuzzy:
            stats["fuzzy_repaired"] += 1
        if match.numeric_entity_changed:
            stats["numeric_entity_repaired"] += 1
    stats["kept"] += 1
    return fixed, stats


def _fix_attribution(item, match):
    """Apply the matched source id to a painpoint item."""
    if item.get("comment_id") == match.comment_id:
        return False
    item["comment_id"] = match.comment_id
    return True


@dataclass(frozen=True)
class _SourceEntry:
    comment_id: int | None
    text: str


@dataclass(frozen=True)
class _QuoteMatch:
    comment_id: int | None
    text: str
    score: float
    fuzzy: bool
    numeric_entity_changed: bool = False


def _build_source_lookup(batch):
    source_lookup = {}
    for post, comments in batch:
        post_text = post.get("title") or ""
        if post.get("selftext"):
            post_text = f"{post_text} {post.get('selftext') or ''}".strip()
        entries = [_SourceEntry(None, post_text)]
        entries.extend(
            _SourceEntry(c["id"], c.get("body") or "") for c in comments
        )
        source_lookup[post["id"]] = entries
    return source_lookup


def _find_quote_match(quote, sources, current_comment_id):
    exact_matches = []
    for source in sources:
        exact_matches.extend(_exact_quote_matches(quote, source))
    if exact_matches:
        for match in exact_matches:
            if match.comment_id == current_comment_id:
                return match
        return exact_matches[0]

    # Lower-threshold paraphrase recovery can be reconsidered if drop
    # metrics are too high; v1 intentionally avoids loose semantic repairs.
    return _find_fuzzy_quote_match(quote, sources)


def _exact_quote_matches(quote, source):
    quote_norm = _normalize_text(quote)
    if not quote_norm:
        return []
    source_norm, mapping = _normalize_with_mapping(source.text)
    matches = []
    start = source_norm.find(quote_norm)
    while start != -1:
        end = start + len(quote_norm)
        original_start = mapping[start]
        original_end = mapping[end - 1] + 1
        matches.append(_QuoteMatch(
            comment_id=source.comment_id,
            text=source.text[original_start:original_end],
            score=1.0,
            fuzzy=False,
            numeric_entity_changed=False,
        ))
        start = source_norm.find(quote_norm, start + 1)
    return matches


def _find_fuzzy_quote_match(quote, sources):
    quote_norm = _normalize_text(quote)
    quote_tokens = _tokens(quote_norm)
    if len(quote_tokens) < 4:
        return None

    threshold = _fuzzy_threshold(len(quote_tokens))
    candidates = []
    for source in sources:
        for span in _candidate_spans(source.text, len(quote_tokens)):
            span_norm = _normalize_text(span)
            if not span_norm or not _span_length_is_similar(quote_norm, span_norm):
                continue
            score = SequenceMatcher(None, quote_norm, span_norm).ratio()
            if score >= threshold:
                candidates.append(_QuoteMatch(
                    comment_id=source.comment_id,
                    text=span,
                    score=score,
                    fuzzy=True,
                    numeric_entity_changed=(
                        _special_tokens(quote) != _special_tokens(span)
                    ),
                ))

    if not candidates:
        return None
    candidates.sort(key=lambda m: m.score, reverse=True)
    if len(candidates) > 1:
        if candidates[0].score - candidates[1].score < FUZZY_UNIQUE_MARGIN:
            return None
    return candidates[0]


def _candidate_spans(text, quote_token_count):
    tokens = list(re.finditer(r"\S+", text or ""))
    if not tokens:
        return []
    min_tokens = max(1, int(quote_token_count * SPAN_LENGTH_MIN_RATIO))
    max_tokens = max(
        min_tokens, int(quote_token_count * SPAN_LENGTH_MAX_RATIO + 0.999),
    )
    spans = []
    seen = set()
    for start_idx in range(len(tokens)):
        for size in range(min_tokens, max_tokens + 1):
            end_idx = start_idx + size - 1
            if end_idx >= len(tokens):
                break
            start = tokens[start_idx].start()
            end = tokens[end_idx].end()
            key = (start, end)
            if key in seen:
                continue
            spans.append(text[start:end])
            seen.add(key)
    return spans


def _normalize_text(text):
    return _normalize_with_mapping(text)[0]


def _normalize_with_mapping(text):
    normalized = []
    mapping = []
    last_was_space = True
    for idx, char in enumerate(text or ""):
        char = _normalize_char(char)
        if not char:
            continue
        if char.isspace():
            if not last_was_space:
                normalized.append(" ")
                mapping.append(idx)
            last_was_space = True
            continue
        normalized.append(char.lower())
        mapping.append(idx)
        last_was_space = False
    if normalized and normalized[-1] == " ":
        normalized.pop()
        mapping.pop()
    return "".join(normalized), mapping


def _normalize_char(char):
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    char = replacements.get(char, char)
    if char in {"*", "_", "`"}:
        return ""
    return char


def _tokens(normalized_text):
    return re.findall(r"\S+", normalized_text or "")


def _fuzzy_threshold(token_count):
    if token_count <= 6:
        return 0.95
    if token_count <= 12:
        return 0.92
    return 0.90


def _span_length_is_similar(quote_norm, span_norm):
    if not quote_norm or not span_norm:
        return False
    ratio = len(span_norm) / len(quote_norm)
    return SPAN_LENGTH_MIN_RATIO <= ratio <= SPAN_LENGTH_MAX_RATIO


def _special_tokens(text):
    return re.findall(
        r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?%?|"
        r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|"
        r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)",
        text or "",
    )


def _empty_postprocess_stats():
    return {
        "kept": 0,
        "attribution_fixed": 0,
        "fuzzy_repaired": 0,
        "numeric_entity_repaired": 0,
        "dropped": 0,
    }


def _merge_postprocess_stats(total, delta):
    for key, value in delta.items():
        total[key] += value
