"""Hybrid (BM25 + dense) category retrieval for the creation gate.

Purpose: stop the split pass and uncat-review from minting new
categories that are near-duplicates of something already in the tree.
Before those paths call `_upsert_category_by_name`, they call
`find_similar_category` — if the top candidate's dense cosine is above
`SIMILAR_CATEGORY_THRESHOLD`, the caller routes its would-be members to
the existing category instead of creating a new one.

Why hybrid and not just dense: dense cosine on `text-embedding-3-small`
can miss exact-keyword overlaps (rare vocabulary like "ATS", product
names like "Stripe") that BM25 catches cleanly. BM25 surfaces those as
candidates; the final accept still uses dense cosine so pure keyword
noise can't force a match.

Fusion is Reciprocal Rank Fusion (k=60, per Tian Pan's hybrid-retrieval
article). RRF gives us recall; dense cosine gives us the semantic
accept gate.
"""

import re
import struct
from typing import Optional

from . import UNCATEGORIZED_NAME
from .embeddings import EMBEDDING_DIM, _pack_embedding

# Above this dense cosine, the proposed category is considered a
# near-duplicate of an existing one — skip creation, route painpoints
# to the existing. Chosen from the E2E probe (probe_sibling_sims.py):
#
#   noise floor (cross-tree):       0.21 – 0.32
#   distinct siblings (keep apart): 0.34 – 0.53 (max seen: 0.526)
#   duplicate pairs:                0.58 – 0.75
#
# 0.60 catches duplicates while leaving genuine-but-related siblings
# alone. The borderline Video Tools ↔ Video Transcription at 0.596
# stays separate — transcription is a legitimately distinct task.
SIMILAR_CATEGORY_THRESHOLD = 0.60

# When split proposes a sub-category, its default parent is the split
# source's parent. That's correct most of the time but occasionally
# wrong — the LLM clustered a group of painpoints about a cross-domain
# topic (e.g. 8 marketing-about-AI painpoints that were living under
# `AI/ML > AI Coding Tools`), and the sub really belongs under a
# different root. We re-parent only when the top global candidate is
# BOTH decisively better AND passes a minimum semantic threshold —
# otherwise we keep the default parent.
CROSS_PARENT_REPARENT_MIN_COS = 0.45  # noise floor ~0.32, want clear signal
CROSS_PARENT_REPARENT_MARGIN = 0.10   # top must beat default by this much

# RRF rank constant. 60 is the standard from the RRF paper (Cormack et
# al., 2009) and the figure used in Tian Pan's article — empirically
# robust across different retriever score distributions.
RRF_K = 60

# How many candidates each retriever (dense + BM25) returns before
# fusion. Larger K = better recall at negligible cost on a <200-row
# categories table.
TOP_K = 10


# ---------------------------------------------------------------------------
# FTS5 sync — keep category_fts in lock-step with writes to `categories`.
# ---------------------------------------------------------------------------


def init_category_fts(conn):
    """Bulk-populate category_fts for any category missing from it.

    Idempotent: rows already present are skipped by the anti-join. Runs
    at init_db() time so existing DBs (created before the FTS5 table
    existed) get filled without a one-shot script.
    """
    rows = conn.execute(
        "SELECT c.id, c.name, c.description "
        "FROM categories c LEFT JOIN category_fts f ON f.rowid = c.id "
        "WHERE f.rowid IS NULL"
    ).fetchall()
    for r in rows:
        conn.execute(
            "INSERT INTO category_fts (rowid, name, description) VALUES (?,?,?)",
            (r["id"], r["name"] or "", r["description"] or ""),
        )


def sync_category_fts(conn, category_id, name, description):
    """Upsert a category row into category_fts.

    FTS5 contentful tables need explicit delete-then-insert for replace
    semantics; there is no UPSERT sugar for virtual tables.
    """
    conn.execute("DELETE FROM category_fts WHERE rowid = ?", (category_id,))
    conn.execute(
        "INSERT INTO category_fts (rowid, name, description) VALUES (?,?,?)",
        (category_id, name or "", description or ""),
    )


def delete_category_fts(conn, category_id):
    """Drop a category's FTS5 row. Paired with category_vec / anchor
    deletes in the same mutations (delete_category, merge loser)."""
    conn.execute("DELETE FROM category_fts WHERE rowid = ?", (category_id,))


# ---------------------------------------------------------------------------
# Query building: sanitize free text to a safe FTS5 OR-query.
# ---------------------------------------------------------------------------

# FTS5 reserved operators — a user-provided token matching these would
# parse as the operator, not as a term. Drop them.
_FTS5_OPERATORS = frozenset({"and", "or", "not", "near"})


def _tokenize_for_fts(text: str) -> list[str]:
    """Extract searchable tokens from free text.

    FTS5 is strict about special characters, so we pre-tokenize:
    - split on any non-alphanumeric
    - lowercase
    - drop 1-char tokens (noise) and FTS5 reserved operators
    - dedupe while preserving first-seen order
    """
    if not text:
        return []
    seen = set()
    out = []
    for tok in re.split(r"[^A-Za-z0-9]+", text.lower()):
        if len(tok) < 2 or tok in _FTS5_OPERATORS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _build_fts_query(text: str) -> Optional[str]:
    """Build a single FTS5 OR-query from free text, or None if empty.

    We use OR (not the AND default) because we want recall — any
    overlapping keyword pulls the category into the candidate pool, and
    the dense-cosine accept gate decides whether it's actually relevant.
    """
    tokens = _tokenize_for_fts(text)
    if not tokens:
        return None
    return " OR ".join(tokens)


# ---------------------------------------------------------------------------
# Hybrid retrieval with RRF fusion.
# ---------------------------------------------------------------------------


def _dense_top_k(conn, embedding, top_k):
    """Top-K category rowids by dense cosine, with cosine sim returned.

    Returns list of (category_id, cosine_sim) in descending sim order.
    Empty list if category_vec is empty.
    """
    blob = _pack_embedding(embedding)
    rows = conn.execute(
        "SELECT rowid, distance FROM category_vec "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (blob, top_k),
    ).fetchall()
    return [(r[0], 1.0 - r[1]) for r in rows]


def _bm25_top_k(conn, query, top_k):
    """Top-K category rowids by BM25 score over category_fts.

    Returns list of category_ids only (BM25 score is not a meaningful
    absolute number; it's useful only for ranking). Empty list if the
    query is None/empty or FTS5 has no matches.
    """
    if not query:
        return []
    rows = conn.execute(
        "SELECT rowid FROM category_fts WHERE category_fts MATCH ? "
        "ORDER BY bm25(category_fts) LIMIT ?",
        (query, top_k),
    ).fetchall()
    return [r[0] for r in rows]


def _rrf_fuse(dense_ids, bm25_ids, k=RRF_K):
    """Merge two ranked id lists via Reciprocal Rank Fusion.

    Each id's score = sum of 1/(k + rank_i) across the lists it appears
    in. Returns ids sorted by fused score DESC.
    """
    scores: dict[int, float] = {}
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(bm25_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _uncategorized_id_or_none(conn):
    row = conn.execute(
        "SELECT id FROM categories WHERE name = ?", (UNCATEGORIZED_NAME,),
    ).fetchone()
    return row["id"] if row is not None else None


def _backfill_dense_cos(conn, embedding, missing_ids, dense_cos):
    """Populate `dense_cos[id]` for BM25-only hits by reading
    category_vec directly. Mutates `dense_cos` in place."""
    if not missing_ids:
        return
    placeholders = ",".join("?" * len(missing_ids))
    rows = conn.execute(
        f"SELECT rowid, embedding FROM category_vec "
        f"WHERE rowid IN ({placeholders})",
        missing_ids,
    ).fetchall()
    expected_bytes = EMBEDDING_DIM * 4
    import math
    na = math.sqrt(sum(x * x for x in embedding))
    if na == 0:
        return
    for r in rows:
        blob = r["embedding"]
        if len(blob) != expected_bytes:
            continue
        cat_emb = list(struct.unpack(f"{EMBEDDING_DIM}f", blob))
        nb = math.sqrt(sum(x * x for x in cat_emb))
        if nb == 0:
            continue
        dot = sum(a * b for a, b in zip(embedding, cat_emb))
        dense_cos[r["rowid"]] = dot / (na * nb)


def find_hybrid_candidates(
    conn,
    embedding,
    query_text: str,
    *,
    exclude_ids: Optional[set] = None,
    top_k: int = TOP_K,
):
    """Core primitive: return categories ranked by hybrid (BM25 + dense)
    similarity to a query embedding / text pair.

    `embedding` is assumed pre-computed — caller owns the embedder call
    (lets reroute reuse painpoint_vec without re-embedding).
    `query_text` is the raw text BM25 tokenizes (usually title +
    description).

    Uncategorized is always excluded. Caller can add more via
    `exclude_ids` (e.g. the painpoint's current category when
    rerouting).

    Returns `[(category_id, dense_cos, rrf_score), ...]` in RRF-DESC
    order. BM25-only hits get their dense cosine backfilled from
    category_vec so the caller can make an accept decision on dense
    cos alone.
    """
    exclude = set(exclude_ids or ())
    uncat_id = _uncategorized_id_or_none(conn)
    if uncat_id is not None:
        exclude.add(uncat_id)

    dense_hits = _dense_top_k(conn, embedding, top_k + len(exclude))
    dense_ids = [cid for cid, _cos in dense_hits if cid not in exclude][:top_k]
    dense_cos = {cid: cos for cid, cos in dense_hits}

    query = _build_fts_query(query_text)
    bm25_ids_raw = _bm25_top_k(conn, query, top_k + len(exclude))
    bm25_ids = [cid for cid in bm25_ids_raw if cid not in exclude][:top_k]

    fused = _rrf_fuse(dense_ids, bm25_ids)
    if not fused:
        return []

    _backfill_dense_cos(
        conn, embedding,
        [cid for cid, _ in fused if cid not in dense_cos],
        dense_cos,
    )

    return [(cid, dense_cos.get(cid, 0.0), rrf) for cid, rrf in fused]


def category_dense_cos(conn, embedding, category_id):
    """Cosine between `embedding` and `category_vec` for the given id.
    Returns 0.0 if the category has no vec row (e.g. Uncategorized,
    which is centroid-exempt, or brand-new categories pre-anchor)."""
    row = conn.execute(
        "SELECT embedding FROM category_vec WHERE rowid = ?", (category_id,),
    ).fetchone()
    if row is None or len(row["embedding"]) != EMBEDDING_DIM * 4:
        return 0.0
    cat_emb = list(struct.unpack(f"{EMBEDDING_DIM}f", row["embedding"]))
    import math
    na = math.sqrt(sum(x * x for x in embedding))
    nb = math.sqrt(sum(x * x for x in cat_emb))
    if na == 0 or nb == 0:
        return 0.0
    return sum(a * b for a, b in zip(embedding, cat_emb)) / (na * nb)


def find_similar_category(
    conn,
    name: str,
    description: str,
    embedder,
    *,
    exclude_ids: Optional[set] = None,
    top_k: int = TOP_K,
):
    """Hybrid lookup for a *proposed* new category (name + description).

    Thin wrapper around `find_hybrid_candidates` that embeds the
    proposal text before dispatching. Caller inspects `[0]`'s dense
    cosine to decide whether the proposed category is a near-duplicate
    (>= SIMILAR_CATEGORY_THRESHOLD → skip creation; route members to
    that category instead).

    Returns `[(category_id, dense_cos, rrf_score), ...]` in RRF-DESC
    order. Empty list if the proposal has no text or no candidates.
    """
    text = f"{name or ''} {description or ''}".strip()
    if not text:
        return []
    embedding = embedder.embed(text)
    return find_hybrid_candidates(
        conn, embedding, text,
        exclude_ids=exclude_ids, top_k=top_k,
    )
