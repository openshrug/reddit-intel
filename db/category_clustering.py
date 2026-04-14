"""Clustering primitives for the category worker.

Uses embedding cosine similarity instead of MinHash for all clustering
and inter-category similarity operations.

- Intra-bucket clustering: build connected components in a similarity
  graph at a cosine threshold.
- Inter-category similarity: cosine similarity between category
  embedding vectors.
"""

import math
import struct

from . import in_clause_placeholders
from .embeddings import (
    EMBEDDING_DIM,
    FakeEmbedder,
    _pack_embedding,
    MERGE_COSINE_THRESHOLD,
)


def _cosine_sim(a, b):
    """Cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _components(items, edges):
    """Plain union-find connected components."""
    parent = {i: i for i in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    groups = {}
    for i in items:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def cluster_painpoints(painpoints, threshold=0.40, embedder=None, conn=None):
    """Group a list of painpoint dicts into clusters by embedding similarity.

    Each painpoint must have `id`, `title`, and `description` fields. Two
    painpoints are connected if their embedding cosine similarity is above
    `threshold`. Returns a list of clusters, each a list of painpoint dicts.

    Singletons are returned as 1-element clusters.

    If `conn` is provided, **prefers stored embeddings** from
    `painpoint_vec` over re-computing via the embedder. Falls back to
    the embedder for painpoints without a stored vector. Avoids the
    expensive re-embedding pass that dominated cluster_painpoints
    runtime at scale.
    """
    if not painpoints:
        return []
    if len(painpoints) == 1:
        return [list(painpoints)]

    if embedder is None:
        embedder = FakeEmbedder()

    # Try to bulk-load existing embeddings from painpoint_vec.
    cached = {}
    if conn is not None:
        ids = [p["id"] for p in painpoints]
        rows = conn.execute(
            f"SELECT rowid, embedding FROM painpoint_vec "
            f"WHERE rowid IN ({in_clause_placeholders(len(ids))})",
            ids,
        ).fetchall()
        expected_bytes = EMBEDDING_DIM * 4
        for r in rows:
            blob = r[1]
            if len(blob) == expected_bytes:
                try:
                    cached[r[0]] = list(struct.unpack(f"{EMBEDDING_DIM}f", blob))
                except struct.error:
                    pass  # treat as missing, fall back to embedder

    # Compute embeddings for each painpoint, preferring cached.
    sigs = {}
    for p in painpoints:
        if p["id"] in cached:
            sigs[p["id"]] = cached[p["id"]]
        else:
            text = f"{p['title']} {p.get('description') or ''}".strip()
            sigs[p["id"]] = embedder.embed(text)

    edges = []
    ids = list(sigs.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            sim = _cosine_sim(sigs[a], sigs[b])
            if sim >= threshold:
                edges.append((a, b))

    by_id = {p["id"]: p for p in painpoints}
    components = _components(ids, edges)
    return [[by_id[i] for i in comp] for comp in components]


def category_member_titles(conn, category_id):
    """All painpoint titles in a category, for use in clustering / merge tests."""
    rows = conn.execute(
        "SELECT id, title, description FROM painpoints WHERE category_id = ?",
        (category_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def inter_category_similarity(conn, cat_a_id, cat_b_id, embedder=None):
    """Cosine similarity between two categories' embedding vectors.

    If either category has no embedding in category_vec, falls back to
    computing max pairwise member similarity.

    Returns 0.0 if either category is empty.
    """
    # Try category embeddings first
    vec_a = conn.execute(
        "SELECT embedding FROM category_vec WHERE rowid = ?",
        (cat_a_id,),
    ).fetchone()
    vec_b = conn.execute(
        "SELECT embedding FROM category_vec WHERE rowid = ?",
        (cat_b_id,),
    ).fetchone()

    expected_bytes = EMBEDDING_DIM * 4
    if vec_a is not None and vec_b is not None:
        if len(vec_a[0]) == expected_bytes and len(vec_b[0]) == expected_bytes:
            try:
                emb_a = list(struct.unpack(f"{EMBEDDING_DIM}f", vec_a[0]))
                emb_b = list(struct.unpack(f"{EMBEDDING_DIM}f", vec_b[0]))
                return _cosine_sim(emb_a, emb_b)
            except struct.error:
                pass  # fall through to member-pair fallback
        # Dimension mismatch — fall through to pairwise member similarity

    # Fallback: max pairwise member similarity
    if embedder is None:
        embedder = FakeEmbedder()

    members_a = category_member_titles(conn, cat_a_id)
    members_b = category_member_titles(conn, cat_b_id)
    if not members_a or not members_b:
        return 0.0

    sigs_a = [
        embedder.embed(f"{p['title']} {p.get('description') or ''}".strip())
        for p in members_a
    ]
    sigs_b = [
        embedder.embed(f"{p['title']} {p.get('description') or ''}".strip())
        for p in members_b
    ]

    best = 0.0
    for sa in sigs_a:
        for sb in sigs_b:
            sim = _cosine_sim(sa, sb)
            if sim > best:
                best = sim
                if best >= 1.0:
                    return best
    return best
