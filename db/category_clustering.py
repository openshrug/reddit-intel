"""Clustering primitives for the category worker (§5.1 steps 1, 2, 4).

All operations reuse the §3.2 text-MinHash primitive — we don't bring in
embeddings or any other similarity tool, per the §11 "no vectors" rule.

- Intra-bucket clustering (steps 1 and 2): take a list of painpoints,
  build connected components in a similarity graph at SIM_THRESHOLD.
- Inter-category similarity (step 4): the maximum pairwise text MinHash
  similarity between any two members of the two categories.
"""

from .similarity import SIM_THRESHOLD, make_minhash


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


def cluster_painpoints(painpoints, threshold=SIM_THRESHOLD):
    """Group a list of painpoint dicts into clusters by text similarity.

    Each painpoint must have `id`, `title`, and `description` fields. Two
    painpoints are connected if their MinHash signatures match above
    `threshold`. Returns a list of clusters, each a list of painpoint dicts.

    Singletons are returned as 1-element clusters.
    """
    if not painpoints:
        return []
    if len(painpoints) == 1:
        return [list(painpoints)]

    sigs = {p["id"]: make_minhash(p["title"], p.get("description") or "") for p in painpoints}

    edges = []
    ids = list(sigs.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if sigs[a].jaccard(sigs[b]) >= threshold:
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


def inter_category_similarity(conn, cat_a_id, cat_b_id):
    """Max pairwise text MinHash similarity between any two members of the
    two categories. Used by sweep step 4 to decide merge_categories.

    Returns 0.0 if either category is empty.
    """
    members_a = category_member_titles(conn, cat_a_id)
    members_b = category_member_titles(conn, cat_b_id)
    if not members_a or not members_b:
        return 0.0

    sigs_a = [make_minhash(p["title"], p.get("description") or "") for p in members_a]
    sigs_b = [make_minhash(p["title"], p.get("description") or "") for p in members_b]

    best = 0.0
    for sa in sigs_a:
        for sb in sigs_b:
            j = sa.jaccard(sb)
            if j > best:
                best = j
                if best >= 1.0:
                    return best
    return best
