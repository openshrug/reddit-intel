"""Embedding-based similarity for the painpoint pipeline.

Replaces the MinHash/LSH approach with OpenAI embeddings stored in
sqlite-vec virtual tables. Two embedder classes:

- OpenAIEmbedder: production embedder calling the OpenAI API
- FakeEmbedder: deterministic test double (word-hash → random vector)

sqlite-vec vec0 tables use cosine distance metric so MATCH queries
return cosine distance (0 = identical, 2 = opposite). We convert to
similarity via: cosine_sim = 1 - cosine_distance.
"""

import hashlib
import math
import random
import struct
from typing import Optional

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MERGE_COSINE_THRESHOLD = 0.70   # above this -> merge into existing painpoint
# Tuned from real OpenAI text-embedding-3-small data:
#   same-topic paraphrases: 0.72-0.78 cosine sim
#   different topics:       0.23-0.29 cosine sim
# 0.70 catches paraphrases while keeping different topics cleanly separate.
CATEGORY_COSINE_THRESHOLD = 0.3  # below this for ALL categories -> Uncategorized


# ---------------------------------------------------------------------------
# Embedder classes
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """Production embedder that calls the OpenAI embeddings API."""

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def embed(self, text: str) -> list[float]:
        client = self._get_client()
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return resp.data[0].embedding


class FakeEmbedder:
    """Deterministic test double: hash each word to seed a random vector,
    take the normalized mean. Similar texts (sharing words) produce
    similar vectors. No API calls."""

    def embed(self, text: str) -> list[float]:
        words = text.lower().split()
        if not words:
            return [0.0] * EMBEDDING_DIM
        accum = [0.0] * EMBEDDING_DIM
        for word in words:
            h = hashlib.sha256(word.encode()).digest()
            rng = random.Random(h)
            for i in range(EMBEDDING_DIM):
                accum[i] += rng.gauss(0, 1)
        norm = math.sqrt(sum(x * x for x in accum))
        if norm > 0:
            accum = [x / norm for x in accum]
        return accum


# ---------------------------------------------------------------------------
# sqlite-vec helpers
# ---------------------------------------------------------------------------


def load_sqlite_vec(conn):
    """Load the sqlite-vec extension into a connection."""
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def init_vec_tables(conn):
    """Create the vec0 virtual tables if they don't exist.

    Must be called after sqlite_vec.load(conn).
    """
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS painpoint_vec USING vec0("
        "    embedding float[1536] distance_metric=cosine"
        ")"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS category_vec USING vec0("
        "    embedding float[1536] distance_metric=cosine"
        ")"
    )


def _pack_embedding(embedding):
    """Pack a list of floats into a BLOB for sqlite-vec."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def store_painpoint_embedding(conn, painpoint_id, embedding):
    """INSERT OR REPLACE into painpoint_vec."""
    blob = _pack_embedding(embedding)
    # vec0 tables use INSERT OR REPLACE via delete+insert
    conn.execute(
        "DELETE FROM painpoint_vec WHERE rowid = ?", (painpoint_id,)
    )
    conn.execute(
        "INSERT INTO painpoint_vec (rowid, embedding) VALUES (?, ?)",
        (painpoint_id, blob),
    )


def store_category_embedding(conn, category_id, embedding):
    """INSERT OR REPLACE into category_vec."""
    blob = _pack_embedding(embedding)
    conn.execute(
        "DELETE FROM category_vec WHERE rowid = ?", (category_id,)
    )
    conn.execute(
        "INSERT INTO category_vec (rowid, embedding) VALUES (?, ?)",
        (category_id, blob),
    )


def find_most_similar_painpoint(conn, embedding, exclude_ids=None):
    """KNN search against painpoint_vec.

    Returns (painpoint_id, cosine_sim) or None if the table is empty
    or no result above 0 similarity.
    """
    blob = _pack_embedding(embedding)
    rows = conn.execute(
        "SELECT rowid, distance FROM painpoint_vec "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT 10",
        (blob,),
    ).fetchall()

    exclude = set(exclude_ids or [])
    for row in rows:
        rid = row[0]  # rowid
        if rid in exclude:
            continue
        cosine_sim = 1.0 - row[1]  # distance is cosine distance
        return (rid, cosine_sim)
    return None


def bootstrap_category_embeddings(conn, embedder):
    """Seed embeddings for categories that have no vector in category_vec.

    Uses the category's `name + description` as the text to embed. This
    bootstraps taxonomy categories from seed (which have descriptions but
    no member painpoints yet). As painpoints are added to categories, the
    embeddings get overwritten by the mean-of-members approach in
    update_category_embedding, which is strictly better.
    """
    cats = conn.execute(
        "SELECT id, name, description FROM categories WHERE name != 'Uncategorized'"
    ).fetchall()
    for cat in cats:
        existing = conn.execute(
            "SELECT rowid FROM category_vec WHERE rowid = ?", (cat["id"],)
        ).fetchone()
        if existing is not None:
            continue
        text = f"{cat['name']} {cat['description'] or ''}".strip()
        if not text:
            continue
        emb = embedder.embed(text)
        store_category_embedding(conn, cat["id"], emb)


def find_best_category(conn, embedding, embedder=None):
    """Find the best-matching category by cosine similarity.

    If category_vec is empty and an embedder is provided, bootstraps
    category embeddings from their names + descriptions first (so seeded
    taxonomy categories are usable before any painpoints land in them).

    Falls back to Uncategorized if nothing scores above
    CATEGORY_COSINE_THRESHOLD.
    """
    blob = _pack_embedding(embedding)

    # Bootstrap category embeddings if none exist yet
    count = conn.execute("SELECT COUNT(*) FROM category_vec").fetchone()[0]
    if count == 0:
        if embedder is not None:
            bootstrap_category_embeddings(conn, embedder)
            count = conn.execute("SELECT COUNT(*) FROM category_vec").fetchone()[0]
        if count == 0:
            return _uncategorized_id(conn)

    # Find best overall category by embedding similarity
    rows = conn.execute(
        "SELECT rowid, distance FROM category_vec "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT 5",
        (blob,),
    ).fetchall()

    if not rows:
        return _uncategorized_id(conn)

    best_id = rows[0][0]
    best_sim = 1.0 - rows[0][1]

    if best_sim < CATEGORY_COSINE_THRESHOLD:
        return _uncategorized_id(conn)

    return best_id


def _uncategorized_id(conn):
    """Resolve the Uncategorized sentinel category id."""
    row = conn.execute(
        "SELECT id FROM categories WHERE name = 'Uncategorized'"
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "Uncategorized sentinel category missing -- db.init_db() not run?"
        )
    return row["id"]


def update_category_embedding(conn, category_id, embedder=None):
    """Recompute a category's embedding as the mean of its members'
    painpoint embeddings.

    If the category has no members with embeddings, remove its entry
    from category_vec.
    """
    # Get all painpoint embeddings for this category
    member_ids = conn.execute(
        "SELECT id FROM painpoints WHERE category_id = ?",
        (category_id,),
    ).fetchall()

    if not member_ids:
        conn.execute(
            "DELETE FROM category_vec WHERE rowid = ?", (category_id,)
        )
        return

    # Collect embeddings from painpoint_vec
    accum = [0.0] * EMBEDDING_DIM
    count = 0
    for row in member_ids:
        pp_id = row["id"]
        vec_row = conn.execute(
            "SELECT embedding FROM painpoint_vec WHERE rowid = ?",
            (pp_id,),
        ).fetchone()
        if vec_row is not None:
            emb = struct.unpack(f"{EMBEDDING_DIM}f", vec_row[0])
            for i in range(EMBEDDING_DIM):
                accum[i] += emb[i]
            count += 1

    if count == 0:
        conn.execute(
            "DELETE FROM category_vec WHERE rowid = ?", (category_id,)
        )
        return

    # Average and normalize
    for i in range(EMBEDDING_DIM):
        accum[i] /= count
    norm = math.sqrt(sum(x * x for x in accum))
    if norm > 0:
        accum = [x / norm for x in accum]

    store_category_embedding(conn, category_id, accum)
