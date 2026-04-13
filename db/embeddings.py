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
MERGE_COSINE_THRESHOLD = 0.60   # above this -> merge into existing painpoint
# Tuned empirically:
#   same-topic paraphrases (colloquial): 0.72-0.78 cosine sim
#   same-topic LLM-condensed titles:     0.55-0.70 cosine sim (shorter, more formal)
#   different topics:                    0.23-0.29 cosine sim
# 0.60 catches LLM-condensed duplicates while keeping different topics clearly separate.
CATEGORY_COSINE_THRESHOLD = 0.45  # below this for ALL categories -> Uncategorized
# 0.45 sits clearly above the different-topics band (0.23-0.29) so
# Uncategorized fires on genuinely novel painpoints rather than
# rubber-stamping the nearest category from the noise floor.


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

    def _embed_with_retry(self, inputs, retries=2, backoff_base=4):
        """Call client.embeddings.create with exponential backoff on
        transient errors. Acquires the shared OpenAI concurrency
        semaphore (defined in llm.py) so embedding calls share the same
        in-flight budget as completion calls."""
        import logging
        import time
        from llm import OPENAI_API_SEMAPHORE
        log = logging.getLogger(__name__)
        client = self._get_client()
        last_exc = None
        for attempt in range(1 + retries):
            try:
                # Sleep between retries happens OUTSIDE the semaphore so
                # backing-off threads don't hog a slot.
                with OPENAI_API_SEMAPHORE:
                    return client.embeddings.create(
                        model=EMBEDDING_MODEL, input=inputs
                    )
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    raise
                delay = backoff_base * (2 ** attempt)
                log.warning(
                    "embed attempt %d/%d failed (%s), retrying in %ds",
                    attempt + 1, 1 + retries, exc, delay,
                )
                time.sleep(delay)
        raise last_exc

    @staticmethod
    def _sanitize(text):
        """Empty / whitespace-only strings are unspecified behavior for
        the OpenAI embeddings API (may error or return zero vector).
        Replace with a single space so the API always returns a valid
        embedding — callers never need to guard themselves."""
        if text is None:
            return " "
        t = text.strip()
        return t if t else " "

    def embed(self, text: str) -> list[float]:
        resp = self._embed_with_retry(self._sanitize(text))
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str], batch_size: int = 256) -> list[list[float]]:
        """Embed a list of texts in one API call (or multiple if the list
        is huge). Much faster than calling embed() in a loop — one HTTP
        round-trip instead of N.

        OpenAI accepts up to 2048 inputs per request; we chunk at 256
        to keep individual requests under the 8192-token per-input cap
        reasonably safe and to allow some concurrency in the server.
        """
        if not texts:
            return []
        sanitized = [self._sanitize(t) for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(sanitized), batch_size):
            chunk = sanitized[i : i + batch_size]
            resp = self._embed_with_retry(chunk)
            # resp.data is in request order
            out.extend(item.embedding for item in resp.data)
        return out


class FakeEmbedder:
    """Deterministic test double: hash each word to seed a random vector,
    take the normalized mean. Similar texts (sharing words) produce
    similar vectors. No API calls."""

    def embed(self, text: str) -> list[float]:
        # Treat empty/None text as a placeholder word so we produce a
        # deterministic non-zero vector. An all-zero vector matches every
        # other vector at cosine_sim=0 which is a bad default.
        if text is None or not text.strip():
            text = "__empty__"
        words = text.lower().split()
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

    def embed_batch(self, texts: list[str], batch_size: int = 256) -> list[list[float]]:
        return [self.embed(t) for t in texts]


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
    """Pack a list of floats into a BLOB for sqlite-vec.

    Asserts the dimension matches `EMBEDDING_DIM` so a wrong-shape
    vector (model switch, caller bug) fails loudly here instead of
    silently writing a malformed BLOB that get_painpoint_embedding
    later returns None for — which would make the painpoint invisible
    to downstream similarity search."""
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"embedding dimension mismatch: got {len(embedding)}, "
            f"expected {EMBEDDING_DIM} (check the embedding model)"
        )
    return struct.pack(f"{EMBEDDING_DIM}f", *embedding)


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


def get_painpoint_embedding(conn, painpoint_id):
    """Unpack a painpoint's embedding from painpoint_vec into a list of floats.
    Returns None if the painpoint has no embedding row."""
    row = conn.execute(
        "SELECT embedding FROM painpoint_vec WHERE rowid = ?", (painpoint_id,)
    ).fetchone()
    if row is None:
        return None
    blob = row[0]
    expected_bytes = EMBEDDING_DIM * 4
    if len(blob) != expected_bytes:
        return None
    return list(struct.unpack(f"{EMBEDDING_DIM}f", blob))


def iter_category_member_embeddings(conn, category_id):
    """Yield (painpoint_id, embedding_list) for every member of a category
    that has a valid embedding row. Skips rows with missing / wrong-size
    blobs. Used by the reroute step to compute leave-one-out centroids."""
    rows = conn.execute(
        "SELECT p.id, v.embedding FROM painpoints p "
        "JOIN painpoint_vec v ON v.rowid = p.id WHERE p.category_id = ?",
        (category_id,),
    ).fetchall()
    expected_bytes = EMBEDDING_DIM * 4
    for r in rows:
        blob = r["embedding"]
        if blob is None or len(blob) != expected_bytes:
            continue
        yield r["id"], list(struct.unpack(f"{EMBEDDING_DIM}f", blob))


def leave_one_out_centroid_sim(member_embeddings, target_id, target_embedding):
    """Cosine sim between a painpoint and the centroid of its category's
    OTHER members. Returns None when the category has no other members
    (singleton): caller should treat that as "no signal, always consider
    rerouting away".

    `member_embeddings` is a list of (pp_id, embedding) tuples — typically
    everything from iter_category_member_embeddings for the painpoint's
    current category. `target_id` / `target_embedding` identify the one to
    leave out."""
    accum = [0.0] * EMBEDDING_DIM
    count = 0
    for pp_id, emb in member_embeddings:
        if pp_id == target_id:
            continue
        for i in range(EMBEDDING_DIM):
            accum[i] += emb[i]
        count += 1
    if count == 0:
        return None
    norm = math.sqrt(sum(x * x for x in accum))
    if norm == 0:
        return None
    centroid = [x / norm for x in accum]
    return sum(a * b for a, b in zip(target_embedding, centroid))


def find_best_category_ranked(conn, embedding, limit=50):
    """KNN over category_vec, returning the top-K categories ranked by cosine
    similarity DESC. Unlike find_best_category this returns the full ranked
    list (no Uncategorized fallback, no bootstrap) — callers want to compare
    similarities across multiple categories (e.g. reroute logic)."""
    blob = _pack_embedding(embedding)
    rows = conn.execute(
        "SELECT rowid, distance FROM category_vec "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (blob, limit),
    ).fetchall()
    return [(r[0], 1.0 - r[1]) for r in rows]


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

    Uses `embed_batch` so all categories embed in a single API round-trip
    instead of N sequential ones (cold-start lock-hold drops from
    ~N×200ms to ~300ms total).
    """
    # Collect categories that need bootstrapping in one query.
    rows = conn.execute("""
        SELECT c.id, c.name, c.description
        FROM categories c
        LEFT JOIN category_vec cv ON cv.rowid = c.id
        WHERE c.name != 'Uncategorized' AND cv.rowid IS NULL
    """).fetchall()

    pending = []
    for cat in rows:
        text = f"{cat['name']} {cat['description'] or ''}".strip()
        if not text:
            continue
        pending.append((cat["id"], text))

    if not pending:
        return

    texts = [t for _, t in pending]
    embeddings = embedder.embed_batch(texts)
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"bootstrap_category_embeddings: embedder returned "
            f"{len(embeddings)} vectors for {len(texts)} inputs — "
            f"would silently mis-pair categories"
        )
    for (cat_id, _text), emb in zip(pending, embeddings):
        store_category_embedding(conn, cat_id, emb)


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
    """Local alias for db.uncategorized_id — kept private to avoid
    a circular import at module load (db.__init__ imports nothing
    from this module)."""
    from . import uncategorized_id
    return uncategorized_id(conn)


def update_category_embedding(conn, category_id, embedder=None):
    """Recompute a category's embedding as the mean of its members'
    painpoint embeddings.

    If the category has no members with embeddings, remove its entry
    from category_vec.

    Skips the Uncategorized sentinel — it's a dumping ground for
    heterogeneous painpoints with no semantic coherence, so a centroid
    would be a noisy vector that pulls future painpoints into
    Uncategorized instead of real categories.
    """
    # Never compute an embedding for the Uncategorized sentinel.
    uncat_row = conn.execute(
        "SELECT id FROM categories WHERE name = 'Uncategorized'"
    ).fetchone()
    if uncat_row and uncat_row["id"] == category_id:
        # Make sure we don't have a stale vec entry from before this guard.
        conn.execute("DELETE FROM category_vec WHERE rowid = ?", (category_id,))
        return

    # Pull every member's embedding in a single JOIN — was a per-member
    # SELECT (N+1) before; now one round-trip regardless of category size.
    expected_bytes = EMBEDDING_DIM * 4  # float32 = 4 bytes
    rows = conn.execute("""
        SELECT pv.rowid AS pp_id, pv.embedding
        FROM painpoint_vec pv
        JOIN painpoints p ON p.id = pv.rowid
        WHERE p.category_id = ?
    """, (category_id,)).fetchall()

    if not rows:
        conn.execute(
            "DELETE FROM category_vec WHERE rowid = ?", (category_id,)
        )
        return

    accum = [0.0] * EMBEDDING_DIM
    count = 0
    for row in rows:
        pp_id = row["pp_id"]
        blob = row["embedding"]
        if len(blob) != expected_bytes:
            # Dimension mismatch (stale embedding from a different model).
            # Skip this painpoint rather than crash the whole centroid calc.
            import logging
            logging.getLogger(__name__).warning(
                "painpoint_vec rowid=%s has %d bytes, expected %d "
                "(dimension mismatch — skipping in centroid)",
                pp_id, len(blob), expected_bytes,
            )
            continue
        try:
            emb = struct.unpack(f"{EMBEDDING_DIM}f", blob)
        except struct.error as e:
            import logging
            logging.getLogger(__name__).warning(
                "painpoint_vec rowid=%s unpack failed: %s — skipping", pp_id, e,
            )
            continue
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
