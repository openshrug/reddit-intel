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

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MERGE_COSINE_THRESHOLD = 0.60   # above this -> merge into existing painpoint
# Pending-stage dedup threshold. Stricter than MERGE_COSINE_THRESHOLD
# because pending dedup is about paraphrases of the SAME observation
# (e.g. the LLM emits "Slow FastAPI startup" from one post and "FastAPI
# app takes forever to boot" from another — same complaint, different
# wording). Painpoint merge at 0.60 is a topic-level merge; pending
# merge at 0.65 is an observation-level merge, so we lean tighter to
# avoid collapsing genuinely distinct observations of adjacent topics.
PENDING_MERGE_THRESHOLD = 0.65
# Tuned empirically:
#   same-topic paraphrases (colloquial): 0.72-0.78 cosine sim
#   same-topic LLM-condensed titles:     0.55-0.70 cosine sim (shorter, more formal)
#   different topics:                    0.23-0.29 cosine sim
# 0.60 catches LLM-condensed duplicates while keeping different topics clearly separate.
CATEGORY_COSINE_THRESHOLD = 0.35  # below this for ALL categories -> Uncategorized
# Lowered from 0.45 after introducing ANCHOR_WEIGHT: anchored
# category_vec represents declared intent (a static, curated embedding),
# not drifted member noise — so a less-strict match threshold no longer
# rubber-stamps everything, and legitimate painpoints that share topic
# with a seed category can land in it. Original 0.30 failed *because*
# centroids drifted; now that drift is bounded by the anchor, 0.35 sits
# clear of the different-topic noise floor (0.23-0.29) while avoiding
# the overly-tight 0.45 that sent 64% of painpoints to Uncategorized
# in the live run.

# Category vector = ANCHOR_WEIGHT · anchor(name+desc) + (1-ANCHOR_WEIGHT) · mean(members).
# 0.85 (up from 0.7) tightens the anchor's authority after observing
# residual mis-routing: under 0.7 the 30% member contribution still
# let an "AI Coding Tools" bucket creep toward general AI-business
# painpoints. 0.15 member share keeps the blend responsive to
# well-routed members without letting drift take over.
ANCHOR_WEIGHT = 0.85


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
            # 60s per-request timeout — same reasoning as llm.get_client:
            # SDK default is 10min, which stalls parallel fan-outs when a
            # single call sticks. Embedding calls are typically <2s.
            self._client = openai.OpenAI(timeout=60.0)
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

    Four tables:
    - pending_painpoint_vec: one row per pending_painpoint, populated at
      save time so the extractor can dedupe against existing pendings
      (cross-batch and within-batch) before inserting a new row.
    - painpoint_vec: one row per painpoint, embedding of its text.
    - category_vec: the blended vector used for matching (see
      update_category_embedding). Queried by find_best_category.
    - category_anchor_vec: the stable anchor (embedding of
      name+description), recomputed only when the category's
      description changes. Blended with the member mean to produce
      category_vec.
    """
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS pending_painpoint_vec USING vec0("
        "    embedding float[1536] distance_metric=cosine"
        ")"
    )
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
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS category_anchor_vec USING vec0("
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


def store_pending_painpoint_embedding(conn, pending_id, embedding):
    """INSERT OR REPLACE into pending_painpoint_vec. Populated by the
    extractor at pending-save time so cross-batch dedup can query it.
    The promoter reads this first and only re-embeds on miss."""
    blob = _pack_embedding(embedding)
    conn.execute(
        "DELETE FROM pending_painpoint_vec WHERE rowid = ?", (pending_id,)
    )
    conn.execute(
        "INSERT INTO pending_painpoint_vec (rowid, embedding) VALUES (?, ?)",
        (pending_id, blob),
    )


def get_pending_painpoint_embedding(conn, pending_id):
    """Return a pending painpoint's stored embedding as a list of floats,
    or None if no row (pending inserted before pending_painpoint_vec
    existed, or the embedder was None at save time)."""
    row = conn.execute(
        "SELECT embedding FROM pending_painpoint_vec WHERE rowid = ?",
        (pending_id,),
    ).fetchone()
    if row is None:
        return None
    blob = row[0]
    if len(blob) != EMBEDDING_DIM * 4:
        return None
    return list(struct.unpack(f"{EMBEDDING_DIM}f", blob))


def find_most_similar_pending(conn, embedding, threshold=PENDING_MERGE_THRESHOLD):
    """KNN over pending_painpoint_vec. Returns (pending_id, cos) for the
    closest pending above `threshold`, or None.

    Used by save_pending_painpoints_batch to decide "is this new
    extraction already represented by an existing pending row?" — if so,
    we tack the new evidence onto that pending via pending_painpoint_sources
    instead of creating another near-duplicate row.
    """
    blob = _pack_embedding(embedding)
    rows = conn.execute(
        "SELECT rowid, distance FROM pending_painpoint_vec "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
        (blob,),
    ).fetchall()
    if not rows:
        return None
    cos = 1.0 - rows[0][1]
    if cos < threshold:
        return None
    return (rows[0][0], cos)


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


# ---------------------------------------------------------------------------
# Category anchor: static embedding of name + description.
# Stored separately from category_vec so it survives member-mean updates
# and only re-embeds when the description actually changes.
# ---------------------------------------------------------------------------


def _anchor_text(name, description):
    """Canonical text used to compute a category's anchor embedding."""
    desc = (description or "").strip()
    name = (name or "").strip()
    return f"{name} {desc}".strip()


def store_category_anchor(conn, category_id, name, description, embedder):
    """Compute and store the anchor embedding for a category.

    Call on category creation and on description changes. Idempotent —
    re-calling with the same name/description overwrites the same row.
    No-op if the combined text is empty (pathological seed rows).
    """
    text = _anchor_text(name, description)
    if not text:
        return
    emb = embedder.embed(text)
    blob = _pack_embedding(emb)
    conn.execute(
        "DELETE FROM category_anchor_vec WHERE rowid = ?", (category_id,)
    )
    conn.execute(
        "INSERT INTO category_anchor_vec (rowid, embedding) VALUES (?, ?)",
        (category_id, blob),
    )


def get_category_anchor(conn, category_id):
    """Return the category's anchor embedding as a list of floats, or
    None if there isn't one (older DBs, seed not yet bootstrapped)."""
    row = conn.execute(
        "SELECT embedding FROM category_anchor_vec WHERE rowid = ?",
        (category_id,),
    ).fetchone()
    if row is None:
        return None
    blob = row[0]
    expected_bytes = EMBEDDING_DIM * 4
    if len(blob) != expected_bytes:
        return None
    return list(struct.unpack(f"{EMBEDDING_DIM}f", blob))


def delete_category_anchor(conn, category_id):
    """Drop the anchor for a category being deleted/merged away."""
    conn.execute(
        "DELETE FROM category_anchor_vec WHERE rowid = ?", (category_id,)
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
    """Seed anchor embeddings for categories that don't have one yet,
    and propagate the anchor into category_vec so find_best_category
    can match against seed categories before any member lands.

    Uses `embed_batch` so all categories embed in a single API
    round-trip instead of N sequential ones.
    """
    rows = conn.execute("""
        SELECT c.id, c.name, c.description
        FROM categories c
        LEFT JOIN category_anchor_vec av ON av.rowid = c.id
        WHERE c.name != 'Uncategorized' AND av.rowid IS NULL
    """).fetchall()

    pending = []
    for cat in rows:
        text = _anchor_text(cat["name"], cat["description"])
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
        blob = _pack_embedding(emb)
        conn.execute(
            "DELETE FROM category_anchor_vec WHERE rowid = ?", (cat_id,)
        )
        conn.execute(
            "INSERT INTO category_anchor_vec (rowid, embedding) VALUES (?, ?)",
            (cat_id, blob),
        )
        # Populate category_vec via the blend logic so seeded categories
        # are immediately queryable. With no members yet the blend
        # degenerates to the anchor itself.
        update_category_embedding(conn, cat_id)


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


def _normalize(v):
    """In-place normalize to unit length. Returns v (or an all-zero v
    unchanged if the norm is zero — callers decide how to handle)."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm > 0:
        for i in range(len(v)):
            v[i] /= norm
    return v


# ---------------------------------------------------------------------------
# Incremental centroid state
#
# `categories.member_emb_sum_blob` holds the raw element-wise sum of every
# current member's embedding. `categories.member_emb_count` holds the
# member count. The normalized mean-of-members is derived on demand as
# `sum / count → normalize`.
#
# add_member_to_centroid / remove_member_from_centroid update the sum/count
# in O(EMBEDDING_DIM) instead of re-scanning every member's painpoint_vec
# row. update_category_embedding then re-blends anchor + derived mean
# without touching painpoint_vec at all — turns the hottest path from
# O(N members) into O(1) per mutation.
# ---------------------------------------------------------------------------


def _unpack_sum(blob):
    if blob is None:
        return [0.0] * EMBEDDING_DIM
    if len(blob) != EMBEDDING_DIM * 4:
        return [0.0] * EMBEDDING_DIM
    return list(struct.unpack(f"{EMBEDDING_DIM}f", blob))


def _pack_sum(sum_vec):
    # Same layout as _pack_embedding but without dimension-assertion
    # (the sum can be any magnitude; we only care about the shape).
    return struct.pack(f"{EMBEDDING_DIM}f", *sum_vec)


def _bump_centroid_meta(conn, category_id, member_set_changed=True):
    """Stamp category timestamps. `centroid_updated_at` fires on every
    cache update (feeds reroute-skip logic); `member_set_last_changed_at`
    fires only when real membership changed (feeds the staleness /
    delete-stale-category logic).

    Rebuilds from painpoint_vec are NOT member-set changes — they're
    cache repairs — so callers pass `member_set_changed=False` there.
    """
    now = _now_iso()
    if member_set_changed:
        conn.execute(
            "UPDATE categories SET centroid_updated_at = ?, "
            "member_set_last_changed_at = ? WHERE id = ?",
            (now, now, category_id),
        )
    else:
        conn.execute(
            "UPDATE categories SET centroid_updated_at = ? WHERE id = ?",
            (now, category_id),
        )


def _now_iso():
    # Local import avoids a circular import at module load.
    from . import _now
    return _now()


def add_member_to_centroid(conn, category_id, member_embedding):
    """Increment the cached (sum, count) with one member's embedding.
    Caller is responsible for also writing painpoint_vec for the member
    and for calling `update_category_embedding` afterwards if they want
    category_vec refreshed in this transaction.
    """
    row = conn.execute(
        "SELECT member_emb_sum_blob, member_emb_count FROM categories "
        "WHERE id = ?", (category_id,),
    ).fetchone()
    if row is None:
        return
    sum_vec = _unpack_sum(row["member_emb_sum_blob"])
    for i in range(EMBEDDING_DIM):
        sum_vec[i] += member_embedding[i]
    new_count = (row["member_emb_count"] or 0) + 1
    conn.execute(
        "UPDATE categories SET member_emb_sum_blob = ?, member_emb_count = ? "
        "WHERE id = ?",
        (_pack_sum(sum_vec), new_count, category_id),
    )
    _bump_centroid_meta(conn, category_id)


def remove_member_from_centroid(conn, category_id, member_embedding):
    """Decrement the cached (sum, count) by one member's embedding.
    Clamps the count at 0 if something is off — the category might
    still have stale state from a legacy DB without the migration run.
    """
    row = conn.execute(
        "SELECT member_emb_sum_blob, member_emb_count FROM categories "
        "WHERE id = ?", (category_id,),
    ).fetchone()
    if row is None:
        return
    sum_vec = _unpack_sum(row["member_emb_sum_blob"])
    for i in range(EMBEDDING_DIM):
        sum_vec[i] -= member_embedding[i]
    new_count = max(0, (row["member_emb_count"] or 0) - 1)
    if new_count == 0:
        sum_vec = [0.0] * EMBEDDING_DIM
    conn.execute(
        "UPDATE categories SET member_emb_sum_blob = ?, member_emb_count = ? "
        "WHERE id = ?",
        (_pack_sum(sum_vec), new_count, category_id),
    )
    _bump_centroid_meta(conn, category_id)


def rebuild_centroid_from_members(conn, category_id):
    """Full recompute of (sum, count) from painpoint_vec. Used when a
    category's state needs to be rebuilt — bulk member moves (split
    source / merge loser / delete fallback), first-run migration for
    pre-existing categories, or as a correctness rescue if cached state
    has drifted.
    """
    expected_bytes = EMBEDDING_DIM * 4
    rows = conn.execute(
        """
        SELECT pv.embedding
        FROM painpoint_vec pv
        JOIN painpoints p ON p.id = pv.rowid
        WHERE p.category_id = ?
        """,
        (category_id,),
    ).fetchall()
    sum_vec = [0.0] * EMBEDDING_DIM
    count = 0
    for row in rows:
        blob = row["embedding"]
        if len(blob) != expected_bytes:
            continue
        try:
            emb = struct.unpack(f"{EMBEDDING_DIM}f", blob)
        except struct.error:
            continue
        for i in range(EMBEDDING_DIM):
            sum_vec[i] += emb[i]
        count += 1
    conn.execute(
        "UPDATE categories SET member_emb_sum_blob = ?, member_emb_count = ? "
        "WHERE id = ?",
        (_pack_sum(sum_vec) if count else None, count, category_id),
    )
    # Pure cache repair — do NOT claim the member set changed. Callers
    # that actually moved members have already stamped member_set_last_changed_at.
    _bump_centroid_meta(conn, category_id, member_set_changed=False)


def _member_mean_from_cache(conn, category_id):
    """Read the cached sum/count and derive the normalized mean. Returns
    None when the category has no members with embeddings. Falls back
    to a one-shot rebuild if the cache is empty but the category has
    members (legacy-DB case after the migration added the columns but
    before anyone populated them).
    """
    row = conn.execute(
        "SELECT member_emb_sum_blob, member_emb_count FROM categories "
        "WHERE id = ?", (category_id,),
    ).fetchone()
    if row is None:
        return None
    count = row["member_emb_count"] or 0
    if count == 0:
        # If there's a stale member that wasn't tracked (e.g. legacy
        # categories pre-migration), rebuild. Otherwise truly empty.
        has_members = conn.execute(
            "SELECT 1 FROM painpoints WHERE category_id = ? LIMIT 1",
            (category_id,),
        ).fetchone()
        if has_members is None:
            return None
        rebuild_centroid_from_members(conn, category_id)
        row = conn.execute(
            "SELECT member_emb_sum_blob, member_emb_count FROM categories "
            "WHERE id = ?", (category_id,),
        ).fetchone()
        count = row["member_emb_count"] or 0
        if count == 0:
            return None
    sum_vec = _unpack_sum(row["member_emb_sum_blob"])
    mean = [s / count for s in sum_vec]
    _normalize(mean)
    return mean


def update_category_embedding(conn, category_id, embedder=None):
    """Refresh a category's `category_vec` entry as a blend of its
    static anchor and the cached member mean.

    Blend keeps the declared intent in charge (see ANCHOR_WEIGHT
    docstring). Cheap now — derived from the cached (sum, count) rather
    than a JOIN over painpoint_vec. Callers that move members call
    `add_member_to_centroid` / `remove_member_from_centroid` before
    this, or `rebuild_centroid_from_members` for bulk changes.

    Precedence:
    - anchor AND mean → blend
    - anchor only → anchor
    - mean only (legacy without anchor) → mean
    - neither → remove category_vec row
    """
    del embedder  # unused — kept for API compatibility with older callers
    uncat_row = conn.execute(
        "SELECT id FROM categories WHERE name = 'Uncategorized'"
    ).fetchone()
    if uncat_row and uncat_row["id"] == category_id:
        conn.execute("DELETE FROM category_vec WHERE rowid = ?", (category_id,))
        return

    anchor = get_category_anchor(conn, category_id)
    mean = _member_mean_from_cache(conn, category_id)

    if anchor is None and mean is None:
        conn.execute(
            "DELETE FROM category_vec WHERE rowid = ?", (category_id,)
        )
        return

    if anchor is not None and mean is not None:
        blended = [
            ANCHOR_WEIGHT * anchor[i] + (1.0 - ANCHOR_WEIGHT) * mean[i]
            for i in range(EMBEDDING_DIM)
        ]
        _normalize(blended)
        store_category_embedding(conn, category_id, blended)
    elif anchor is not None:
        store_category_embedding(conn, category_id, anchor)
    else:
        store_category_embedding(conn, category_id, mean)
    conn.execute(
        "UPDATE categories SET centroid_updated_at = ? WHERE id = ?",
        (_now_iso(), category_id),
    )
