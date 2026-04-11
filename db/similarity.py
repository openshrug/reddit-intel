"""Similarity layers for the painpoint promoter (§3 of the plan).

Two layers, both at insert time. Layer A is a SQL prefilter that asks "does
any source overlap with an existing merged painpoint?" Layer B is a
MinHash + LSH text-similarity match. Layer C (offline source-set Jaccard
reconciliation) was provably dead code under sticky-source semantics — the
plan §3.3 has the autopsy.
"""

import pickle
import re
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from . import DB_PATH, get_db

# Tunables — see §10 of the plan.
# SIM_THRESHOLD is the *promote-time* match threshold (Layer B). Set high
# (0.65) so we only link at promote time when titles are clearly the same;
# painpoints that are merely related land as singletons in Uncategorized
# and the category worker's sweep clusterer (SWEEP_CLUSTER_THRESHOLD = 0.40
# in db/category_events.py) groups them. Operating different thresholds at
# different layers is intentional — promote wants high precision, sweep
# clustering wants high recall. See db/category_events.py for the
# longer rationale.
SIM_THRESHOLD = 0.65
MINHASH_NUM_PERM = 128
SHINGLE_SIZE = 4   # character k-shingles


# ---------------------------------------------------------------------------
# MinHash signature helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _shingles(text):
    """Yield character k-shingles from a normalized text. Lowercased,
    whitespace-collapsed, then a sliding window of length SHINGLE_SIZE."""
    text = " ".join(_WORD_RE.findall((text or "").lower()))
    if len(text) < SHINGLE_SIZE:
        if text:
            yield text
        return
    for i in range(len(text) - SHINGLE_SIZE + 1):
        yield text[i : i + SHINGLE_SIZE]


def make_minhash(title, description=""):
    """Build a MinHash signature for a painpoint's `title + description`."""
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    text = f"{title or ''} {description or ''}".strip()
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def serialize_minhash(m):
    """Pickle a MinHash for storage in `painpoints.minhash_blob`."""
    return pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_minhash(blob):
    """Inverse of serialize_minhash; returns None if blob is None."""
    if blob is None:
        return None
    return pickle.loads(blob)


# ---------------------------------------------------------------------------
# Layer A — exact source SQL prefilter (§3.1)
# ---------------------------------------------------------------------------


def get_pending_sources(conn, pending_id):
    """Return the full source set of a pending painpoint as a list of
    `(post_id, comment_id)` tuples, unioning the primary columns and the
    extras junction (§7.5)."""
    rows = conn.execute(
        "SELECT post_id, comment_id FROM pending_painpoint_all_sources "
        "WHERE pending_painpoint_id = ?",
        (pending_id,),
    ).fetchall()
    return [(r["post_id"], r["comment_id"]) for r in rows]


def exact_source_lookup(conn, sources):
    """Layer A: which merged painpoints already cite *any* of these sources?

    Joins through `pending_painpoint_sources`-or-primary-columns via the
    `pending_painpoint_all_sources` view (§7.5). The COALESCE(.., -1) dance
    is the standard SQLite workaround for NULL not equalling NULL in tuple
    comparisons; comment_id is a positive autoincrement so -1 is safe as a
    sentinel.

    Returns a set of painpoint ids.
    """
    if not sources:
        return set()

    placeholders = ",".join("(?, COALESCE(?, -1))" for _ in sources)
    params = [v for s in sources for v in (s[0], s[1])]

    rows = conn.execute(
        f"""
        SELECT DISTINCT ps.painpoint_id
        FROM painpoint_sources ps
        JOIN pending_painpoint_all_sources pps
          ON pps.pending_painpoint_id = ps.pending_painpoint_id
        WHERE (pps.post_id, COALESCE(pps.comment_id, -1)) IN ({placeholders})
        """,
        params,
    ).fetchall()
    return {r["painpoint_id"] for r in rows}


# ---------------------------------------------------------------------------
# Layer B — text MinHash + LSH (§3.2, §3.4)
# ---------------------------------------------------------------------------


class PainpointLSH:
    """Wraps a datasketch.MinHashLSH that maps painpoint_id → MinHash.

    Persisted to a pickle file next to trends.db so cold start doesn't pay
    the rebuild cost. The lifecycle (§3.4):
      - rebuild from `painpoints` rows on first use
      - insert on each new painpoint
      - remove on each delete (only happens via merge_painpoints, §3.5)
      - remove + reinsert on rename (rename_painpoint is v2, §11)
    """

    def __init__(self, threshold=SIM_THRESHOLD, num_perm=MINHASH_NUM_PERM):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._signatures = {}   # id -> MinHash, kept for direct lookup / removal

    @classmethod
    def index_path_for(cls, db_path):
        return Path(str(db_path) + ".lsh.pkl")

    @classmethod
    def load_or_build(cls, conn, db_path=None):
        """Load from disk if a pickle exists, otherwise rebuild from the
        `painpoints` table.

        Either path leaves the index in sync with `painpoints` *as of the
        moment we load*. The promoter then keeps it in sync incrementally;
        the category worker rebuilds at the start of every sweep (per §3.4
        cross-process synchronization).
        """
        db_path = db_path or DB_PATH
        path = cls.index_path_for(db_path)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass   # corrupt or version-mismatched pickle → rebuild
        return cls.rebuild_from_db(conn)

    @classmethod
    def rebuild_from_db(cls, conn):
        idx = cls()
        rows = conn.execute(
            "SELECT id, title, description, minhash_blob FROM painpoints"
        ).fetchall()
        for r in rows:
            mh = deserialize_minhash(r["minhash_blob"])
            if mh is None:
                mh = make_minhash(r["title"], r["description"])
            idx._insert(r["id"], mh)
        return idx

    def persist(self, db_path=None):
        db_path = db_path or DB_PATH
        path = self.index_path_for(db_path)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---------- mutation ----------

    def _key(self, painpoint_id):
        return f"pp_{painpoint_id}"

    def _insert(self, painpoint_id, minhash):
        key = self._key(painpoint_id)
        if key in self.lsh:
            self.lsh.remove(key)
        self.lsh.insert(key, minhash)
        self._signatures[painpoint_id] = minhash

    def insert(self, painpoint_id, title, description=""):
        mh = make_minhash(title, description)
        self._insert(painpoint_id, mh)
        return mh

    def remove(self, painpoint_id):
        key = self._key(painpoint_id)
        if key in self.lsh:
            self.lsh.remove(key)
        self._signatures.pop(painpoint_id, None)

    def __contains__(self, painpoint_id):
        return painpoint_id in self._signatures

    def __len__(self):
        return len(self._signatures)

    # ---------- query ----------

    def query(self, title, description=""):
        """Return the set of painpoint ids whose signature matches above
        SIM_THRESHOLD."""
        mh = make_minhash(title, description)
        keys = self.lsh.query(mh)
        out = set()
        for k in keys:
            if k.startswith("pp_"):
                out.add(int(k[3:]))
        return out

    def jaccard(self, painpoint_id, title, description=""):
        """Estimated Jaccard similarity between an existing painpoint's
        signature and a new (title, description). Used for tie-breaking
        when LSH returns multiple candidates."""
        if painpoint_id not in self._signatures:
            return 0.0
        new_mh = make_minhash(title, description)
        return self._signatures[painpoint_id].jaccard(new_mh)
