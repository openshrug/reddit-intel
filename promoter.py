"""Top-level promoter loop -- picks pending painpoints from the queue and
promotes them via db.painpoints.promote_pending.

This is a thin runner around the promote_pending entry point. The real
work lives in db/painpoints.py and db/embeddings.py. The promoter never
touches categories -- that's the category worker's job (category_worker.py).
"""

import logging
import time

import db
from db.painpoints import promote_pending
from db.embeddings import OpenAIEmbedder

log = logging.getLogger(__name__)


def pick_unmerged_pending(conn, limit=100):
    """Find pending painpoints that haven't been linked into the merged
    table yet."""
    rows = conn.execute(
        """
        SELECT pp.id
        FROM pending_painpoints pp
        LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id
        WHERE ps.painpoint_id IS NULL
        ORDER BY pp.id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [r["id"] for r in rows]


def run_once(embedder=None):
    """Drain the pending queue once. Returns counts.

    Uses batch embedding: fetches all pending pp texts, calls the embedder
    once with the full batch (OpenAI supports up to 2048 inputs per
    request), then passes the pre-computed embedding to each
    `promote_pending` call. Dramatically faster than per-pp sequential
    API calls (one round-trip instead of N).
    """
    conn = db.get_db()
    try:
        ids = pick_unmerged_pending(conn)
    finally:
        conn.close()

    if not ids:
        return {"processed": 0, "dropped": 0, "linked": 0}

    if embedder is None:
        embedder = OpenAIEmbedder()

    # Batch-fetch texts and embeddings for every pending pp up front.
    # This turns N sequential HTTP round-trips into ~ceil(N/256) batched ones.
    conn = db.get_db()
    try:
        rows = conn.execute(
            f"SELECT id, title, description FROM pending_painpoints "
            f"WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
    finally:
        conn.close()

    text_by_id = {
        r["id"]: f"{r['title']} {r['description'] or ''}".strip()
        for r in rows
    }
    ordered_ids = [pp_id for pp_id in ids if pp_id in text_by_id]
    texts = [text_by_id[pp_id] for pp_id in ordered_ids]

    log.info("promoter: batch-embedding %d pending painpoints", len(texts))
    embeddings = embedder.embed_batch(texts)
    embedding_by_id = dict(zip(ordered_ids, embeddings))

    dropped = 0
    linked = 0
    for pp_id in ids:
        emb = embedding_by_id.get(pp_id)
        result = promote_pending(pp_id, embedder=embedder, embedding=emb)
        if result is None:
            dropped += 1
        else:
            linked += 1

    return {"processed": len(ids), "dropped": dropped, "linked": linked}


def run_forever(sleep_seconds=10):
    """Long-running daemon loop. Drains the pending queue, sleeps, repeats."""
    db.init_db()
    embedder = OpenAIEmbedder()
    while True:
        summary = run_once(embedder=embedder)
        if summary["processed"] > 0:
            log.info("promoter pass: %s", summary)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forever()
