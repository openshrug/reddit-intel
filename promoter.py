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
    `promote_pending` call. One HTTP round-trip instead of N.
    """
    conn = db.get_db()
    try:
        ids = pick_unmerged_pending(conn)
    finally:
        conn.close()

    if not ids:
        return {"processed": 0, "linked": 0}

    if embedder is None:
        embedder = OpenAIEmbedder()

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

    linked = 0
    lock_timeouts = 0
    for pp_id in ordered_ids:
        emb = embedding_by_id[pp_id]
        try:
            promote_pending(pp_id, embedder=embedder, embedding=emb)
            linked += 1
        except TimeoutError as e:
            # The category worker is mid-sweep holding the merge_lock.
            # Skip this pp; it stays unmerged in the queue and the next
            # promoter pass will pick it up. Don't kill the daemon.
            lock_timeouts += 1
            log.warning(
                "promoter: merge_lock timeout on pp_id=%s (%s) — "
                "leaving in queue for next pass", pp_id, e,
            )

    return {
        "processed": len(ordered_ids),
        "linked": linked,
        "lock_timeouts": lock_timeouts,
    }


def run_forever(sleep_seconds=10):
    """Long-running daemon loop. Drains the pending queue, sleeps, repeats.

    Catches per-pass exceptions so transient failures (lock timeouts,
    embedding API errors, etc.) don't kill the daemon. Pending pps that
    weren't promoted stay in the queue for the next pass.
    """
    db.init_db()
    embedder = OpenAIEmbedder()
    while True:
        try:
            summary = run_once(embedder=embedder)
            if summary["processed"] > 0:
                log.info("promoter pass: %s", summary)
        except Exception as e:
            # Catch-all: the daemon must not die from a single bad pass.
            log.exception("promoter: run_once raised %s — sleeping and retrying",
                          type(e).__name__)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forever()
