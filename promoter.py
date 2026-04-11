"""Top-level promoter loop — picks pending painpoints from the queue and
promotes them via db.painpoints.promote_pending (§3, §4 of the plan).

This is a thin runner around the promote_pending entry point. The real
work lives in db/painpoints.py and db/similarity.py. The promoter never
touches categories — that's the category worker's job (category_worker.py).
"""

import logging
import time

import db
from db.painpoints import promote_pending
from db.similarity import PainpointLSH

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


def run_once(lsh_index=None):
    """Drain the pending queue once. Returns counts."""
    conn = db.get_db()
    try:
        ids = pick_unmerged_pending(conn)
    finally:
        conn.close()

    if not ids:
        return {"processed": 0, "dropped": 0, "linked": 0}

    if lsh_index is None:
        conn = db.get_db()
        try:
            lsh_index = PainpointLSH.load_or_build(conn)
        finally:
            conn.close()

    dropped = 0
    linked = 0
    for pp_id in ids:
        result = promote_pending(pp_id, lsh_index=lsh_index)
        if result is None:
            dropped += 1
        else:
            linked += 1

    return {"processed": len(ids), "dropped": dropped, "linked": linked}


def run_forever(sleep_seconds=10):
    """Long-running daemon loop. Drains the pending queue, sleeps, repeats."""
    db.init_db()
    lsh_index = None
    while True:
        if lsh_index is None:
            conn = db.get_db()
            try:
                lsh_index = PainpointLSH.load_or_build(conn)
            finally:
                conn.close()
        summary = run_once(lsh_index=lsh_index)
        if summary["processed"] > 0:
            log.info("promoter pass: %s", summary)
            lsh_index.persist()
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forever()
