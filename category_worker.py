"""Top-level category worker — periodic sweep that mutates the taxonomy
(§5 of docs/PAINPOINT_INGEST_PLAN.md).

Runs as a separate OS process from the promoter. Acquires the merge lock
once per sweep, runs four passes (Uncategorized → split → delete → merge),
each emitting events through the per-event acceptance test in
db/category_events.py. Idempotent: running twice in a row produces no new
*accepted* events on the second run, just possibly some rejected-event
audit rows.
"""

import logging

import db
from db.category_events import (
    apply_with_test,
    propose_delete_events,
    propose_merge_events,
    propose_split_events,
    propose_uncategorized_events,
)
from db.llm_naming import LLMNamer
from db.locks import merge_lock

log = logging.getLogger(__name__)

WORKER_LOCK_TIMEOUT_SEC = 300


def run_sweep(namer=None):
    """Run one full taxonomy-maintenance sweep (§5.1, §5.4).

    Args:
        namer: an `LLMNamer`-style object. Tests pass a `FakeNamer`.
               If None, a real `LLMNamer` is constructed lazily, which
               will hit the OpenAI API on first use — DO NOT call without
               an explicit namer in tests.

    Returns a summary dict with counts per event type and per outcome.
    """
    if namer is None:
        namer = LLMNamer()

    summary = {
        "uncategorized": {"proposed": 0, "accepted": 0},
        "split":         {"proposed": 0, "accepted": 0},
        "delete":        {"proposed": 0, "accepted": 0},
        "merge":         {"proposed": 0, "accepted": 0},
    }

    conn = db.get_db()
    try:
        with merge_lock(conn, timeout=WORKER_LOCK_TIMEOUT_SEC):
            # Step 1 — process Uncategorized
            for event in list(propose_uncategorized_events(conn)):
                summary["uncategorized"]["proposed"] += 1
                if apply_with_test(conn, event, namer):
                    summary["uncategorized"]["accepted"] += 1

            # Step 2 — split crowded categories
            for event in list(propose_split_events(conn)):
                summary["split"]["proposed"] += 1
                if apply_with_test(conn, event, namer):
                    summary["split"]["accepted"] += 1

            # Step 3 — delete dead categories
            for event in list(propose_delete_events(conn)):
                summary["delete"]["proposed"] += 1
                if apply_with_test(conn, event, namer):
                    summary["delete"]["accepted"] += 1

            # Step 4 — merge duplicate sibling categories
            for event in list(propose_merge_events(conn)):
                summary["merge"]["proposed"] += 1
                if apply_with_test(conn, event, namer):
                    summary["merge"]["accepted"] += 1
    finally:
        conn.close()

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db.init_db()
    summary = run_sweep()
    log.info("sweep complete: %s", summary)
