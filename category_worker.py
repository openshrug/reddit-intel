"""Top-level category worker -- periodic sweep that mutates the taxonomy.

Runs as a separate OS process from the promoter. Acquires the merge lock
once per sweep, runs five passes (Uncategorized -> split -> delete ->
merge -> reroute), each emitting events through the per-event acceptance
test in db/category_events.py. Idempotent: running twice in a row
produces no new *accepted* events on the second run.
"""

import logging

import db
from db.category_events import (
    apply_with_test,
    prefetch_llm_batch,
    propose_delete_events,
    propose_merge_events,
    propose_reroute_events,
    propose_split_events,
    propose_uncategorized_events,
)
from db.embeddings import FakeEmbedder
from db.llm_naming import LLMNamer
from db.locks import merge_lock

log = logging.getLogger(__name__)

WORKER_LOCK_TIMEOUT_SEC = 300


def run_sweep(namer=None, embedder=None):
    """Run one full taxonomy-maintenance sweep.

    Args:
        namer: an `LLMNamer`-style object. Tests pass a `FakeNamer`.
               If None, a real `LLMNamer` is constructed lazily.
        embedder: an embedder for clustering. Tests pass `FakeEmbedder()`.
                  If None, a FakeEmbedder is used (sweep clustering doesn't
                  need the OpenAI API -- it uses local similarity).

    Returns a summary dict with counts per event type and per outcome.
    """
    if namer is None:
        namer = LLMNamer()
    if embedder is None:
        embedder = FakeEmbedder()

    summary = {
        "uncategorized": {"proposed": 0, "accepted": 0},
        "split":         {"proposed": 0, "accepted": 0},
        "delete":        {"proposed": 0, "accepted": 0},
        "merge":         {"proposed": 0, "accepted": 0},
        "reroute":       {"proposed": 0, "accepted": 0},
    }

    conn = db.get_db()
    try:
        with merge_lock(conn, timeout=WORKER_LOCK_TIMEOUT_SEC):
            # Step 1 -- process Uncategorized
            # Batch: propose all events, fan out LLM naming calls in
            # parallel via ThreadPoolExecutor, then apply sequentially.
            # Turns N×(2-3s serial LLM calls) into ~(N/5)×wall-clock.
            uncat_events = list(propose_uncategorized_events(conn, embedder=embedder))
            prefetch_llm_batch(conn, uncat_events, namer)
            for event in uncat_events:
                summary["uncategorized"]["proposed"] += 1
                if apply_with_test(conn, event, namer):
                    summary["uncategorized"]["accepted"] += 1

            # Step 2 -- split crowded categories (LLM-decided in propose,
            # no LLM call in apply — nothing to prefetch here).
            for event in list(propose_split_events(conn, embedder=embedder, namer=namer)):
                summary["split"]["proposed"] += 1
                if apply_with_test(conn, event, namer):
                    summary["split"]["accepted"] += 1

            # Step 3 -- delete dead categories (no LLM call in apply)
            for event in list(propose_delete_events(conn)):
                summary["delete"]["proposed"] += 1
                if apply_with_test(conn, event, namer):
                    summary["delete"]["accepted"] += 1

            # Step 4 -- merge duplicate sibling categories
            # (each merge makes one describe_merged_category LLM call —
            # parallelize across the batch.)
            merge_events = list(propose_merge_events(conn, embedder=embedder))
            prefetch_llm_batch(conn, merge_events, namer)
            for event in merge_events:
                summary["merge"]["proposed"] += 1
                if apply_with_test(conn, event, namer):
                    summary["merge"]["accepted"] += 1

            # Step 5 -- per-painpoint reroute (no LLM calls, pure embedding).
            # Handles singleton mis-routings the split step can't cluster.
            # Must run AFTER merges so the target centroids reflect final state.
            for event in list(propose_reroute_events(conn, embedder=embedder)):
                summary["reroute"]["proposed"] += 1
                if apply_with_test(conn, event, namer):
                    summary["reroute"]["accepted"] += 1
    finally:
        conn.close()

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db.init_db()
    summary = run_sweep()
    log.info("sweep complete: %s", summary)
