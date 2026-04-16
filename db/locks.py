"""Global merge lock for the painpoint ingest pipeline (§4 of
docs/_internal/PAINPOINT_INGEST_PLAN.md).

The promoter (db.painpoints.promote_pending) and the category worker
(category_worker.run_sweep) are separate OS processes. A Python
threading.Lock wouldn't span the boundary, so we use SQLite's
BEGIN IMMEDIATE — it acquires a reserved lock at the SQLite layer
that exactly one writer at a time can hold across processes.
"""

import contextlib
import sqlite3
import time


@contextlib.contextmanager
def merge_lock(conn, timeout=30):
    """Acquire SQLite's reserved-lock as a context manager.

    On entry: BEGIN IMMEDIATE, retrying every 50ms until `timeout` elapses.
    On normal exit: COMMIT.
    On exception: ROLLBACK and re-raise.

    Used by both the promoter and the category worker (§4 of the plan).
    Default timeout is the promoter's; the worker passes a longer one.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            break
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() and "busy" not in str(e).lower():
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"merge_lock timed out after {timeout}s waiting for SQLite reserved lock"
                ) from e
            time.sleep(0.05)

    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
