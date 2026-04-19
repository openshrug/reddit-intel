"""Snapshot the live ``trends.db`` after a pipeline stage.

Every invocation of ``quality_eval.run_pipeline`` allocates its own
*run directory* under ``runs/`` whose name is the slugified subreddit
list joined with an ``YYYYMMDD-HHMMSS`` timestamp -- e.g.
``runs/openclaw_claudeai_sideproject_20260419-101530/``. Per-stage
snapshots and the evaluator's ``report.md`` all live underneath that
single directory so multiple runs never overwrite each other.

A snapshot is a directory ``<run_dir>/<NN_label>/`` containing:

* ``trends.db``   -- file copy taken after a WAL checkpoint
* ``dump.md``     -- human-readable per-stage dump (see ``dump.py``)
* ``metrics.json``-- programmatic metrics + cross-snapshot delta
                     (see ``metrics.py``)

The snapshot DB file is never reopened in write mode; ``inspect_db``
opens it via ``db.get_db()`` for reads only. Keeping the live
``trends.db`` and the snapshot copies physically separate avoids any
chance of an inspect query mutating the running pipeline's DB.
"""

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import db

from . import dump, metrics

log = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).parent / "runs"


def create_run_dir(subreddits, *, now=None):
    """Allocate a fresh per-run directory under ``runs/``.

    The directory name is ``<sub1>_<sub2>_..._<YYYYMMDD-HHMMSS>`` so
    multiple runs (against the same or different subreddit lists)
    coexist on disk and sort by start time. The timestamp uses local
    time at second precision -- enough resolution for human runs while
    staying readable.

    Args:
        subreddits: iterable of subreddit names, in the order
            ``run_pipeline`` will analyze them.
        now: optional ``datetime`` for tests; defaults to
            ``datetime.now()``.

    Returns the created Path.
    """
    if now is None:
        now = datetime.now()
    slugs = "_".join(slug_subreddit(s) for s in subreddits)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    name = f"{slugs}_{stamp}" if slugs else stamp
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def take(label, *, run_dir, extras=None):
    """Snapshot the current ``db.DB_PATH`` under ``<run_dir>/<label>/``.

    Args:
        label: stage directory name, conventionally ``"NN_<slug>"`` so
            stages within a run sort chronologically.
        run_dir: the per-run directory previously created by
            ``create_run_dir``. All snapshots for one pipeline
            invocation share the same ``run_dir``.
        extras: optional dict passed through to ``dump.write_dump`` so
            stage-specific payloads (analyze() summary, sweep summary)
            land in the dump header.

    Returns the snapshot directory path.
    """
    out = run_dir / label
    out.mkdir(parents=True, exist_ok=True)

    snap_db = out / "trends.db"
    log.info("snapshot %s: copying %s -> %s", label, db.DB_PATH, snap_db)
    _checkpoint_and_copy(db.DB_PATH, snap_db)

    previous = _previous_snapshot_db(label, run_dir)
    log.info("snapshot %s: writing dump.md (prev=%s)", label,
             previous.parent.name if previous else "<none>")
    dump.write_dump(snap_db, out / "dump.md",
                    previous_path=previous, extras=extras)

    log.info("snapshot %s: writing metrics.json", label)
    metrics.write_metrics(snap_db, out / "metrics.json",
                          previous_path=previous)

    return out


def slug_subreddit(name):
    """Lowercase + sanitize a subreddit name for use in a filesystem
    path (so ``ClaudeAI`` -> ``claudeai``). Shared by ``create_run_dir``
    and the per-stage label builder in ``run_pipeline`` so both produce
    matching slugs."""
    return "".join(c.lower() if c.isalnum() else "_" for c in name)


def _checkpoint_and_copy(src, dst):
    """Force a WAL checkpoint on ``src`` then copy the DB file (+ wal/shm
    siblings, in case checkpoint left bytes behind).

    PRAGMA wal_checkpoint(TRUNCATE) flushes uncommitted WAL frames into
    the main DB file and truncates the WAL to zero, so a single
    ``shutil.copy2`` gets a self-consistent point-in-time snapshot.
    Falls back to copying the wal/shm siblings too if they still exist
    (defensive -- e.g. if SQLite couldn't truncate the WAL because
    another reader was attached).
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"source DB does not exist: {src}")

    conn = sqlite3.connect(src, timeout=30)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()

    shutil.copy2(src, dst)
    for suffix in ("-wal", "-shm"):
        sibling = src.with_name(src.name + suffix)
        if sibling.exists():
            shutil.copy2(sibling, dst.with_name(dst.name + suffix))


def _previous_snapshot_db(current_label, run_dir):
    """Find the snapshot directory inside ``run_dir`` that lexically
    precedes ``current_label`` (snapshots are named ``NN_*`` so sort =
    order), and return its ``trends.db`` path -- or ``None`` if none
    exists.
    """
    if not run_dir.exists():
        return None
    siblings = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name < current_label
    )
    for d in reversed(siblings):
        candidate = d / "trends.db"
        if candidate.exists():
            return candidate
    return None
