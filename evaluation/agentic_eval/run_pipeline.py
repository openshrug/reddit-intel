"""Drive the full pipeline on a clean DB and snapshot every stage.

Each invocation allocates its own run directory under
``evaluation/agentic_eval/runs/`` whose name is
``<sub1>_<sub2>_..._<YYYYMMDD-HHMMSS>`` (see ``snapshot.create_run_dir``).
All per-stage snapshots and the evaluator's ``report.md`` live underneath
that single directory so multiple runs never overwrite each other.

Stages (matches the plan):

    00_clean             -- empty seeded DB
    01_openclaw          -- after analyze("openclaw")
    02_claudeai          -- after analyze("ClaudeAI")
    03_sideproject       -- after analyze("SideProject")
    04_post_sweep        -- after one run_sweep()

Each stage is captured by ``snapshot.take(label, run_dir=..., extras=...)``
which copies ``trends.db`` and writes ``dump.md`` + ``metrics.json``.
The existing pipeline modules (``subreddit_pipeline.analyze``,
``category_worker.run_sweep``) are reused verbatim -- this driver does
no pipeline work itself.

Run::

    python -m evaluation.agentic_eval.run_pipeline
    python -m evaluation.agentic_eval.run_pipeline --subreddits foo bar
    python -m evaluation.agentic_eval.run_pipeline --keep-existing-db
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

import db
from category_worker import run_sweep
from db.embeddings import OpenAIEmbedder
from db.llm_naming import LLMNamer
from subreddit_pipeline import analyze

from . import snapshot

log = logging.getLogger(__name__)

DEFAULT_SUBREDDITS = ["openclaw", "ClaudeAI", "SideProject"]


def main():
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description="Run the pipeline on a clean DB and snapshot every stage.",
    )
    parser.add_argument(
        "--subreddits", nargs="+", default=DEFAULT_SUBREDDITS,
        help="Subreddits to scrape, in order (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-existing-db", action="store_true",
        help="Skip the trends.db wipe (resume against the live DB). "
             "A fresh per-run dir is still created.",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Don't run the category worker at the end.",
    )
    parser.add_argument(
        "--min-score", type=int, default=0,
        help="Min post score forwarded to scrape_subreddit_full.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required (extraction + embeddings + sweep "
            "all call OpenAI). Set it in your .env."
        )

    run_dir = snapshot.create_run_dir(args.subreddits)
    log.info("Run dir: %s", run_dir)

    if not args.keep_existing_db:
        _wipe_live_db()
    db.init_db()
    snapshot.take("00_clean", run_dir=run_dir)

    summaries = []
    for i, sr in enumerate(args.subreddits, start=1):
        label = f"{i:02d}_{snapshot.slug_subreddit(sr)}"
        log.info("=" * 60)
        log.info("STAGE %s -- analyze(%r)", label, sr)
        log.info("=" * 60)
        try:
            summary = asyncio.run(analyze(sr, min_score=args.min_score))
        except Exception as exc:
            log.exception("analyze(%r) failed -- snapshotting anyway", sr)
            summary = {"subreddit": sr, "error": f"{type(exc).__name__}: {exc}"}
        summaries.append(summary)
        snapshot.take(label, run_dir=run_dir,
                      extras={"analyze_summary": summary})

    if args.skip_sweep:
        log.info("--skip-sweep set; not running category worker")
    else:
        label = f"{len(args.subreddits) + 1:02d}_post_sweep"
        log.info("=" * 60)
        log.info("STAGE %s -- run_sweep()", label)
        log.info("=" * 60)
        sweep_summary = run_sweep(namer=LLMNamer(), embedder=OpenAIEmbedder())
        snapshot.take(label, run_dir=run_dir,
                      extras={"sweep_summary": sweep_summary})

    log.info("Done. Snapshots in %s", run_dir)


def _wipe_live_db():
    """Delete ``trends.db`` (+ wal/shm siblings) so ``db.init_db()`` builds
    a fresh DB. Used at the start of every run unless --keep-existing-db
    was passed."""
    base = Path(db.DB_PATH)
    for p in (base, base.with_name(base.name + "-wal"),
              base.with_name(base.name + "-shm")):
        if p.exists():
            log.info("wiping %s", p)
            p.unlink()


if __name__ == "__main__":
    main()
