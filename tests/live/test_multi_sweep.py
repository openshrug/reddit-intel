"""Multi-sweep diagnostic — measures the impact of the Phase-1 fixes.

Same workload as test_iterative_sweeps.py but snapshots the specific
metrics we're trying to move: Uncategorized %, empty-category count,
singleton count, and reroute volume. Compare against multi_sweep_out.txt
from before the fix.

Run:
    pytest tests/live/test_multi_sweep.py -v -s
"""

import asyncio
import os

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)

import db

import reddit_scraper as _rs
_rs.POSTS_PER_WINDOW = 30
_rs.POSTS_WITH_COMMENTS = 15

from subreddit_pipeline import analyze


SUBREDDITS = [
    "iOSAppsMarketing",
    "BuildToAttract",
    "rSocialskillsAscend",
    "AIIncomeLab",
    "AIToolsPromptWorkflow",
]
N_SWEEPS = 5


requires_api = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or "your-" in os.getenv("OPENAI_API_KEY", ""),
    reason="OPENAI_API_KEY required",
)


@requires_api
class TestMultiSweep:

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path, monkeypatch):
        path = tmp_path / "multi_sweep.db"
        monkeypatch.setattr(db, "DB_PATH", path)
        db.init_db()

    def test_multi_sweep(self):
        print("\n" + "=" * 100)
        print(f"MULTI-SWEEP DIAGNOSTIC (Phase-1 fixes, {N_SWEEPS} sweeps)")
        print("=" * 100)

        for sr in SUBREDDITS:
            print(f"\n--- Analyzing r/{sr} ---", flush=True)
            try:
                s = asyncio.run(analyze(sr, min_score=0))
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"  extracted={s.get('painpoints_extracted', 0)} "
                  f"linked={s.get('painpoints_linked', 0)}", flush=True)

        def snapshot(label):
            conn = db.get_db()
            try:
                total = conn.execute(
                    "SELECT COUNT(*) FROM painpoints"
                ).fetchone()[0]
                uncat_id = db.uncategorized_id(conn)
                uncat = conn.execute(
                    "SELECT COUNT(*) FROM painpoints WHERE category_id = ?",
                    (uncat_id,),
                ).fetchone()[0]
                cats = conn.execute(
                    """SELECT c.id, c.name, c.parent_id,
                              (SELECT COUNT(*) FROM painpoints p WHERE p.category_id = c.id) AS n
                       FROM categories c
                       WHERE c.name != 'Uncategorized'
                       ORDER BY n DESC"""
                ).fetchall()
                singletons = [c for c in cats if c["n"] == 1]
                empty = [c for c in cats if c["n"] == 0]
                roots = [c for c in cats if c["parent_id"] is None and c["n"] > 0]
                print(f"\n=== {label} ===")
                print(f"  total painpoints:    {total}")
                print(f"  in Uncategorized:    {uncat} ({100*uncat/max(total,1):.1f}%)")
                print(f"  total categories:    {len(cats)} (roots={len(roots)}, "
                      f"singletons={len(singletons)}, empty={len(empty)})")
                print("  top-10 by size:")
                for c in cats[:10]:
                    marker = "ROOT" if c["parent_id"] is None else "     "
                    print(f"    {marker}  n={c['n']:>3}  {c['name']}")
                print("  singleton categories:")
                for c in singletons[:15]:
                    marker = "ROOT" if c["parent_id"] is None else "     "
                    print(f"    {marker}  {c['name']}")
                if len(singletons) > 15:
                    print(f"    ... and {len(singletons) - 15} more")
            finally:
                conn.close()

        def snapshot_events(n):
            conn = db.get_db()
            try:
                rows = conn.execute(
                    """SELECT event_type, accepted, COUNT(*) AS n
                       FROM category_events
                       GROUP BY event_type, accepted
                       ORDER BY event_type"""
                ).fetchall()
                print(f"  cumulative events after sweep {n}:")
                for r in rows:
                    a = "ACCEPT" if r["accepted"] else "REJECT"
                    print(f"    {r['event_type']:<25} {a:<7} {r['n']:>3}")
            finally:
                conn.close()

        snapshot("STATE AFTER PROMOTE (pre-sweep)")

        from category_worker import run_sweep
        from db.embeddings import OpenAIEmbedder
        from db.llm_naming import LLMNamer

        namer = LLMNamer()
        embedder = OpenAIEmbedder()

        for i in range(1, N_SWEEPS + 1):
            print(f"\n{'#' * 100}")
            print(f"# SWEEP {i}/{N_SWEEPS}")
            print(f"{'#' * 100}", flush=True)
            summary = run_sweep(namer=namer, embedder=embedder)
            print(f"  sweep summary: {summary}", flush=True)
            snapshot(f"STATE AFTER SWEEP {i}")
            snapshot_events(i)

        print(f"\n{'=' * 100}")
        print("TOP-SIGNAL PAINPOINTS STILL IN UNCATEGORIZED (sig × sev)")
        print(f"{'=' * 100}")
        conn = db.get_db()
        try:
            uncat_id = db.uncategorized_id(conn)
            rows = conn.execute(
                """SELECT id, title, severity, signal_count,
                          (signal_count * severity) AS score
                   FROM painpoints WHERE category_id = ?
                   ORDER BY score DESC LIMIT 30""",
                (uncat_id,),
            ).fetchall()
            for r in rows:
                print(f"  [{r['id']:>3}] sig={r['signal_count']:>2} "
                      f"sev={r['severity']:>2} score={r['score']:>3}  {r['title'][:80]}")
        finally:
            conn.close()

        print(f"\n{'=' * 100}")
        print("END MULTI-SWEEP DIAGNOSTIC")
        print(f"{'=' * 100}")
