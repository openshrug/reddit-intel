"""Test hypothesis: successive sweeps iteratively clean up hijacked categories.

Runs the full scrape/extract/promote pipeline once, then runs the category
worker sweep 3 times in a row. After each sweep prints:
  - sweep summary (proposed / accepted per event type)
  - top-10 categories by member count
  - top-5 painpoints in 'AI Coding Tools' (the known hijacked bucket)

If the hypothesis is right, the hijacked dating/relationship painpoints
should migrate out of 'AI Coding Tools' over successive sweeps (via split
→ new sub-category → delete of whatever stayed below relevance mass).

Run:
    pytest tests/test_iterative_sweeps.py -v -s
"""

import asyncio
import os
import textwrap

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)

import db
import reddit_scraper as _rs

_rs.POSTS_PER_WINDOW = 15
_rs.POSTS_WITH_COMMENTS = 6

from subreddit_pipeline import analyze

SUBREDDITS = [
    "iOSAppsMarketing",
    "BuildToAttract",
    "rSocialskillsAscend",
    "AIIncomeLab",
    "AIToolsPromptWorkflow",
]


requires_api = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or "your-" in os.getenv("OPENAI_API_KEY", ""),
    reason="OPENAI_API_KEY required",
)


@requires_api
class TestIterativeSweeps:

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path, monkeypatch):
        path = tmp_path / "iter_sweeps.db"
        monkeypatch.setattr(db, "DB_PATH", path)
        db.init_db()

    def test_three_sweeps(self):
        print("\n" + "=" * 100)
        print("ITERATIVE SWEEP TEST — scrape once, sweep 3x")
        print("=" * 100)

        # ------------------------------------------------------------------
        # Pipeline for every subreddit first
        # ------------------------------------------------------------------
        import traceback
        for sr in SUBREDDITS:
            print(f"\n--- Analyzing r/{sr} ---", flush=True)
            try:
                s = asyncio.run(analyze(sr, min_score=0))
                print(f"  posts={s.get('posts_scraped', 0)} "
                      f"extracted={s.get('painpoints_extracted', 0)} "
                      f"linked={s.get('painpoints_linked', 0)}", flush=True)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

        # ------------------------------------------------------------------
        # Three successive sweeps
        # ------------------------------------------------------------------
        from category_worker import run_sweep
        from db.embeddings import OpenAIEmbedder
        from db.llm_naming import LLMNamer

        for pass_num in (1, 2, 3):
            print("\n" + "=" * 100)
            print(f"SWEEP PASS {pass_num}/3")
            print("=" * 100, flush=True)

            # Reset the split-recheck trigger between passes so the LLM
            # re-evaluates every category every pass — otherwise passes 2/3
            # would skip splits because nothing grew between sweeps.
            if pass_num > 1:
                _reset_split_triggers()

            summary = run_sweep(namer=LLMNamer(), embedder=OpenAIEmbedder())
            print(f"\n  Summary: {summary}", flush=True)
            r = summary.get("reroute", {})
            print(f"  Reroute: proposed={r.get('proposed', 0)} "
                  f"accepted={r.get('accepted', 0)}", flush=True)

            # After each sweep: show top categories by size + the hijacked bucket
            _print_category_sizes(top_n=12)
            _print_hijacked_bucket_samples(top_k=5)

        # ------------------------------------------------------------------
        # Final tree + audit log
        # ------------------------------------------------------------------
        print("\n" + "=" * 100)
        print("FINAL CATEGORY EVENTS (audit log, all 3 sweeps)")
        print("=" * 100)
        conn = db.get_db()
        try:
            events = conn.execute(
                "SELECT event_type, accepted, metric_name, metric_value, threshold, "
                "reason FROM category_events ORDER BY id"
            ).fetchall()
            for e in events:
                a = "ACCEPT" if e["accepted"] else "REJECT"
                reason = (e["reason"] or "")[:60]
                print(f"  {e['event_type']:<22} {a:<7} "
                      f"{e['metric_name']}={e['metric_value']:.3f} "
                      f"(thr={e['threshold']:.3f}) {reason}")
        finally:
            conn.close()


def _reset_split_triggers():
    """Zero out painpoint_count_at_last_check so the next sweep's split
    proposer re-examines every category (SPLIT_RECHECK_DELTA=10 would
    otherwise gate out all of them after the first sweep)."""
    conn = db.get_db()
    try:
        conn.execute("UPDATE categories SET painpoint_count_at_last_check = 0")
        conn.commit()
    finally:
        conn.close()


def _print_category_sizes(top_n=10):
    print(f"\n  Top {top_n} categories by member count:", flush=True)
    conn = db.get_db()
    try:
        rows = conn.execute("""
            SELECT c.id, c.name, p.name AS parent_name,
                   (SELECT COUNT(*) FROM painpoints pp WHERE pp.category_id = c.id) AS cnt
            FROM categories c
            LEFT JOIN categories p ON p.id = c.parent_id
            ORDER BY cnt DESC LIMIT ?
        """, (top_n,)).fetchall()
        for r in rows:
            parent = r["parent_name"] or "(root)"
            print(f"    {r['cnt']:>4}  {parent} > {r['name']}", flush=True)
    finally:
        conn.close()


def _print_hijacked_bucket_samples(top_k=5):
    """Show the top painpoints in 'AI Coding Tools' by relevance — this is
    where dating/marketing hijacking was observed in prior runs."""
    conn = db.get_db()
    try:
        row = conn.execute(
            "SELECT id FROM categories WHERE name = 'AI Coding Tools'"
        ).fetchone()
        if row is None:
            print("\n  (no 'AI Coding Tools' category — maybe it got split/merged away)",
                  flush=True)
            return
        cat_id = row["id"]
        total = conn.execute(
            "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat_id,)
        ).fetchone()[0]
        print(f"\n  'AI Coding Tools' now has {total} members. Top {top_k} by signal:",
              flush=True)
        pps = conn.execute("""
            SELECT id, title, signal_count FROM painpoints
            WHERE category_id = ? LIMIT ?
        """, (cat_id, top_k)).fetchall()
        for p in pps:
            print(f"    [{p['id']}] sig={p['signal_count']}  "
                  f"{textwrap.shorten(p['title'], 80)}", flush=True)
    finally:
        conn.close()
