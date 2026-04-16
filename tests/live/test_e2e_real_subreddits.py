"""LIVE end-to-end test on REAL subreddits.

Runs the full pipeline (scrape → persist → extract → promote) on a list
of actual subreddits using real Reddit + real OpenAI APIs. Prints the
resulting category tree with painpoints and source samples for manual
inspection.

Run manually:
    pytest tests/test_e2e_real_subreddits.py -v -s

Takes several minutes and costs a few cents in LLM tokens. Fails
gracefully on private/missing subreddits (logs a warning, continues).
"""

import asyncio
import os
import textwrap

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)

import db

# Debug-only: smaller scrape so we iterate faster. Revert before commit.
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


requires_api = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or "your-" in os.getenv("OPENAI_API_KEY", ""),
    reason="OPENAI_API_KEY required",
)


@requires_api
class TestRealSubreddits:

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path, monkeypatch):
        path = tmp_path / "real_sr.db"
        monkeypatch.setattr(db, "DB_PATH", path)
        db.init_db()

    def test_scrape_and_promote(self):
        print("\n" + "=" * 100)
        print("LIVE REAL-SUBREDDIT END-TO-END TEST")
        print("=" * 100)

        # ------------------------------------------------------------------
        # Run the full pipeline for each subreddit
        # ------------------------------------------------------------------
        summaries = []
        for sr in SUBREDDITS:
            print(f"\n{'-' * 100}")
            print(f"Analyzing r/{sr}...")
            print(f"{'-' * 100}")
            try:
                summary = asyncio.run(analyze(sr, min_score=0))
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                summaries.append({"subreddit": sr, "error": str(e)})
                continue
            summaries.append(summary)
            print(f"  posts_scraped:    {summary.get('posts_scraped', 0)}")
            print(f"  posts_persisted:  {summary.get('posts_persisted', 0)}")
            print(f"  comments:         {summary.get('comments_persisted', 0)}")
            print(f"  painpoints extracted: {summary.get('painpoints_extracted', 0)}")
            print(f"  painpoints linked:    {summary.get('painpoints_linked', 0)}")
            print(f"  painpoints dropped:   {summary.get('painpoints_dropped', 0)}")
            if summary.get("promote_error"):
                print(f"  promote_error:    {summary['promote_error']}")

        # ------------------------------------------------------------------
        # Aggregate summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 100)
        print("AGGREGATE SUMMARY")
        print("=" * 100)
        for s in summaries:
            if "error" in s:
                print(f"  r/{s['subreddit']:<30} ERROR: {s['error'][:70]}")
            else:
                print(f"  r/{s['subreddit']:<30} posts={s.get('posts_scraped', 0):>4} "
                      f"extracted={s.get('painpoints_extracted', 0):>3} "
                      f"linked={s.get('painpoints_linked', 0):>3} "
                      f"dropped={s.get('painpoints_dropped', 0):>3}")

        # ------------------------------------------------------------------
        # RUN THE CATEGORY WORKER SWEEP (real LLM naming)
        # ------------------------------------------------------------------
        print("\n" + "=" * 100)
        print("CATEGORY WORKER SWEEP (real LLM)")
        print("=" * 100)

        from category_worker import run_sweep
        from db.embeddings import OpenAIEmbedder
        from db.llm_naming import LLMNamer

        sweep_summary = run_sweep(namer=LLMNamer(), embedder=OpenAIEmbedder())
        print(f"  Sweep: {sweep_summary}")

        # ------------------------------------------------------------------
        # CATEGORY TREE (post-sweep)
        # ------------------------------------------------------------------
        print("\n" + "=" * 100)
        print("CATEGORY TREE AFTER SWEEP (categories → painpoints → sources)")
        print("=" * 100)

        conn = db.get_db()
        try:
            # Get the full tree structure
            roots = conn.execute(
                "SELECT id, name, description FROM categories "
                "WHERE parent_id IS NULL ORDER BY name"
            ).fetchall()

            for root in roots:
                root_member_count = _count_members_recursive(conn, root["id"])
                if root_member_count == 0:
                    continue   # skip empty roots
                print(f"\n📁 {root['name']} (id={root['id']}, total members: {root_member_count})")
                _print_category(conn, root["id"], indent=1)
        finally:
            conn.close()

        # ------------------------------------------------------------------
        # FLAT PAINPOINT DUMP (sorted by relevance) — easier to skim
        # ------------------------------------------------------------------
        print("\n" + "=" * 100)
        print("TOP PAINPOINTS BY RELEVANCE")
        print("=" * 100)

        conn = db.get_db()
        try:
            rows = conn.execute("""
                SELECT p.id, p.title, p.description, p.severity, p.signal_count, c.name AS cat, pc.name AS parent_cat
                FROM painpoints p
                LEFT JOIN categories c ON c.id = p.category_id
                LEFT JOIN categories pc ON pc.id = c.parent_id
                LIMIT 50
            """).fetchall()

            for r in rows:
                cat_path = (f"{r['parent_cat']} > {r['cat']}"
                            if r['parent_cat'] else (r['cat'] or "?"))
                print(f"\n  [{r['id']}] sig={r['signal_count']} sev={r['severity']} "
                      f"| {cat_path}")
                print(f"      TITLE: {r['title']}")
                if r['description']:
                    print(f"      DESC:  {textwrap.shorten(r['description'], 150)}")

                # Show up to 2 source samples
                sources = conn.execute("""
                    SELECT DISTINCT po.title AS post_title, po.permalink AS post_permalink,
                           po.subreddit, cm.body AS comment_body
                    FROM painpoint_sources ps
                    JOIN pending_painpoint_all_sources pps
                      ON pps.pending_painpoint_id = ps.pending_painpoint_id
                    JOIN posts po ON po.id = pps.post_id
                    LEFT JOIN comments cm ON cm.id = pps.comment_id
                    WHERE ps.painpoint_id = ?
                    LIMIT 2
                """, (r['id'],)).fetchall()
                for s in sources:
                    print(f"      📝 r/{s['subreddit']}: {textwrap.shorten(s['post_title'], 80)}")
                    if s['comment_body']:
                        print(f"         💬 {textwrap.shorten(s['comment_body'], 150)}")
        finally:
            conn.close()

        # ------------------------------------------------------------------
        # AUDIT LOG
        # ------------------------------------------------------------------
        print("\n" + "=" * 100)
        print("CATEGORY EVENTS (audit log)")
        print("=" * 100)

        conn = db.get_db()
        try:
            events = conn.execute(
                "SELECT event_type, accepted, metric_name, metric_value, threshold, "
                "reason, proposed_at FROM category_events ORDER BY id"
            ).fetchall()
            if not events:
                print("  (no category worker events — that's expected; pipeline doesn't "
                      "run the sweep, only the promoter)")
            for e in events:
                a = "ACCEPT" if e["accepted"] else "REJECT"
                print(f"  {e['event_type']:<25} {a:<8} "
                      f"{e['metric_name']}={e['metric_value']:.3f} "
                      f"(thr={e['threshold']:.3f})")
        finally:
            conn.close()

        # ------------------------------------------------------------------
        # INTEGRITY
        # ------------------------------------------------------------------
        print("\n" + "=" * 100)
        print("DATA INTEGRITY")
        print("=" * 100)

        conn = db.get_db()
        try:
            total_pp = conn.execute("SELECT COUNT(*) FROM painpoints").fetchone()[0]
            total_pending = conn.execute(
                "SELECT COUNT(*) FROM pending_painpoints"
            ).fetchone()[0]
            orphan = conn.execute(
                "SELECT COUNT(*) FROM pending_painpoints pp "
                "LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id "
                "WHERE ps.painpoint_id IS NULL"
            ).fetchone()[0]
            mismatches = conn.execute("""
                SELECT COUNT(*) FROM (
                  SELECT p.id FROM painpoints p
                  LEFT JOIN painpoint_sources ps ON ps.painpoint_id = p.id
                  GROUP BY p.id HAVING p.signal_count != COUNT(ps.pending_painpoint_id)
                )
            """).fetchone()[0]
            print(f"  Total painpoints:         {total_pp}")
            print(f"  Total pending (queue):    {total_pending}")
            print(f"  Orphan pendings:          {orphan}  "
                  f"{'(expected > 0 if promote errored)' if orphan else ''}")
            print(f"  signal_count mismatches:  {mismatches}")
        finally:
            conn.close()

        print("\n" + "=" * 100)
        print("END")
        print("=" * 100)


def _count_members_recursive(conn, cat_id):
    """Sum of painpoints directly in this category + all descendants."""
    direct = conn.execute(
        "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat_id,)
    ).fetchone()[0]
    children = conn.execute(
        "SELECT id FROM categories WHERE parent_id = ?", (cat_id,)
    ).fetchall()
    return direct + sum(_count_members_recursive(conn, c["id"]) for c in children)


def _print_category(conn, cat_id, indent):
    """Recursively print a category subtree with its painpoints + sources."""
    prefix = "  " * indent

    # Print painpoints directly in this category
    pps = conn.execute("""
        SELECT id, title, severity, signal_count
        FROM painpoints WHERE category_id = ?
    """, (cat_id,)).fetchall()

    for p in pps:
        print(f"{prefix}📌 [{p['id']}] {p['title'][:75]}")
        print(f"{prefix}    sig={p['signal_count']} sev={p['severity']}")

        # Show up to 2 source samples
        sources = conn.execute("""
            SELECT DISTINCT po.subreddit, po.title AS post_title, cm.body AS comment_body
            FROM painpoint_sources ps
            JOIN pending_painpoint_all_sources pps
              ON pps.pending_painpoint_id = ps.pending_painpoint_id
            JOIN posts po ON po.id = pps.post_id
            LEFT JOIN comments cm ON cm.id = pps.comment_id
            WHERE ps.painpoint_id = ? LIMIT 2
        """, (p["id"],)).fetchall()
        for s in sources:
            body = s["comment_body"] or s["post_title"]
            kind = "💬 comment" if s["comment_body"] else "📝 post"
            print(f"{prefix}    {kind} r/{s['subreddit']}: {textwrap.shorten(body, 100)}")

    # Recurse into children that have members
    children = conn.execute(
        "SELECT id, name FROM categories WHERE parent_id = ? ORDER BY name", (cat_id,)
    ).fetchall()
    for child in children:
        member_count = _count_members_recursive(conn, child["id"])
        if member_count == 0:
            continue
        print(f"{prefix}📁 {child['name']} ({member_count} member(s))")
        _print_category(conn, child["id"], indent=indent + 1)
