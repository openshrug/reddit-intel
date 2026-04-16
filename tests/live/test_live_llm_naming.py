"""ONE-SHOT live test that calls the real OpenAI API through LLMNamer.

Run manually with:  pytest tests/test_live_llm_naming.py -v -s
Requires OPENAI_API_KEY in .env. Costs a few cents.

This test is NOT part of the regular suite — it's gated behind an env
check so `pytest tests/` doesn't accidentally burn API credits.
"""

import os
import time

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)

import db
from category_worker import run_sweep
from db.llm_naming import LLMNamer
from db.painpoints import get_uncategorized_id, promote_pending, save_pending_painpoint
from db.posts import upsert_post

requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("your-"),
    reason="OPENAI_API_KEY not set — skipping live LLM test",
)


def _make_post(name, *, score=200, num_comments=50, age_seconds=3600, title="X"):
    return upsert_post({
        "name": name, "subreddit": "test", "title": title,
        "selftext": "", "permalink": f"/r/test/{name}",
        "score": score, "num_comments": num_comments, "upvote_ratio": 0.95,
        "created_utc": time.time() - age_seconds, "is_self": True,
    })


def _seed_painpoints_in_category(cat_id, titles, severity=7):
    for i, title in enumerate(titles):
        post_id = _make_post(f"t3_live_{cat_id}_{i}", score=300, num_comments=80, title=title)
        pp_id = save_pending_painpoint(post_id, title, severity=severity)
        conn = db.get_db()
        try:
            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated) VALUES (?, '', ?, 1, ?, ?, ?)",
                (title, severity, cat_id, now, now),
            )
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                "VALUES (?, ?)",
                (new_id, pp_id),
            )
            conn.commit()
        finally:
            conn.close()


def _seed_uncategorized(titles, severity=7):
    for i, title in enumerate(titles):
        post_id = _make_post(f"t3_uncat_{i}", score=200, num_comments=50, title=title)
        pp_id = save_pending_painpoint(post_id, title, severity=severity)
        conn = db.get_db()
        try:
            uncat_id = get_uncategorized_id(conn=conn)
            now = db._now()
            conn.execute(
                "INSERT INTO painpoints (title, description, severity, signal_count, "
                "category_id, first_seen, last_updated) VALUES (?, '', ?, 1, ?, ?, ?)",
                (title, severity, uncat_id, now, now),
            )
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                "VALUES (?, ?)",
                (new_id, pp_id),
            )
            conn.commit()
        finally:
            conn.close()


@requires_api_key
class TestLiveLLMNaming:
    """Calls the real OpenAI API. Exercises all LLM-naming paths:
    add_category_new (Uncategorized cluster), add_category_split
    (bloated category), and merge_categories (sibling merge with
    optional rename).
    """

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path, monkeypatch):
        path = tmp_path / "live_test.db"
        monkeypatch.setattr(db, "DB_PATH", path)
        db.init_db()

    def test_full_pipeline_with_real_llm(self):
        conn = db.get_db()
        try:
            parent_cat = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
        finally:
            conn.close()

        # --- 1. Uncategorized cluster → add_category_new ---
        # The LLM should name this something about Redis caching.
        _seed_uncategorized([
            "Redis cache eviction too aggressive production env",
            "Redis cache eviction aggressive production load",
            "Redis cache eviction too aggressive in production",
            "Redis cache eviction production aggressive policy",
            "Redis cache eviction aggressive for production workloads",
        ])

        # --- 2. Bloated category → add_category_split ---
        # Two sub-topics. The LLM should name them separately.
        conn = db.get_db()
        try:
            bloated_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Mixed DevOps Issues', ?, 'test', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]
            conn.execute(
                "UPDATE categories SET painpoint_count_at_last_check = 0 "
                "WHERE id = ?", (bloated_id,),
            )
            conn.commit()
        finally:
            conn.close()

        _seed_painpoints_in_category(bloated_id, [
            "GitHub Actions timeout monorepo build slow",
            "GitHub Actions timeout monorepo build hangs",
            "GitHub Actions timeout monorepo build fails",
            "GitHub Actions timeout monorepo build retry",
            "GitHub Actions timeout monorepo build flake",
            "Postgres connection pool exhausted high traffic",
            "Postgres connection pool exhausted heavy traffic",
            "Postgres connection pool exhausted traffic spikes",
            "Postgres connection pool exhausted traffic surge",
            "Postgres connection pool exhausted peak load now",
        ])

        # --- 3. Merge siblings → merge_categories ---
        conn = db.get_db()
        try:
            sib_a = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('K8s Pod Eviction', ?, 'test', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]
            sib_b = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Kubernetes Eviction Issues', ?, 'test', datetime('now')) RETURNING id",
                (parent_cat,),
            ).fetchone()["id"]
            conn.commit()
        finally:
            conn.close()

        _seed_painpoints_in_category(sib_a, [
            "Kubernetes pod eviction during autoscaling events",
            "Kubernetes pod eviction during cluster scaling",
        ])
        _seed_painpoints_in_category(sib_b, [
            "Kubernetes pod eviction during autoscaling cycles",
            "Kubernetes pod eviction during scaling actions",
        ])

        # --- 4. Promote one via the promoter with LLM-proposed category ---
        llm_post = _make_post("t3_llm_direct", score=400, num_comments=100)
        llm_pp = save_pending_painpoint(
            llm_post,
            "AI code completion hallucinating wrong imports",
            category_name="AI Coding Tools",
            severity=8,
        )
        promote_pending(llm_pp)

        # === RUN THE SWEEP WITH THE REAL LLM ===
        print("\n" + "=" * 80)
        print("LIVE LLM NAMING TEST — calling OpenAI API")
        print("=" * 80)

        real_namer = LLMNamer()
        summary = run_sweep(namer=real_namer)

        print(f"\nSweep summary: {summary}")

        # === PRINT THE RESULTS ===
        conn = db.get_db()
        try:
            print("\n--- Categories created / modified by the sweep ---")
            # New categories (not from taxonomy.yaml seed)
            cats = conn.execute(
                "SELECT c.id, c.name, c.parent_id, p.name AS parent_name "
                "FROM categories c LEFT JOIN categories p ON p.id = c.parent_id "
                "ORDER BY c.id"
            ).fetchall()
            for c in cats:
                members = conn.execute(
                    "SELECT COUNT(*) FROM painpoints WHERE category_id = ?",
                    (c["id"],),
                ).fetchone()[0]
                if members > 0 or c["name"] in ("Uncategorized",):
                    print(
                        f"  [{c['id']:>3}] {c['name']:<45} "
                        f"parent={c['parent_name'] or '(root)':<25} "
                        f"members={members}"
                    )

            print("\n--- Painpoints and their categories ---")
            pps = conn.execute(
                "SELECT p.id, p.title, p.severity, p.signal_count, "
                "c.name AS category "
                "FROM painpoints p "
                "LEFT JOIN categories c ON c.id = p.category_id "
                "ORDER BY c.name, p.title"
            ).fetchall()
            for p in pps:
                print(
                    f"  [{p['id']:>3}] sig={p['signal_count']:>3}  sev={p['severity']}  "
                    f"cat={p['category'] or 'NULL':<40}  "
                    f"{p['title'][:55]}"
                )

            print("\n--- Audit log ---")
            events = conn.execute(
                "SELECT event_type, accepted, metric_name, metric_value, "
                "threshold, reason "
                "FROM category_events ORDER BY id"
            ).fetchall()
            for e in events:
                acc = "ACCEPTED" if e["accepted"] else "REJECTED"
                print(
                    f"  {e['event_type']:<25} {acc:<10} "
                    f"{e['metric_name']}={e['metric_value']:.3f} "
                    f"(thr={e['threshold']:.3f})  {e['reason']}"
                )

            # --- Basic sanity checks (not exhaustive — this is for eyeballing) ---
            print("\n--- Sanity checks ---")

            # Uncategorized cluster should have produced a real name (not AutoCat)
            auto_cats = conn.execute(
                "SELECT name FROM categories WHERE name LIKE 'AutoCat-%'"
            ).fetchall()
            if auto_cats:
                print(f"  WARNING: FakeNamer names found: {[r['name'] for r in auto_cats]}")
            else:
                print("  OK: No FakeNamer placeholder names found")

            # The bloated category should be gone (split into sub-cats)
            bloated_row = conn.execute(
                "SELECT id FROM categories WHERE name = 'Mixed DevOps Issues'"
            ).fetchone()
            if bloated_row is None:
                print("  OK: 'Mixed DevOps Issues' was split and retired")
            else:
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM painpoints WHERE category_id = ?",
                    (bloated_row["id"],),
                ).fetchone()[0]
                print(f"  INFO: 'Mixed DevOps Issues' still exists with {remaining} members")

            # One of the merge siblings should be gone
            sib_count = conn.execute(
                "SELECT COUNT(*) FROM categories WHERE id IN (?, ?)",
                (sib_a, sib_b),
            ).fetchone()[0]
            print(f"  Merge siblings remaining: {sib_count}/2 "
                  f"({'OK — merged' if sib_count == 1 else 'INFO — both still exist'})")

            # The LLM-labeled painpoint should be in AI Coding Tools
            llm_cat = conn.execute(
                "SELECT c.name FROM painpoints p "
                "JOIN categories c ON c.id = p.category_id "
                "WHERE p.title LIKE '%hallucinating%'"
            ).fetchone()
            if llm_cat:
                print(f"  LLM-labeled painpoint landed in: '{llm_cat['name']}'")

            print("\n" + "=" * 80)
        finally:
            conn.close()
