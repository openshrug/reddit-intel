"""LIVE end-to-end test — calls real OpenAI API for both embeddings AND
LLM naming. Costs a few cents. Run manually:

    pytest tests/test_e2e_live.py -v -s

Requires OPENAI_API_KEY in .env.
"""

import json
import os
import threading
import time

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)

import db
from db.posts import upsert_post
from db.painpoints import (
    get_uncategorized_id,
    promote_pending,
    save_pending_painpoint,
    add_pending_source,
)
from db.relevance import per_source_relevance
from db.embeddings import OpenAIEmbedder
from db.llm_naming import LLMNamer
from db.locks import merge_lock
from category_worker import run_sweep


requires_api = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or "your-" in os.getenv("OPENAI_API_KEY", ""),
    reason="OPENAI_API_KEY not set",
)


def _post(name, **kw):
    defaults = dict(
        subreddit="test", title=name, selftext="", permalink=f"/r/test/{name}",
        score=200, num_comments=50, upvote_ratio=0.95,
        created_utc=time.time() - 3600, is_self=True,
    )
    defaults.update(kw)
    defaults["name"] = name
    return upsert_post(defaults)


def _seed_pp_in_cat(cat_id, title, severity, age_days=1, score=300):
    pid = _post(f"t3_s_{hash(title) % 99999}", score=score, num_comments=80,
                created_utc=time.time() - age_days * 86400)
    pp_id = save_pending_painpoint(pid, title, severity=severity)
    conn = db.get_db()
    try:
        now = db._now()
        conn.execute(
            "INSERT INTO painpoints (title, description, severity, signal_count, "
            "category_id, first_seen, last_updated) "
            "VALUES (?, '', ?, 1, ?, ?, ?)",
            (title, severity, cat_id, now, now),
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
            "VALUES (?, ?)", (new_id, pp_id),
        )
        conn.commit()
        return new_id
    finally:
        conn.close()


@requires_api
class TestLiveEndToEnd:

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path, monkeypatch):
        path = tmp_path / "e2e_live.db"
        monkeypatch.setattr(db, "DB_PATH", path)
        db.init_db()

    def test_full_pipeline_real_embeddings_real_llm(self):
        embedder = OpenAIEmbedder()
        namer = LLMNamer()

        conn = db.get_db()
        try:
            parent_id = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
        finally:
            conn.close()

        print("\n" + "=" * 90)
        print("LIVE E2E TEST — real OpenAI embeddings + real LLM naming")
        print("=" * 90)

        # ==============================================================
        # 1. RELEVANCE DROP — verify old stuff dies
        # ==============================================================
        print("\n--- 1. Relevance drop ---")

        # Relevance-based drop was removed; every pending promotes. Print
        # the per-source relevance for inspection only.
        def _rel_for(post_id, sev):
            conn = db.get_db()
            try:
                p = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
                return per_source_relevance(p, None, severity=sev)
            finally:
                conn.close()

        ancient = _post("t3_ancient", score=0, num_comments=0,
                        created_utc=time.time() - 365 * 86400)
        pp_ancient = save_pending_painpoint(ancient, "Ancient irrelevant complaint", severity=1)
        result = promote_pending(pp_ancient, embedder=embedder)
        print(f"  Ancient (rel={_rel_for(ancient, 1):.4f}): "
              f"{'DROPPED' if result is None else f'KEPT → painpoint #{result}'}")

        fresh = _post("t3_fresh", score=500, num_comments=200)
        pp_fresh = save_pending_painpoint(fresh, "Fresh hot complaint about API latency", severity=9)
        result = promote_pending(pp_fresh, embedder=embedder)
        print(f"  Fresh   (rel={_rel_for(fresh, 9):.4f}): "
              f"{'DROPPED' if result is None else f'KEPT → painpoint #{result}'}")

        # ==============================================================
        # 2. EMBEDDING MERGE — similar painpoints collapse
        # ==============================================================
        print("\n--- 2. Embedding similarity merging ---")

        titles_same_topic = [
            "Kubernetes pod eviction during autoscaling causes downtime",
            "K8s pods get evicted whenever the cluster autoscaler kicks in",
            "Pod evictions happen during Kubernetes autoscale events",
            "Autoscaler in Kubernetes keeps evicting pods unexpectedly",
        ]

        merge_results = []
        for title in titles_same_topic:
            pid = _post(f"t3_k8s_{hash(title) % 99999}", score=300, num_comments=80)
            pp_id = save_pending_painpoint(pid, title, severity=7)
            result = promote_pending(pp_id, embedder=embedder)
            merge_results.append(result)
            print(f"  '{title[:60]}' → painpoint #{result}")

        unique_ids = set(merge_results)
        print(f"  → {len(titles_same_topic)} paraphrases → {len(unique_ids)} painpoint(s) "
              f"{'(MERGED — good!)' if len(unique_ids) == 1 else '(REVIEW)'}")

        # Completely different topic — should NOT merge with the K8s ones
        diff_pid = _post("t3_diff", score=300, num_comments=80)
        diff_pp = save_pending_painpoint(
            diff_pid, "PostgreSQL vacuum bloat causes query timeouts on large tables",
            severity=7,
        )
        diff_result = promote_pending(diff_pp, embedder=embedder)
        merged_with_k8s = diff_result in unique_ids
        print(f"  Postgres vacuum → painpoint #{diff_result} "
              f"{'(WRONGLY MERGED with K8s!)' if merged_with_k8s else '(separate — good!)'}")

        # ==============================================================
        # 3. CATEGORY ASSIGNMENT — new painpoints go to right categories
        # ==============================================================
        print("\n--- 3. Category assignment via embeddings ---")

        assignment_tests = [
            ("AI model hallucination in code generation tools", "AI Coding Tools"),
            ("LLM inference extremely slow on consumer GPUs", "LLM Infrastructure"),
            ("GitHub Actions CI pipeline keeps timing out", "CI/CD & DevOps"),
            ("OAuth2 token refresh has a race condition", "Auth & Identity"),
            ("Grafana dashboard query keeps timing out", "Observability"),
            ("Stripe payment webhook delivery is unreliable", "Payments"),
            ("MongoDB connection pool exhaustion under load", "Databases"),
        ]

        for title, expected in assignment_tests:
            pid = _post(f"t3_cat_{hash(title) % 99999}", score=200, num_comments=50)
            pp_id = save_pending_painpoint(pid, title, severity=7)
            result = promote_pending(pp_id, embedder=embedder)
            if result:
                conn = db.get_db()
                try:
                    row = conn.execute(
                        "SELECT c.name, p.name AS parent FROM painpoints pp "
                        "JOIN categories c ON c.id = pp.category_id "
                        "LEFT JOIN categories p ON p.id = c.parent_id "
                        "WHERE pp.id = ?", (result,)
                    ).fetchone()
                    cat_name = row["name"] if row else "NULL"
                    parent_name = row["parent"] if row else ""
                    full = f"{parent_name} > {cat_name}" if parent_name else cat_name
                finally:
                    conn.close()
                match = "OK" if cat_name == expected else "REVIEW"
                print(f"  {title[:55]:<55} → {full:<35} (want: {expected}) {match}")

        # ==============================================================
        # 4. MULTI-SOURCE — LLM batched extraction
        # ==============================================================
        print("\n--- 4. Multi-source pending painpoint ---")

        posts_multi = [_post(f"t3_multi_{i}", score=300, num_comments=80) for i in range(3)]
        pp_multi = save_pending_painpoint(
            posts_multi[0],
            "Docker build context too large in monorepo CI pipeline",
            severity=7,
        )
        add_pending_source(pp_multi, posts_multi[1])
        add_pending_source(pp_multi, posts_multi[2])
        multi_result = promote_pending(pp_multi, embedder=embedder)
        print(f"  3-source pending → painpoint #{multi_result}")

        conn = db.get_db()
        try:
            sources = conn.execute(
                "SELECT DISTINCT pps.post_id FROM painpoint_sources ps "
                "JOIN pending_painpoint_all_sources pps "
                "ON pps.pending_painpoint_id = ps.pending_painpoint_id "
                "WHERE ps.painpoint_id = ?", (multi_result,)
            ).fetchall()
            print(f"  Sources inherited: {len(sources)} post(s) "
                  f"{'(all 3 — good!)' if len(sources) == 3 else '(REVIEW)'}")
        finally:
            conn.close()

        # ==============================================================
        # 5. SEED SWEEP TARGETS + RUN SWEEP
        # ==============================================================
        print("\n--- 5. Sweep targets + sweep ---")

        # Dead category
        conn = db.get_db()
        try:
            dead_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('DeadCat-Legacy', ?, 'test', datetime('now')) RETURNING id",
                (parent_id,)
            ).fetchone()["id"]
            conn.commit()
        finally:
            conn.close()
        for i in range(3):
            pid = _post(f"t3_dead_{i}", score=2, num_comments=0,
                        created_utc=time.time() - 200 * 86400)
            pp_id = save_pending_painpoint(pid, f"Ancient legacy issue {i}", severity=1)
            _seed_pp_in_cat(dead_id, f"Ancient legacy issue {i}", 1,
                            age_days=200, score=2)
            # Backdate last_updated so the staleness-based delete sweep fires.
            from datetime import datetime, timedelta, timezone
            from db.category_events import CATEGORY_STALE_DAYS
            stale_ts = (
                datetime.now(timezone.utc)
                - timedelta(days=CATEGORY_STALE_DAYS + 5)
            ).isoformat()
            conn = db.get_db()
            try:
                conn.execute(
                    "UPDATE painpoints SET last_updated = ? "
                    "WHERE category_id = ?", (stale_ts, dead_id))
                conn.commit()
            finally:
                conn.close()
        print(f"  Dead category seeded: DeadCat-Legacy (3 stale members)")

        # Bloated category to split
        conn = db.get_db()
        try:
            bloat_id = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at, "
                "painpoint_count_at_last_check) "
                "VALUES ('MixedDevOps', ?, 'test', datetime('now'), 0) RETURNING id",
                (parent_id,)
            ).fetchone()["id"]
            conn.commit()
        finally:
            conn.close()
        docker_titles = [
            "Docker build context too large monorepo slow",
            "Docker build context monorepo excessive size problem",
            "Docker build context bloated in monorepo setup",
            "Docker monorepo build context size is massive",
            "Docker build context monorepo taking extremely long",
        ]
        npm_titles = [
            "npm install extremely slow on CI with many dependencies",
            "npm install takes ages on CI with huge package lock",
            "npm install slow in CI environment many deps",
            "npm install CI pipeline very slow for large project",
            "npm install slow on CI with massive node modules",
        ]
        for title in docker_titles + npm_titles:
            _seed_pp_in_cat(bloat_id, title, 7)
        print(f"  Bloated category seeded: MixedDevOps (10 members: Docker × 5 + npm × 5)")

        # Merge siblings
        conn = db.get_db()
        try:
            sib_a = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Redis-Caching-Issues', ?, 'test', datetime('now')) RETURNING id",
                (parent_id,)
            ).fetchone()["id"]
            sib_b = conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) "
                "VALUES ('Redis-Cache-Problems', ?, 'test', datetime('now')) RETURNING id",
                (parent_id,)
            ).fetchone()["id"]
            conn.commit()
        finally:
            conn.close()
        _seed_pp_in_cat(sib_a, "Redis cache eviction too aggressive in production", 7)
        _seed_pp_in_cat(sib_a, "Redis cache eviction aggressive production workload", 7)
        _seed_pp_in_cat(sib_b, "Redis cache eviction policy too aggressive prod env", 7)
        _seed_pp_in_cat(sib_b, "Redis cache eviction aggressive for production use", 7)
        print(f"  Merge siblings seeded: Redis-Caching-Issues (2) + Redis-Cache-Problems (2)")

        # Run the sweep with REAL LLM naming
        print(f"\n  Running sweep with real LLM...")
        summary = run_sweep(namer=namer, embedder=embedder)
        print(f"  Sweep: {json.dumps(summary, indent=4)}")

        # ==============================================================
        # 6. FULL STATE DUMP
        # ==============================================================
        print("\n--- 6. Full state ---")

        conn = db.get_db()
        try:
            print("\n  CATEGORIES WITH MEMBERS:")
            cats = conn.execute(
                "SELECT c.id, c.name, p.name AS parent FROM categories c "
                "LEFT JOIN categories p ON p.id = c.parent_id "
                "ORDER BY COALESCE(p.name, c.name), c.name"
            ).fetchall()
            for c in cats:
                n = conn.execute(
                    "SELECT COUNT(*) FROM painpoints WHERE category_id = ?",
                    (c["id"],)
                ).fetchone()[0]
                if n > 0 or c["name"] == "Uncategorized":
                    print(f"    [{c['id']:>3}] {c['name']:<45} "
                          f"parent={c['parent'] or '(root)':<25} members={n}")

            print(f"\n  ALL PAINPOINTS ({conn.execute('SELECT COUNT(*) FROM painpoints').fetchone()[0]}):")
            pps = conn.execute(
                "SELECT p.id, p.title, p.severity, p.signal_count, p.relevance, "
                "c.name AS cat FROM painpoints p "
                "LEFT JOIN categories c ON c.id = p.category_id "
                "ORDER BY p.relevance DESC NULLS LAST"
            ).fetchall()
            print(f"    {'id':>4} {'rel':>8} {'sig':>4} {'sev':>4} {'category':<35} title")
            print(f"    {'-'*4} {'-'*8} {'-'*4} {'-'*4} {'-'*35} {'-'*55}")
            for p in pps:
                r = f"{p['relevance']:.3f}" if p['relevance'] else "NULL"
                print(f"    {p['id']:>4} {r:>8} {p['signal_count']:>4} {p['severity']:>4} "
                      f"{(p['cat'] or 'NULL'):<35} {p['title'][:60]}")

            # Checks
            print("\n  CHECKS:")
            dead_row = conn.execute("SELECT id FROM categories WHERE id = ?", (dead_id,)).fetchone()
            print(f"    DeadCat-Legacy: {'DELETED' if dead_row is None else 'EXISTS'}")

            bloat_row = conn.execute("SELECT id FROM categories WHERE id = ?", (bloat_id,)).fetchone()
            print(f"    MixedDevOps: {'SPLIT+RETIRED' if bloat_row is None else 'EXISTS'}")

            sib_n = conn.execute(
                "SELECT COUNT(*) FROM categories WHERE id IN (?,?)", (sib_a, sib_b)
            ).fetchone()[0]
            print(f"    Redis siblings: {sib_n}/2 remaining "
                  f"{'(MERGED)' if sib_n == 1 else '(both exist)'}")

            uncat_n = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?",
                (get_uncategorized_id(conn=conn),)
            ).fetchone()[0]
            print(f"    In Uncategorized: {uncat_n}")

            print("\n  AUDIT LOG:")
            for e in conn.execute(
                "SELECT event_type, accepted, metric_name, metric_value, threshold, reason "
                "FROM category_events ORDER BY id"
            ).fetchall():
                a = "ACCEPT" if e["accepted"] else "REJECT"
                print(f"    {e['event_type']:<25} {a:<8} "
                      f"{e['metric_name']}={e['metric_value']:.3f} "
                      f"(thr={e['threshold']:.3f}) {e['reason']}")

            print(f"\n  INTEGRITY:")
            orphans = conn.execute(
                "SELECT COUNT(*) FROM pending_painpoints pp "
                "LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id "
                "WHERE ps.painpoint_id IS NULL"
            ).fetchone()[0]
            print(f"    Orphan pendings: {orphans}")
            mismatches = conn.execute(
                "SELECT COUNT(*) FROM painpoints p "
                "LEFT JOIN painpoint_sources ps ON ps.painpoint_id = p.id "
                "GROUP BY p.id HAVING p.signal_count != COUNT(ps.pending_painpoint_id)"
            ).fetchall()
            print(f"    signal_count mismatches: {len(mismatches)}")

        finally:
            conn.close()

        print("\n" + "=" * 90)
        print("END LIVE E2E")
        print("=" * 90)
