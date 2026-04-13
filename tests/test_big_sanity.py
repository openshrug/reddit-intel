"""Big sanity test — generates a realistic-scale dataset and prints the
full pipeline state for human inspection. No automatic asserts — run
with `pytest tests/test_big_sanity.py -v -s` and read the output.

Exercises EVERY component:
  - Relevance drop (old/weak painpoints get killed early)
  - Embedding similarity merging (similar painpoints collapse into one)
  - Category assignment via embedding tree traversal
  - Sweep: Uncategorized clustering → new categories (LLM-named)
  - Sweep: split a bloated category into sub-categories (LLM-named)
  - Sweep: delete dead categories (old, decayed)
  - Sweep: merge near-duplicate sibling categories
  - Stress: concurrent promoters don't deadlock or lose data
  - Final state inspection: categories, painpoints, relevance, audit log
"""

import json
import threading
import time

import pytest

import db
from db.posts import upsert_post, upsert_comment
from db.painpoints import (
    add_pending_source,
    get_uncategorized_id,
    promote_pending,
    save_pending_painpoint,
)
from db.relevance import per_source_relevance
from db.embeddings import FakeEmbedder
from db.llm_naming import FakeNamer
from db.locks import merge_lock
from category_worker import run_sweep


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    path = tmp_path / "big_sanity.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    yield path


def _post(name, **kw):
    defaults = dict(
        subreddit="test", title=name, selftext="", permalink=f"/r/test/{name}",
        score=200, num_comments=50, upvote_ratio=0.95,
        created_utc=time.time() - 3600, is_self=True,
    )
    defaults.update(kw)
    defaults["name"] = name
    return upsert_post(defaults)


def _seed_category(name, parent_id):
    conn = db.get_db()
    try:
        conn.execute(
            "INSERT INTO categories (name, parent_id, description, created_at) "
            "VALUES (?, ?, 'test', datetime('now'))", (name, parent_id),
        )
        cat_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        return cat_id
    finally:
        conn.close()


def _seed_painpoint_in_cat(cat_id, title, severity, post_id, pp_id, relevance=None):
    del relevance  # the cached relevance column was dropped; arg kept for callers
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


class TestBigSanity:

    def test_full_pipeline_big(self, fresh_db):
        embedder = FakeEmbedder()
        namer = FakeNamer()

        conn = db.get_db()
        try:
            parent_id = conn.execute(
                "SELECT id FROM categories WHERE parent_id IS NULL "
                "AND name != 'Uncategorized' LIMIT 1"
            ).fetchone()["id"]
        finally:
            conn.close()

        print("\n" + "=" * 90)
        print("BIG SANITY TEST — full pipeline, human inspection")
        print("=" * 90)

        # ==============================================================
        # 1. RELEVANCE DROP — old/weak painpoints should die early
        # ==============================================================
        print("\n--- Phase 1: Relevance drop ---")

        drop_candidates = [
            # (label, score, num_comments, age_days, severity, expect_drop)
            ("Ancient zero-engagement trivial", 0, 0, 365, 1, True),
            ("Old low-traction mild", 5, 2, 90, 2, True),
            ("Old moderate traction mild", 50, 10, 120, 3, True),
            ("Fresh high-traction severe", 500, 200, 0.5, 9, False),
            ("Fresh moderate traction moderate", 100, 30, 1, 6, False),
            ("Week-old decent traction", 80, 20, 7, 5, False),
            ("Month-old low traction severe", 10, 3, 30, 9, False),
            ("Very old but was viral", 2000, 500, 180, 7, True),
        ]

        # The relevance-based drop was removed from promote_pending; every
        # pending now promotes. Keep the scenario table for human inspection
        # (printed relevance + promote result) but drop the drop-expectation.
        drop_results = []
        for label, score, cmts, age_days, sev, _expect_drop in drop_candidates:
            pid = _post(f"t3_drop_{label[:10]}", score=score, num_comments=cmts,
                        created_utc=time.time() - age_days * 86400)
            pp_id = save_pending_painpoint(pid, f"Drop test: {label}", severity=sev)
            conn = db.get_db()
            try:
                post = conn.execute("SELECT * FROM posts WHERE id = ?", (pid,)).fetchone()
                rel = per_source_relevance(post, None, severity=sev)
            finally:
                conn.close()
            result = promote_pending(pp_id, embedder=embedder)
            drop_results.append((label, rel, result))

        print(f"  {'Label':<40} {'Relevance':>10} {'Promoted':>10}")
        print(f"  {'-'*40} {'-'*10} {'-'*10}")
        for label, rel, result in drop_results:
            print(f"  {label:<40} {rel:>10.4f} {'YES' if result else 'no':>10}")

        # ==============================================================
        # 2. EMBEDDING SIMILARITY — similar painpoints merge
        # ==============================================================
        print("\n--- Phase 2: Embedding similarity merging ---")

        merge_groups = [
            # Group A: should all merge into one (very similar)
            [
                "Kubernetes pod eviction during autoscaling events in production",
                "Kubernetes pod eviction during cluster autoscaling events",
                "Kubernetes pod eviction happening during autoscale cycles",
                "K8s pods being evicted when the cluster autoscales",
            ],
            # Group B: should all merge into one
            [
                "React Server Components hydration mismatch causing blank pages",
                "React Server Components hydration error leads to blank screen",
                "RSC hydration mismatch bug causing blank page render",
            ],
            # Group C: completely different from A and B
            [
                "PostgreSQL connection pool exhaustion under heavy API load",
                "Postgres connection pool running out during traffic spikes",
            ],
            # Singletons: should each stay separate
            [
                "Terraform state locking causes drift in multi-team setup",
            ],
            [
                "Flutter rendering glitch on Android tablets with custom fonts",
            ],
            [
                "Webpack 5 module federation memory leak in development mode",
            ],
        ]

        all_merged_ids = []
        for i, group in enumerate(merge_groups):
            group_ids = []
            for j, title in enumerate(group):
                pid = _post(f"t3_merge_{i}_{j}", score=300, num_comments=80)
                pp_id = save_pending_painpoint(pid, title, severity=7)
                result = promote_pending(pp_id, embedder=embedder)
                if result is not None:
                    group_ids.append(result)
            all_merged_ids.extend(group_ids)
            unique_ids = set(group_ids)
            print(f"  Group {i} ({len(group)} pps): merged into {len(unique_ids)} painpoint(s) "
                  f"{'OK' if len(unique_ids) <= 2 else 'REVIEW'}")

        conn = db.get_db()
        try:
            total_painpoints = conn.execute("SELECT COUNT(*) FROM painpoints").fetchone()[0]
        finally:
            conn.close()
        print(f"  Total painpoints after merging: {total_painpoints}")

        # ==============================================================
        # 3. CATEGORY ASSIGNMENT — new painpoints go to right categories
        # ==============================================================
        print("\n--- Phase 3: Category assignment via embeddings ---")

        # These should go to existing taxonomy categories (via find_best_category)
        assignment_tests = [
            ("AI model hallucination in code generation", "AI Coding Tools"),
            ("LLM inference speed on consumer GPUs", "LLM Infrastructure"),
            ("GitHub Actions CI timeout on large repos", "CI/CD & DevOps"),
            ("OAuth token refresh race condition", "Auth & Identity"),
            ("Grafana dashboard query timeout", "Observability"),
        ]

        for title, expected_cat in assignment_tests:
            pid = _post(f"t3_assign_{title[:10]}", score=200, num_comments=50)
            pp_id = save_pending_painpoint(pid, title, severity=7)
            result = promote_pending(pp_id, embedder=embedder)
            if result:
                conn = db.get_db()
                try:
                    row = conn.execute(
                        "SELECT c.name FROM painpoints p JOIN categories c ON c.id = p.category_id "
                        "WHERE p.id = ?", (result,)
                    ).fetchone()
                    actual = row["name"] if row else "NULL"
                finally:
                    conn.close()
                match = "OK" if actual == expected_cat else "REVIEW"
                print(f"  {title[:50]:<50} → {actual:<25} (expected: {expected_cat}) {match}")

        # ==============================================================
        # 4. SEED FOR SWEEP — dead categories, bloated, merge siblings
        # ==============================================================
        print("\n--- Phase 4: Seeding sweep targets ---")

        # Dead categories (old, low relevance)
        dead_cats = []
        for name in ["DeadCat-jQuery", "DeadCat-Flash", "DeadCat-Perl"]:
            cat_id = _seed_category(name, parent_id)
            for i in range(3):
                pid = _post(f"t3_dead_{name}_{i}", score=2, num_comments=0,
                            created_utc=time.time() - 200 * 86400)
                pp_id = save_pending_painpoint(pid, f"Ancient {name} issue {i}", severity=1)
                _seed_painpoint_in_cat(cat_id, f"Ancient {name} issue {i}", 1, pid, pp_id,
                                       relevance=0.001)
            dead_cats.append(cat_id)
            print(f"  Created dead category: {name} (3 stale members)")

        # Bloated category (two distinct sub-topics, 10 painpoints)
        bloated_id = _seed_category("BloatedMixed", parent_id)
        conn = db.get_db()
        try:
            conn.execute("UPDATE categories SET painpoint_count_at_last_check = 0 WHERE id = ?",
                         (bloated_id,))
            conn.commit()
        finally:
            conn.close()
        subtopic_a = [
            "Docker build context too large monorepo slow",
            "Docker build context monorepo excessive size",
            "Docker build context bloated in monorepo setup",
            "Docker monorepo build context size is huge",
            "Docker build context monorepo taking forever",
        ]
        subtopic_b = [
            "npm install extremely slow on CI with many deps",
            "npm install takes ages on CI with large package lock",
            "npm install slow CI environment many dependencies",
            "npm install CI pipeline very slow large project",
            "npm install slow on CI with huge node modules",
        ]
        for i, title in enumerate(subtopic_a + subtopic_b):
            pid = _post(f"t3_bloat_{i}", score=300, num_comments=80)
            pp_id = save_pending_painpoint(pid, title, severity=7)
            _seed_painpoint_in_cat(bloated_id, title, 7, pid, pp_id, relevance=7.0)
        print(f"  Created bloated category: BloatedMixed (10 members, 2 sub-topics)")

        # Merge siblings (near-identical categories)
        sib_a = _seed_category("K8s-Eviction-A", parent_id)
        sib_b = _seed_category("K8s-Eviction-B", parent_id)
        for i, title in enumerate([
            "Kubernetes pod eviction during autoscaling",
            "Kubernetes pod eviction autoscale event",
        ]):
            pid = _post(f"t3_sib_a_{i}", score=200, num_comments=50)
            pp_id = save_pending_painpoint(pid, title, severity=7)
            _seed_painpoint_in_cat(sib_a, title, 7, pid, pp_id, relevance=6.0)
        for i, title in enumerate([
            "Kubernetes pod eviction during scaling cycle",
            "Kubernetes pod eviction cluster scale event",
        ]):
            pid = _post(f"t3_sib_b_{i}", score=200, num_comments=50)
            pp_id = save_pending_painpoint(pid, title, severity=7)
            _seed_painpoint_in_cat(sib_b, title, 7, pid, pp_id, relevance=6.0)
        print(f"  Created merge siblings: K8s-Eviction-A (2 members) + K8s-Eviction-B (2 members)")

        # Uncategorized cluster (should form a new category)
        for i, title in enumerate([
            "Redis cache eviction too aggressive production",
            "Redis cache eviction aggressive production env",
            "Redis cache eviction too aggressive in prod",
            "Redis cache eviction production too aggressive",
            "Redis cache eviction aggressive for production",
        ]):
            pid = _post(f"t3_uncat_{i}", score=200, num_comments=50)
            pp_id = save_pending_painpoint(pid, title, severity=7)
            conn = db.get_db()
            try:
                uncat_id = get_uncategorized_id(conn=conn)
                _seed_painpoint_in_cat(uncat_id, title, 7, pid, pp_id, relevance=7.0)
            finally:
                conn.close()
        print(f"  Created Uncategorized cluster: 5 Redis cache painpoints")

        # ==============================================================
        # 5. STRESS — concurrent promoters
        # ==============================================================
        print("\n--- Phase 5: Concurrent promoter stress test ---")

        stress_pp_ids = []
        for i in range(30):
            pid = _post(f"t3_stress_{i}", score=200, num_comments=50)
            pp_id = save_pending_painpoint(
                pid, f"Stress test painpoint variant {i % 5} about topic {i % 5}",
                severity=7,
            )
            stress_pp_ids.append(pp_id)

        errors = []
        results = {0: [], 1: [], 2: []}

        def worker(wid, chunk):
            try:
                for pp_id in chunk:
                    results[wid].append(promote_pending(pp_id, embedder=embedder))
            except Exception as e:
                errors.append((wid, e))

        chunks = [stress_pp_ids[i::3] for i in range(3)]
        threads = [threading.Thread(target=worker, args=(i, chunks[i])) for i in range(3)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            if t.is_alive():
                print(f"  DEADLOCK: thread {t.name} hung!")
        elapsed = time.monotonic() - t0

        all_stress = sum(results.values(), [])
        print(f"  3 workers × 10 pps each: completed in {elapsed:.2f}s")
        print(f"  Errors: {len(errors)}")
        print(f"  Dropped: {sum(1 for r in all_stress if r is None)}")
        print(f"  Linked: {sum(1 for r in all_stress if r is not None)}")

        # ==============================================================
        # 6. RUN THE SWEEP
        # ==============================================================
        print("\n--- Phase 6: Category worker sweep ---")

        summary = run_sweep(namer=namer, embedder=embedder)
        print(f"  Sweep summary: {json.dumps(summary, indent=4)}")

        # ==============================================================
        # 7. FULL STATE DUMP
        # ==============================================================
        print("\n--- Phase 7: Full state inspection ---")

        conn = db.get_db()
        try:
            # Categories with members
            print("\n  CATEGORIES (with members):")
            cats = conn.execute(
                "SELECT c.id, c.name, p.name AS parent_name "
                "FROM categories c LEFT JOIN categories p ON p.id = c.parent_id "
                "ORDER BY COALESCE(p.name, c.name), c.name"
            ).fetchall()
            for c in cats:
                count = conn.execute(
                    "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (c["id"],)
                ).fetchone()[0]
                if count > 0 or c["name"] == "Uncategorized":
                    print(f"    [{c['id']:>3}] {c['name']:<45} "
                          f"parent={c['parent_name'] or '(root)':<25} members={count}")

            # Painpoints ordered by signal_count (relevance column dropped).
            print(f"\n  PAINPOINTS ({conn.execute('SELECT COUNT(*) FROM painpoints').fetchone()[0]} total):")
            pps = conn.execute(
                "SELECT p.id, p.title, p.severity, p.signal_count, "
                "c.name AS category "
                "FROM painpoints p LEFT JOIN categories c ON c.id = p.category_id "
                "ORDER BY p.signal_count DESC, p.id"
            ).fetchall()
            print(f"    {'id':>4} {'sig':>4} {'sev':>4} {'category':<35} title")
            print(f"    {'-'*4} {'-'*4} {'-'*4} {'-'*35} {'-'*50}")
            for p in pps:
                print(f"    {p['id']:>4} {p['signal_count']:>4} {p['severity']:>4} "
                      f"{(p['category'] or 'NULL'):<35} {p['title'][:55]}")

            # Dead categories — should be gone
            print("\n  DEAD CATEGORIES CHECK:")
            for cat_id in dead_cats:
                row = conn.execute("SELECT name FROM categories WHERE id = ?", (cat_id,)).fetchone()
                status = "DELETED (good)" if row is None else f"STILL EXISTS ({row['name']})"
                print(f"    cat_id={cat_id}: {status}")

            # Bloated category — should be split
            print("\n  BLOATED CATEGORY CHECK:")
            bloated_row = conn.execute(
                "SELECT name FROM categories WHERE id = ?", (bloated_id,)
            ).fetchone()
            if bloated_row is None:
                print(f"    BloatedMixed (id={bloated_id}): SPLIT AND RETIRED (good)")
            else:
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (bloated_id,)
                ).fetchone()[0]
                print(f"    BloatedMixed: STILL EXISTS with {remaining} members")

            # Merge siblings — one should be gone
            print("\n  MERGE SIBLINGS CHECK:")
            sib_count = conn.execute(
                "SELECT COUNT(*) FROM categories WHERE id IN (?, ?)", (sib_a, sib_b)
            ).fetchone()[0]
            print(f"    K8s-Eviction-A + B: {sib_count}/2 remaining "
                  f"({'MERGED (good)' if sib_count == 1 else 'BOTH EXIST'})")

            # Uncategorized — Redis cluster should be gone
            print("\n  UNCATEGORIZED CHECK:")
            uncat_id = get_uncategorized_id(conn=conn)
            in_uncat = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (uncat_id,)
            ).fetchone()[0]
            print(f"    Painpoints still in Uncategorized: {in_uncat}")

            # Auto-named categories
            print("\n  AUTO-NAMED CATEGORIES:")
            auto = conn.execute(
                "SELECT id, name FROM categories WHERE name LIKE 'AutoCat-%'"
            ).fetchall()
            for a in auto:
                count = conn.execute(
                    "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (a["id"],)
                ).fetchone()[0]
                print(f"    {a['name']}: {count} painpoints")
            if not auto:
                print("    (none — FakeNamer names weren't created, or cluster was too small)")

            # Audit log
            print("\n  AUDIT LOG:")
            events = conn.execute(
                "SELECT event_type, accepted, metric_name, metric_value, threshold, reason "
                "FROM category_events ORDER BY id"
            ).fetchall()
            for e in events:
                acc = "ACCEPT" if e["accepted"] else "REJECT"
                print(f"    {e['event_type']:<25} {acc:<8} "
                      f"{e['metric_name']}={e['metric_value']:.3f} "
                      f"(thr={e['threshold']:.3f}) {e['reason']}")
            if not events:
                print("    (no events)")

            # Data integrity
            print("\n  DATA INTEGRITY:")
            orphan_pending = conn.execute(
                "SELECT COUNT(*) FROM pending_painpoints pp "
                "LEFT JOIN painpoint_sources ps ON ps.pending_painpoint_id = pp.id "
                "WHERE ps.painpoint_id IS NULL"
            ).fetchone()[0]
            print(f"    Orphan pending painpoints: {orphan_pending}")

            mismatches = conn.execute(
                "SELECT p.id, p.signal_count, COUNT(ps.pending_painpoint_id) AS actual "
                "FROM painpoints p "
                "LEFT JOIN painpoint_sources ps ON ps.painpoint_id = p.id "
                "GROUP BY p.id HAVING p.signal_count != actual"
            ).fetchall()
            print(f"    signal_count mismatches: {len(mismatches)}")

            total_pp = conn.execute("SELECT COUNT(*) FROM painpoints").fetchone()[0]
            total_cats = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
            total_events = conn.execute("SELECT COUNT(*) FROM category_events").fetchone()[0]
            print(f"\n  SUMMARY: {total_pp} painpoints, {total_cats} categories, "
                  f"{total_events} audit events")

        finally:
            conn.close()

        print("\n" + "=" * 90)
        print("END BIG SANITY TEST — review the output above")
        print("=" * 90)
