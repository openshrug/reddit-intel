"""Debug the sweep: run the full pipeline, then instrument the sweep
steps to print every metric and why decisions are made.

Run with:  pytest tests/test_sweep_debug.py -v -s
"""

import asyncio
import os
import textwrap

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)

import db
from subreddit_pipeline import analyze
from db.category_clustering import cluster_painpoints, category_member_titles
from db.category_events import (
    CATEGORY_STALE_DAYS,
    MIN_SUB_CLUSTER_SIZE,
    SPLIT_RECHECK_DELTA,
    MERGE_CATEGORY_THRESHOLD,
    UNCATEGORIZED_NAME,
    _category_last_activity,
)
from db.embeddings import (
    OpenAIEmbedder,
    CATEGORY_COSINE_THRESHOLD,
    MERGE_COSINE_THRESHOLD,
)


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
class TestSweepDebug:

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path, monkeypatch):
        path = tmp_path / "sweep_debug.db"
        monkeypatch.setattr(db, "DB_PATH", path)
        db.init_db()

    def test_debug_sweep(self):
        # ------------------------------------------------------------------
        # Run the pipeline first
        # ------------------------------------------------------------------
        for sr in SUBREDDITS:
            print(f"\nAnalyzing r/{sr}...")
            try:
                s = asyncio.run(analyze(sr, min_score=0))
                print(f"  extracted={s.get('painpoints_extracted')} "
                      f"linked={s.get('painpoints_linked')} "
                      f"dropped={s.get('painpoints_dropped')}")
            except Exception as e:
                print(f"  FAILED: {e}")

        embedder = OpenAIEmbedder()

        print("\n" + "=" * 100)
        print("SWEEP DEBUG — instrumented per-step analysis")
        print("=" * 100)
        print(f"\nTUNABLES:")
        print(f"  MERGE_COSINE_THRESHOLD    = {MERGE_COSINE_THRESHOLD}  (promote-time link + sweep clustering)")
        print(f"  CATEGORY_COSINE_THRESHOLD = {CATEGORY_COSINE_THRESHOLD}  (min sim for find_best_category)")
        print(f"  MIN_SUB_CLUSTER_SIZE      = {MIN_SUB_CLUSTER_SIZE}")
        print(f"  SPLIT_RECHECK_DELTA       = {SPLIT_RECHECK_DELTA}")
        print(f"  CATEGORY_STALE_DAYS       = {CATEGORY_STALE_DAYS}")
        print(f"  MERGE_CATEGORY_THRESHOLD  = {MERGE_CATEGORY_THRESHOLD}")

        conn = db.get_db()

        # ------------------------------------------------------------------
        # STEP 1 debug: Uncategorized clustering
        # ------------------------------------------------------------------
        print("\n" + "-" * 100)
        print("STEP 1: Process Uncategorized")
        print("-" * 100)
        uncat_id = conn.execute(
            "SELECT id FROM categories WHERE name = ?", (UNCATEGORIZED_NAME,)
        ).fetchone()["id"]
        uncat_members = [
            dict(r) for r in conn.execute(
                "SELECT id, title, description FROM painpoints WHERE category_id = ?",
                (uncat_id,),
            ).fetchall()
        ]
        print(f"  Uncategorized has {len(uncat_members)} painpoints")
        if len(uncat_members) > 0:
            clusters = cluster_painpoints(uncat_members, threshold=MERGE_COSINE_THRESHOLD,
                                          embedder=embedder)
            for i, c in enumerate(clusters):
                status = "ELIGIBLE" if len(c) >= MIN_SUB_CLUSTER_SIZE else "too small"
                print(f"  Cluster {i}: size={len(c)} [{status}]")
                for p in c[:3]:
                    print(f"    - {p['title'][:70]}")
        else:
            print("  (nothing to cluster — no step 1 events)")

        # ------------------------------------------------------------------
        # STEP 2 debug: Split check
        # ------------------------------------------------------------------
        print("\n" + "-" * 100)
        print("STEP 2: Split crowded categories")
        print("-" * 100)
        cats = conn.execute(
            "SELECT id, name, parent_id, painpoint_count_at_last_check FROM categories "
            "WHERE name != ? ORDER BY id", (UNCATEGORIZED_NAME,)
        ).fetchall()
        for cat in cats:
            current = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat["id"],)
            ).fetchone()[0]
            if current == 0:
                continue
            last = cat["painpoint_count_at_last_check"] or 0
            delta = current - last
            if delta < SPLIT_RECHECK_DELTA:
                print(f"  '{cat['name']:<35}' {current:>4} members, delta={delta} < {SPLIT_RECHECK_DELTA} "
                      f"→ skip (not grown enough)")
                continue

            members = category_member_titles(conn, cat["id"])
            clusters = cluster_painpoints(members, threshold=MERGE_COSINE_THRESHOLD,
                                          embedder=embedder)
            valid = [c for c in clusters if len(c) >= MIN_SUB_CLUSTER_SIZE]
            cluster_sizes = sorted([len(c) for c in clusters], reverse=True)

            print(f"\n  '{cat['name']}' ({current} members):")
            print(f"    Clusters found: {len(clusters)}, sizes={cluster_sizes[:10]}")
            print(f"    Valid (size ≥ {MIN_SUB_CLUSTER_SIZE}): {len(valid)}")
            if len(valid) >= 2:
                print(f"    → SPLIT would fire with {len(valid)} sub-clusters")
                for i, v in enumerate(valid[:4]):
                    print(f"      Sub-cluster {i} ({len(v)}):")
                    for p in v[:3]:
                        print(f"        - {p['title'][:75]}")
            else:
                print(f"    → reject: need ≥2 clusters of size ≥{MIN_SUB_CLUSTER_SIZE}")
                # Show what the top clusters look like to understand why
                for i, c in enumerate(sorted(clusters, key=len, reverse=True)[:3]):
                    print(f"      Top cluster {i} ({len(c)}):")
                    for p in c[:3]:
                        print(f"        - {p['title'][:75]}")

        # ------------------------------------------------------------------
        # STEP 3 debug: Delete check
        # ------------------------------------------------------------------
        print("\n" + "-" * 100)
        print("STEP 3: Delete dead categories")
        print("-" * 100)
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=CATEGORY_STALE_DAYS)
        for cat in cats:
            member_count = conn.execute(
                "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (cat["id"],)
            ).fetchone()[0]
            if member_count == 0:
                continue
            last = _category_last_activity(conn, cat["id"])
            if last is None:
                continue
            age_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400.0
            if last >= cutoff:
                print(f"  '{cat['name']:<35}' last_activity={age_days:.1f}d ago "
                      f"< {CATEGORY_STALE_DAYS} → skip (fresh)")
            else:
                print(f"  '{cat['name']:<35}' last_activity={age_days:.1f}d ago "
                      f"≥ {CATEGORY_STALE_DAYS} → would DELETE (stale)")

        # ------------------------------------------------------------------
        # STEP 4 debug: Merge sibling categories
        # ------------------------------------------------------------------
        print("\n" + "-" * 100)
        print("STEP 4: Merge sibling categories")
        print("-" * 100)
        from db.category_clustering import inter_category_similarity
        parents = conn.execute(
            "SELECT DISTINCT parent_id FROM categories WHERE parent_id IS NOT NULL"
        ).fetchall()
        for p in parents:
            siblings = conn.execute(
                "SELECT id, name FROM categories WHERE parent_id = ? AND name != ? "
                "ORDER BY id", (p["parent_id"], UNCATEGORIZED_NAME)
            ).fetchall()
            if len(siblings) < 2:
                continue
            for i in range(len(siblings)):
                for j in range(i + 1, len(siblings)):
                    a, b = siblings[i], siblings[j]
                    cnt_a = conn.execute(
                        "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (a["id"],)
                    ).fetchone()[0]
                    cnt_b = conn.execute(
                        "SELECT COUNT(*) FROM painpoints WHERE category_id = ?", (b["id"],)
                    ).fetchone()[0]
                    if cnt_a == 0 or cnt_b == 0:
                        continue
                    sim = inter_category_similarity(conn, a["id"], b["id"], embedder=embedder)
                    marker = "→ MERGE!" if sim >= MERGE_CATEGORY_THRESHOLD else ""
                    print(f"  {a['name']:<30} × {b['name']:<30} sim={sim:.3f} {marker}")

        # ------------------------------------------------------------------
        # MISPLACED PAINPOINTS: painpoints with weak similarity to their category
        # ------------------------------------------------------------------
        print("\n" + "-" * 100)
        print("NEW IDEA: painpoints with weak similarity to their assigned category")
        print("-" * 100)
        import struct
        from db.embeddings import EMBEDDING_DIM

        # For each painpoint, compute cosine sim between its embedding and its category's embedding
        rows = conn.execute("""
            SELECT p.id, p.title, p.category_id, c.name AS cat_name,
                   pv.embedding AS pp_emb, cv.embedding AS cat_emb
            FROM painpoints p
            JOIN categories c ON c.id = p.category_id
            LEFT JOIN painpoint_vec pv ON pv.rowid = p.id
            LEFT JOIN category_vec cv ON cv.rowid = p.category_id
            WHERE p.category_id != (SELECT id FROM categories WHERE name = 'Uncategorized')
        """).fetchall()

        import math

        def cosine_from_blobs(blob_a, blob_b):
            if blob_a is None or blob_b is None:
                return None
            a = list(struct.unpack(f"{EMBEDDING_DIM}f", blob_a))
            b = list(struct.unpack(f"{EMBEDDING_DIM}f", blob_b))
            dot = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            if na == 0 or nb == 0:
                return None
            return dot / (na * nb)

        mismatches = []
        for r in rows:
            sim = cosine_from_blobs(r["pp_emb"], r["cat_emb"])
            if sim is None:
                continue
            if sim < 0.35:   # arbitrary "weakly matched" threshold
                mismatches.append((sim, r["id"], r["title"], r["cat_name"]))

        mismatches.sort()
        print(f"\n  Painpoints with cosine sim < 0.35 to their category (top 30 worst):")
        for sim, pp_id, title, cat in mismatches[:30]:
            print(f"    [{pp_id}] sim={sim:.3f} '{title[:60]}' → category: {cat}")

        print(f"\n  Total misplaced painpoints: {len(mismatches)}")
        print(f"  If we move all these to Uncategorized and cluster, we might get new categories.")

        conn.close()

        print("\n" + "=" * 100)
        print("END DEBUG")
        print("=" * 100)
