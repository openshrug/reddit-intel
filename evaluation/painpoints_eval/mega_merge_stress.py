"""Replay a real painpoint cluster against varying merge thresholds.

The classic case is ``pp #48`` (sig=20) from the openclaw / claudeai /
sideproject snapshot — five distinct OpenClaw pains collapsed onto
one painpoint. ``report.md`` Recommendation #2 says "cap mega-clusters
at sig <= 8 unless the LLM judge re-reads each pending"; this script
quantifies the structural side of the same recommendation by asking:

    "If MERGE_COSINE_THRESHOLD were T, would these N pendings still
     greedily cluster into one component, or split into K components
     of sizes [s_1, s_2, ...]?"

For each threshold in the swept range we:

1. Fetch every pending currently linked to the target painpoint (via
   ``painpoint_sources``) -- title + description joined the same way
   the engine joins them in ``save_pending_painpoints_batch`` (so we
   stress the *exact* embedding text the live pipeline produces).
2. Build a similarity graph: edge (i, j) iff cosine(i, j) >= T.
3. Compute connected components. Report:
     * total component count
     * component size histogram
     * per-pending component assignment, so you can see *which*
       components a target's mega-merge would split into.

The output is a single table per threshold plus a small ASCII split
trajectory (cluster count vs. T) to make the elbow obvious.

Usage::

    python -m evaluation.painpoints_eval.mega_merge_stress \\
        --snapshot evaluation/agentic_eval/snapshots/openclaw_claudeai_sideproject/04_post_sweep/trends.db

    python -m evaluation.painpoints_eval.mega_merge_stress \\
        --snapshot .../04_post_sweep/trends.db \\
        --painpoint 48 --start 0.55 --stop 0.80 --step 0.025
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from db.embeddings import OpenAIEmbedder

from ._util import cosine_sim

log = logging.getLogger(__name__)

DEFAULT_PAINPOINT = 48  # the report.md Dim 3 mega-merge poster child


# ---------------------------------------------------------------------------
# Snapshot fetch
# ---------------------------------------------------------------------------

def _fetch_pendings(painpoint_id: int) -> list[dict]:
    """Return every pending linked to ``painpoint_id`` in the
    currently-open snapshot DB. Caller must already be inside an
    ``inspect_db.open_snapshot()`` context.

    Each row carries: ``id, title, description, severity, subreddit,
    post_title``. The ``embed_text`` field reproduces the engine's
    join (``f"{title} {description}".strip()``) so the cosines below
    use identical inputs to the live pipeline.
    """
    from db.queries import get_painpoint_evidence
    rows = get_painpoint_evidence(painpoint_id)
    out = []
    for r in rows:
        title = (r.get("pending_title") or "").strip()
        desc = (r.get("pending_description") or "").strip()
        embed_text = f"{title} {desc}".strip()
        out.append({
            "id": r["pending_id"],
            "title": title,
            "description": desc,
            "severity": r.get("pending_severity"),
            "subreddit": r.get("subreddit"),
            "post_title": r.get("post_title"),
            "embed_text": embed_text,
        })
    return out


# ---------------------------------------------------------------------------
# Greedy connected-components clustering
# ---------------------------------------------------------------------------

def _components(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Plain union-find. Returns ``component[i]`` = canonical id."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return [find(i) for i in range(n)]


def cluster_at(threshold: float, vecs: list[list[float]]) -> list[int]:
    """Return component label per index for a given threshold."""
    n = len(vecs)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if cosine_sim(vecs[i], vecs[j]) >= threshold:
                edges.append((i, j))
    return _components(n, edges)


def cluster_summary(labels: list[int]) -> dict:
    """Group indices by label, sort by size desc."""
    by_label: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        by_label.setdefault(lbl, []).append(i)
    groups = sorted(by_label.values(), key=len, reverse=True)
    sizes = [len(g) for g in groups]
    return {
        "n_components": len(groups),
        "sizes": sizes,
        "groups": groups,
        "max_size": sizes[0] if sizes else 0,
        "n_singletons": sum(1 for s in sizes if s == 1),
    }


def _frange(start: float, stop: float, step: float) -> list[float]:
    out = []
    n = 0
    while True:
        v = round(start + n * step, 4)
        if v > stop + 1e-9:
            break
        out.append(v)
        n += 1
    return out


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def print_per_pending_table(pendings: list[dict],
                            labels_by_threshold: dict[float, list[int]],
                            sorted_thresholds: list[float]) -> None:
    """Print one row per pending, columns = thresholds, cells = component
    label (renumbered 1..K within each threshold for readability)."""
    print("\nPer-pending component membership")
    print("(component labels are renumbered 1..K independently per threshold;\n"
          "  -- means the pending is alone in its component at that threshold)")
    print()
    header = "  pp  | sub" + " " * 8 + "| title" + " " * 49 + "| " + " ".join(f"{t:>5.3f}" for t in sorted_thresholds)
    print(header)
    print("  ----+--------------+------------------------------------------------------+-" + "------" * len(sorted_thresholds))

    # Renumber labels per threshold by descending component size, so component 1
    # is always the biggest.
    renumbered: dict[float, dict[int, int]] = {}
    for thr in sorted_thresholds:
        labels = labels_by_threshold[thr]
        groups: dict[int, list[int]] = {}
        for i, lbl in enumerate(labels):
            groups.setdefault(lbl, []).append(i)
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))
        rmap = {orig: new + 1 for new, (orig, _) in enumerate(ordered)}
        renumbered[thr] = rmap

    for i, p in enumerate(pendings):
        cells = []
        for thr in sorted_thresholds:
            lbl = labels_by_threshold[thr][i]
            new_lbl = renumbered[thr][lbl]
            sizes = [len([x for x in labels_by_threshold[thr] if x == lbl])]
            if sizes[0] == 1:
                cells.append("  -- ")
            else:
                cells.append(f" {new_lbl:>3d} ")
        sub = (p["subreddit"] or "")[:12].ljust(12)
        title = _truncate(p["title"], 52).ljust(52)
        print(f"  #{p['id']:>3d} | {sub} | {title} | {' '.join(cells)}")


def print_threshold_table(sorted_thresholds: list[float],
                          summaries: list[dict]) -> None:
    print("\nCluster count vs. threshold")
    print("  threshold  components  largest  singletons  size histogram")
    for thr, s in zip(sorted_thresholds, summaries):
        sizes = s["sizes"]
        # Compact histogram: just list sizes, e.g. "[5,3,1,1]"
        sizes_str = "[" + ",".join(str(x) for x in sizes) + "]"
        print(f"  {thr:>9.4f}   {s['n_components']:>10d}   {s['max_size']:>6d}   {s['n_singletons']:>10d}   {sizes_str}")


def ascii_trajectory(sorted_thresholds: list[float],
                     summaries: list[dict],
                     n_pendings: int,
                     width: int = 50,
                     height: int = 14) -> str:
    """Tiny ASCII chart: y-axis = component count, x-axis = threshold."""
    if not sorted_thresholds:
        return ""
    n = len(sorted_thresholds)
    width = max(width, n)
    grid = [[" "] * width for _ in range(height)]

    def _x(i):
        return 0 if n == 1 else round(i * (width - 1) / (n - 1))

    def _y(v):
        v = max(1, min(n_pendings, v))
        # y=0 at top represents n_pendings (all singletons),
        # y=height-1 at bottom represents 1 component (mega-merge).
        return height - 1 - round((v - 1) * (height - 1) / max(1, n_pendings - 1))

    for i, s in enumerate(summaries):
        x, y = _x(i), _y(s["n_components"])
        grid[y][x] = "*"

    out = ["\nComponent count trajectory (* = one threshold)\n"]
    out.append(f"  y = #components,  range [1 .. {n_pendings}]\n\n")
    for y in range(height):
        v = n_pendings - round(y * (n_pendings - 1) / max(1, height - 1))
        out.append(f"  {v:>3d} | {''.join(grid[y])}\n")
    out.append("       +" + "-" * width + "\n")

    first = f"{sorted_thresholds[0]:.3f}"
    last = f"{sorted_thresholds[-1]:.3f}"
    line = list(" " * width)
    line[: len(first)] = list(first)
    line[width - len(last) :] = list(last)
    out.append("        " + "".join(line) + "\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description="Greedy-cluster a painpoint's linked pendings at varying thresholds.",
    )
    parser.add_argument(
        "--snapshot", required=True, type=Path,
        help="Path to a snapshot trends.db (typically .../04_post_sweep/trends.db).",
    )
    parser.add_argument(
        "--painpoint", type=int, default=DEFAULT_PAINPOINT,
        help="Painpoint id to stress (default: %(default)s — pp #48 mega-merge).",
    )
    parser.add_argument("--start", type=float, default=0.55)
    parser.add_argument("--stop", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.snapshot.exists():
        print(f"snapshot not found: {args.snapshot}", file=sys.stderr)
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
        return 1
    if args.start >= args.stop or args.step <= 0:
        print("invalid sweep range: need start < stop, step > 0", file=sys.stderr)
        return 2

    # Imported here so a missing snapshot/dotenv error happens before we
    # touch the agentic_eval package.
    from evaluation.agentic_eval import inspect_db

    with inspect_db.open_snapshot(args.snapshot):
        pendings = _fetch_pendings(args.painpoint)

    if not pendings:
        print(f"No pendings linked to painpoint #{args.painpoint} in {args.snapshot}",
              file=sys.stderr)
        return 3

    print()
    print(f"Snapshot      : {args.snapshot}")
    print(f"Target        : painpoint #{args.painpoint}  ({len(pendings)} linked pendings)")
    for p in pendings:
        print(f"  #{p['id']:>3d}  sev={p['severity']!s:>3}  sub={p['subreddit'] or '?':<12}  "
              f"{_truncate(p['title'], 70)}")

    embedder = OpenAIEmbedder()
    log.info("embedding %d pendings", len(pendings))
    t0 = time.monotonic()
    vecs = embedder.embed_batch([p["embed_text"] for p in pendings])
    embed_seconds = time.monotonic() - t0
    print(f"\nEmbed time    : {embed_seconds:.2f}s ({len(vecs)} vectors)")

    thresholds = _frange(args.start, args.stop, args.step)
    labels_by_threshold: dict[float, list[int]] = {}
    summaries: list[dict] = []
    for thr in thresholds:
        labels = cluster_at(thr, vecs)
        labels_by_threshold[thr] = labels
        summaries.append(cluster_summary(labels))

    print_threshold_table(thresholds, summaries)
    print_per_pending_table(pendings, labels_by_threshold, thresholds)
    print(ascii_trajectory(thresholds, summaries, len(pendings)))

    # Find the smallest threshold that splits the cluster off from being one
    # mega-component, and the smallest that drives every pending to a
    # singleton.
    split_thr = next(
        (t for t, s in zip(thresholds, summaries) if s["n_components"] > 1),
        None,
    )
    all_singleton_thr = next(
        (t for t, s in zip(thresholds, summaries) if s["max_size"] == 1),
        None,
    )
    print(f"First threshold splitting the mega-cluster : "
          f"{split_thr if split_thr is not None else f'> {args.stop}'}")
    print(f"First threshold making every pending alone : "
          f"{all_singleton_thr if all_singleton_thr is not None else f'> {args.stop}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
