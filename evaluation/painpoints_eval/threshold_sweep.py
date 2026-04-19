"""Sweep a merge threshold across a range and plot P/R/F1.

Embeds the fixture once (the embeddings don't depend on the
threshold), then re-scores P/R/F1 at each threshold in the swept
range. Cheap to extend the range or shrink the step — the OpenAI
spend is fixed at ``2 * n_pairs`` embeddings (and the cache makes that
even cheaper for fixtures with repeated anchors).

Default range targets ``MERGE_COSINE_THRESHOLD`` (live = 0.60) and
the report.md Recommendation #1 ("raise to ~0.70"):

    [0.55, 0.80] step 0.025 -> 11 thresholds

Outputs:

* CSV table to stdout (and to ``--out-csv`` if given) with columns
  ``threshold, tp, fn, fp, tn, precision, recall, f1, accuracy``.
* ASCII line plot of P / R / F1 curves vs. threshold so you can read
  the elbow off the terminal.

Usage::

    python -m evaluation.painpoints_eval.threshold_sweep \\
        --fixture evaluation/painpoints_eval/fixtures/painpoint_merge_pairs.yaml

    python -m evaluation.painpoints_eval.threshold_sweep \\
        --fixture .../pending_dedup_pairs.yaml \\
        --start 0.50 --stop 0.85 --step 0.025
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from db.embeddings import OpenAIEmbedder

from ._util import cosine_sim, load_pairs, resolve_threshold
from .pair_eval import PairOutcome, score

log = logging.getLogger(__name__)

DEFAULT_RUN_DIR = Path(__file__).parent / "runs"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _frange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range with deterministic rounding (avoids
    0.6000000001 noise that would clutter the printed CSV)."""
    out = []
    n = 0
    while True:
        v = round(start + n * step, 4)
        if v > stop + 1e-9:
            break
        out.append(v)
        n += 1
    return out


def sweep(
    fixture_path: Path,
    start: float,
    stop: float,
    step: float,
    embedder: Optional[OpenAIEmbedder] = None,
) -> dict:
    data = load_pairs(fixture_path)
    pairs = data["pairs"]
    embedder = embedder or OpenAIEmbedder()

    # Embed once (vectors are threshold-independent).
    seen: dict[str, int] = {}
    flat: list[str] = []
    for p in pairs:
        for s in (p["a"], p["b"]):
            if s not in seen:
                seen[s] = len(flat)
                flat.append(s)
    log.info("embedding %d unique strings (%d pairs)", len(flat), len(pairs))
    t0 = time.monotonic()
    vecs = embedder.embed_batch(flat)
    embed_seconds = time.monotonic() - t0

    cosines = [cosine_sim(vecs[seen[p["a"]]], vecs[seen[p["b"]]]) for p in pairs]

    thresholds = _frange(start, stop, step)
    rows = []
    for thr in thresholds:
        outcomes = [
            PairOutcome(p, cos, thr) for p, cos in zip(pairs, cosines)
        ]
        m = score(outcomes)
        rows.append({"threshold": thr, **m})

    return {
        "fixture": str(fixture_path),
        "threshold_name": data["threshold_under_test"],
        "live_threshold_value": resolve_threshold(data["threshold_under_test"]),
        "range": {"start": start, "stop": stop, "step": step},
        "embed_seconds": embed_seconds,
        "n_pairs": len(pairs),
        "n_unique_strings": len(flat),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_FIELDS = ("threshold", "tp", "fn", "fp", "tn",
              "precision", "recall", "f1", "accuracy")


def write_csv(rows: list[dict], file) -> None:
    w = csv.DictWriter(file, fieldnames=CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({
            "threshold": f"{r['threshold']:.4f}",
            "tp": r["tp"], "fn": r["fn"], "fp": r["fp"], "tn": r["tn"],
            "precision": f"{r['precision']:.4f}",
            "recall": f"{r['recall']:.4f}",
            "f1": f"{r['f1']:.4f}",
            "accuracy": f"{r['accuracy']:.4f}",
        })


# ---------------------------------------------------------------------------
# ASCII line plot
# ---------------------------------------------------------------------------

def ascii_plot(
    rows: list[dict],
    metrics: tuple[str, ...] = ("precision", "recall", "f1"),
    width: int = 60,
    height: int = 16,
    live_threshold: Optional[float] = None,
) -> str:
    """Render a tiny ASCII curve for each metric.

    All three metrics share the same y-axis [0.0, 1.0], so they overlay
    on one chart. The live threshold (if given) is drawn as a vertical
    `|` column.
    """
    if not rows:
        return "(no rows)"

    glyphs = {"precision": "P", "recall": "R", "f1": "F", "accuracy": "A"}
    n = len(rows)
    width = max(width, n)
    # Map x columns to row indices so n rows distribute evenly across
    # `width` cols; the live_threshold marker uses the same mapping.
    def _x_for_index(i: int) -> int:
        if n == 1:
            return 0
        return round(i * (width - 1) / (n - 1))

    grid = [[" "] * width for _ in range(height)]

    def _y_for_value(v: float) -> int:
        v = max(0.0, min(1.0, v))
        return height - 1 - round(v * (height - 1))

    # Live-threshold marker
    if live_threshold is not None:
        thresholds = [r["threshold"] for r in rows]
        if min(thresholds) <= live_threshold <= max(thresholds):
            # Find nearest sweep point to the live value
            nearest_i = min(range(n), key=lambda i: abs(rows[i]["threshold"] - live_threshold))
            x = _x_for_index(nearest_i)
            for y in range(height):
                if grid[y][x] == " ":
                    grid[y][x] = ":"

    for metric in metrics:
        for i, row in enumerate(rows):
            x = _x_for_index(i)
            y = _y_for_value(row[metric])
            cur = grid[y][x]
            grid[y][x] = "#" if cur not in (" ", ":") else glyphs[metric]

    # Add Y-axis labels
    out = StringIO()
    out.write(f"  metrics: {', '.join(f'{glyphs[m]}={m}' for m in metrics)}")
    if live_threshold is not None:
        out.write(f"   live threshold = {live_threshold:.3f} (':' column)")
    out.write("\n")
    out.write(f"  '#' = overlap, x-axis = threshold from {rows[0]['threshold']:.3f} to {rows[-1]['threshold']:.3f}\n\n")

    for y in range(height):
        # y maps back to a value: y=0 -> 1.0, y=height-1 -> 0.0
        v = 1.0 - (y / (height - 1)) if height > 1 else 0.0
        out.write(f"  {v:4.2f} | {''.join(grid[y])}\n")

    # X-axis ruler
    out.write("       +")
    out.write("-" * width)
    out.write("\n")
    out.write("        ")
    # First, mid, last threshold labels
    first = f"{rows[0]['threshold']:.3f}"
    mid = f"{rows[n // 2]['threshold']:.3f}"
    last = f"{rows[-1]['threshold']:.3f}"
    line = list(" " * width)
    line[: len(first)] = list(first)
    if width >= len(first) + len(mid):
        m_start = max(len(first) + 1, width // 2 - len(mid) // 2)
        line[m_start : m_start + len(mid)] = list(mid)
    if width >= len(last):
        line[width - len(last) :] = list(last)
    out.write("".join(line))
    out.write("\n")

    return out.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description="Sweep a merge threshold and plot P/R/F1 vs. threshold.",
    )
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--start", type=float, default=0.55)
    parser.add_argument("--stop", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Where to write the CSV. Defaults to "
                             "evaluation/painpoints_eval/runs/threshold_sweep_<fixture>_<ts>.csv.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the ASCII plot.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
        return 1
    if not args.fixture.exists():
        print(f"fixture not found: {args.fixture}", file=sys.stderr)
        return 2
    if args.start >= args.stop or args.step <= 0:
        print("invalid sweep range: need start < stop, step > 0", file=sys.stderr)
        return 2

    result = sweep(args.fixture, args.start, args.stop, args.step)

    print()
    print(f"Fixture       : {args.fixture}")
    print(f"Threshold name: {result['threshold_name']} (live = {result['live_threshold_value']:.4f})")
    print(f"Range         : start={args.start} stop={args.stop} step={args.step}  "
          f"({len(result['rows'])} thresholds)")
    print(f"Pairs         : {result['n_pairs']}  ({result['n_unique_strings']} unique strings)")
    print(f"Embed time    : {result['embed_seconds']:.2f}s (one batched call)")
    print()

    sio = StringIO()
    write_csv(result["rows"], sio)
    print(sio.getvalue())

    if not args.no_plot:
        print(ascii_plot(
            result["rows"],
            live_threshold=result["live_threshold_value"],
        ))

    out_csv = args.out_csv
    if out_csv is None:
        DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_csv = DEFAULT_RUN_DIR / f"threshold_sweep_{args.fixture.stem}_{ts}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        write_csv(result["rows"], f)
    print(f"CSV: {out_csv}")

    # Highlight the best F1 row in case the user wants a one-line answer.
    best = max(result["rows"], key=lambda r: r["f1"])
    print()
    print(f"Best F1: {best['f1']:.3f} at threshold = {best['threshold']:.3f}  "
          f"(P={best['precision']:.3f} R={best['recall']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
