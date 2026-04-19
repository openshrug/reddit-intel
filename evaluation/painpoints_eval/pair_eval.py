"""Pair-cosine evaluation against a fixture YAML.

For each labelled pair in a fixture file:

* Embed both strings with :class:`db.embeddings.OpenAIEmbedder`
  (single batched API call).
* Compute cosine similarity.
* Compare to the live value of the named threshold constant
  (``MERGE_COSINE_THRESHOLD`` / ``PENDING_MERGE_THRESHOLD`` / …).
* Score precision / recall / F1 with **positive = "should merge"**:

  * TP = positive label, cosine >= threshold (we'd merge — correctly)
  * FN = positive label, cosine <  threshold (we'd miss a real merge)
  * FP = negative label, cosine >= threshold (we'd wrongly merge)
  * TN = negative label, cosine <  threshold (we'd correctly skip)

Output:

* Confusion matrix to stdout, with each FP / FN row printed verbatim
  (id, cite, the two strings, cosine).
* JSON dump to ``--out-json`` (default
  ``evaluation/painpoints_eval/runs/pair_eval_<fixture>_<ts>.json``).

The fixture's ``threshold_under_test`` is a constant *name*, not a
number, so retunes of ``MERGE_COSINE_THRESHOLD`` flow through without
touching the YAML.

Usage::

    python -m evaluation.painpoints_eval.pair_eval \\
        --fixture evaluation/painpoints_eval/fixtures/painpoint_merge_pairs.yaml

    python -m evaluation.painpoints_eval.pair_eval \\
        --fixture .../pending_dedup_pairs.yaml \\
        --threshold 0.70   # override the live value for one run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from db.embeddings import OpenAIEmbedder

from ._util import cosine_sim, load_pairs, resolve_threshold

log = logging.getLogger(__name__)

DEFAULT_RUN_DIR = Path(__file__).parent / "runs"


# ---------------------------------------------------------------------------
# Confusion matrix bookkeeping
# ---------------------------------------------------------------------------

class PairOutcome:
    __slots__ = ("pair", "cosine", "predicted_merge", "truth_merge", "outcome")

    def __init__(self, pair: dict, cosine: float, threshold: float):
        self.pair = pair
        self.cosine = cosine
        self.predicted_merge = cosine >= threshold
        self.truth_merge = pair["label"] == "positive"
        if self.truth_merge and self.predicted_merge:
            self.outcome = "TP"
        elif self.truth_merge and not self.predicted_merge:
            self.outcome = "FN"
        elif not self.truth_merge and self.predicted_merge:
            self.outcome = "FP"
        else:
            self.outcome = "TN"


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def score(outcomes: list[PairOutcome]) -> dict:
    tp = sum(1 for o in outcomes if o.outcome == "TP")
    fn = sum(1 for o in outcomes if o.outcome == "FN")
    fp = sum(1 for o in outcomes if o.outcome == "FP")
    tn = sum(1 for o in outcomes if o.outcome == "TN")
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "n_positive": tp + fn,
        "n_negative": fp + tn,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def print_report(
    fixture_path: Path,
    threshold_name: str,
    threshold_value: float,
    threshold_source: str,
    outcomes: list[PairOutcome],
    metrics: dict,
    embed_seconds: float,
) -> None:
    print()
    print(f"Fixture       : {fixture_path}")
    print(f"Threshold     : {threshold_name} = {threshold_value:.4f}  ({threshold_source})")
    print(f"Pairs         : {len(outcomes)}  ({metrics['n_positive']} positive / {metrics['n_negative']} negative)")
    print(f"Embed time    : {embed_seconds:.2f}s")
    print()
    print("Confusion matrix")
    print("                              predicted=merge  predicted=skip")
    print(f"  truth=merge (positive)         TP={metrics['tp']:<5}      FN={metrics['fn']:<5}")
    print(f"  truth=skip  (negative)         FP={metrics['fp']:<5}      TN={metrics['tn']:<5}")
    print()
    print(f"Precision = {metrics['precision']:.3f}    "
          f"Recall = {metrics['recall']:.3f}    "
          f"F1 = {metrics['f1']:.3f}    "
          f"Accuracy = {metrics['accuracy']:.3f}")

    fails = [o for o in outcomes if o.outcome in ("FP", "FN")]
    if fails:
        print()
        print(f"Failures ({len(fails)}):")
        for o in fails:
            p = o.pair
            print(f"  [{o.outcome}] {p['id']}  cosine={o.cosine:.4f}  threshold={threshold_value:.4f}")
            print(f"      a    : {_truncate(p['a'], 90)}")
            print(f"      b    : {_truncate(p['b'], 90)}")
            print(f"      cite : {p['cite']}")
            if p.get("notes"):
                print(f"      note : {_truncate(p['notes'], 100)}")
    else:
        print()
        print("All pairs classified correctly at the current threshold.")

    print()
    print("Cosine distribution by truth label")
    pos_cos = sorted(o.cosine for o in outcomes if o.truth_merge)
    neg_cos = sorted(o.cosine for o in outcomes if not o.truth_merge)
    if pos_cos:
        print(f"  positive (n={len(pos_cos)}): "
              f"min={pos_cos[0]:.3f}  median={pos_cos[len(pos_cos)//2]:.3f}  max={pos_cos[-1]:.3f}")
    if neg_cos:
        print(f"  negative (n={len(neg_cos)}): "
              f"min={neg_cos[0]:.3f}  median={neg_cos[len(neg_cos)//2]:.3f}  max={neg_cos[-1]:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    fixture_path: Path,
    threshold_override: Optional[float] = None,
    embedder: Optional[OpenAIEmbedder] = None,
) -> dict:
    """Run pair_eval against ``fixture_path``.

    Returns a dict with ``metrics`` + serialised per-pair outcomes,
    suitable for JSON dump or further analysis. The terminal report
    is *not* printed by this function — call :func:`print_report` for
    that.
    """
    data = load_pairs(fixture_path)
    threshold_name = data["threshold_under_test"]
    if threshold_override is not None:
        threshold_value = float(threshold_override)
        threshold_source = "override"
    else:
        threshold_value = resolve_threshold(threshold_name)
        threshold_source = f"db.embeddings.{threshold_name}"

    pairs = data["pairs"]
    embedder = embedder or OpenAIEmbedder()

    # Batch all unique strings to minimise round-trips. Pairs commonly
    # repeat the same anchor on multiple rows (e.g. pp #48
    # "Tool unreliable for production" appears in 4 negative pairs).
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

    outcomes: list[PairOutcome] = []
    for p in pairs:
        va = vecs[seen[p["a"]]]
        vb = vecs[seen[p["b"]]]
        outcomes.append(PairOutcome(p, cosine_sim(va, vb), threshold_value))

    metrics = score(outcomes)
    return {
        "fixture": str(fixture_path),
        "threshold_name": threshold_name,
        "threshold_value": threshold_value,
        "threshold_source": threshold_source,
        "metrics": metrics,
        "outcomes": [
            {
                "id": o.pair["id"],
                "label": o.pair["label"],
                "cosine": o.cosine,
                "outcome": o.outcome,
                "predicted_merge": o.predicted_merge,
                "cite": o.pair["cite"],
                "a": o.pair["a"],
                "b": o.pair["b"],
                "notes": o.pair.get("notes", ""),
            }
            for o in outcomes
        ],
        "embed_seconds": embed_seconds,
        "_outcome_objs": outcomes,  # popped before JSON dump
    }


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(
        description="Evaluate a pair-cosine fixture against a named threshold.",
    )
    parser.add_argument(
        "--fixture", required=True, type=Path,
        help="Path to a YAML fixture (see SEEDING.md).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the live value of the fixture's named threshold "
             "for one run (does not write back to db.embeddings).",
    )
    parser.add_argument(
        "--out-json", type=Path, default=None,
        help=f"Where to dump the per-pair JSON. Defaults to "
             f"{DEFAULT_RUN_DIR}/pair_eval_<fixture>_<ts>.json.",
    )
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

    result = evaluate(args.fixture, threshold_override=args.threshold)
    outcome_objs = result.pop("_outcome_objs")

    print_report(
        args.fixture,
        result["threshold_name"],
        result["threshold_value"],
        result["threshold_source"],
        outcome_objs,
        result["metrics"],
        result["embed_seconds"],
    )

    out_json = args.out_json
    if out_json is None:
        DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_json = DEFAULT_RUN_DIR / f"pair_eval_{args.fixture.stem}_{ts}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print()
    print(f"JSON dump  : {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
