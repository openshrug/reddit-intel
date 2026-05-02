"""A/B-compare two pipeline runs and emit a human-readable markdown diff.

Use this whenever you change the extraction prompt, the evidence filter,
the LLM model, or any threshold that affects ``pending_painpoints``.
The runs themselves are produced by ``run_pipeline.py``; this module
only reads their snapshots.

Usage::

    python -m evaluation.agentic_eval.compare_runs \\
        evaluation/agentic_eval/runs/selfhosted_20260502-093017 \\
        evaluation/agentic_eval/runs/selfhosted_20260502-101838

Writes ``compare_<runA_basename>_vs_<runB_basename>.md`` next to run B
(so each NEW run accumulates one comparison file per baseline you pin
it against). Use ``--stdout`` to print instead. Pass ``--snapshot`` to
target a specific stage when the runs have multiple subreddits (default
is the last ``NN_<label>`` directory in each run).

The cell-level diff joins on Reddit ``post.permalink`` +
``comment.permalink`` so synthetic row IDs renumbering between runs
doesn't break the join. See
``inspect_db.cross_run_pending_diff`` for the underlying primitive.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from . import inspect_db

log = logging.getLogger(__name__)

# Cells whose severity moved by >= this delta are surfaced first in the
# side-by-side renderer because severity recalibration is usually the
# clearest qualitative signal of a prompt change.
SEVERITY_DELTA_HIGHLIGHT = 2

# How much of each text snippet to render before truncating with "...".
QUOTE_TRUNCATE = 180
TITLE_TRUNCATE = 90
COMMENT_TRUNCATE = 140


# ============================================================
# Public CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="A/B-compare two agentic_eval runs.",
    )
    parser.add_argument("run_a", type=Path, help="Baseline run directory.")
    parser.add_argument("run_b", type=Path, help="New run directory.")
    parser.add_argument(
        "--snapshot", default=None,
        help="Snapshot label to compare on each side (e.g. '01_selfhosted'). "
             "Defaults to the last NN_<label> dir in each run.",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Write to stdout instead of a file.",
    )
    parser.add_argument(
        "--all-common", action="store_true",
        help="Render every common cell side-by-side, not just changed ones.",
    )
    parser.add_argument(
        "--limit", type=int, default=40,
        help="Max common-cell side-by-side entries to render (default 40). "
             "Top changes (largest severity delta first) win.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    md = render_comparison(
        run_a=args.run_a, run_b=args.run_b,
        snapshot_label=args.snapshot,
        all_common=args.all_common,
        limit=args.limit,
    )
    if args.stdout:
        print(md)
    else:
        out = args.run_b / f"compare_{args.run_a.name}_vs_{args.run_b.name}.md"
        out.write_text(md)
        log.info("wrote %s", out)


# ============================================================
# Public API
# ============================================================

def render_comparison(
    *, run_a, run_b, snapshot_label=None, all_common=False, limit=40,
):
    """Return a markdown string A/B-comparing two runs.

    Args:
        run_a, run_b: ``Path`` objects pointing at run directories under
            ``runs/`` (each containing ``NN_<label>/`` snapshot dirs).
        snapshot_label: which snapshot to diff on each side. ``None``
            picks the last ``NN_<label>`` dir per run, which is the
            common single-subreddit case (skip 00_clean and the sweep).
            Pass an explicit label like ``"01_selfhosted"`` if the runs
            have multiple subreddits or different ordering.
        all_common: if True, render every common-cell side-by-side
            instead of just changed ones.
        limit: max common-cell entries to render in the side-by-side
            (top sorted by severity delta).

    The driver is structured as: pick snapshots -> compute headline
    deltas from metrics.json -> get cell-level diff via
    ``cross_run_pending_diff`` -> stitch into markdown.
    """
    snap_a, snap_b = _pick_snapshot(run_a, snapshot_label), \
                     _pick_snapshot(run_b, snapshot_label)

    metrics_a = _load_metrics(snap_a)
    metrics_b = _load_metrics(snap_b)

    diff = inspect_db.cross_run_pending_diff(
        snap_a / "trends.db", snap_b / "trends.db",
    )

    parts = [
        _render_header(run_a, run_b, snap_a, snap_b),
        _render_headline_table(metrics_a, metrics_b),
        _render_cell_overview(diff),
        _render_side_by_side(diff["common"], all_common=all_common, limit=limit),
        _render_only_side(diff["only_a"], side_label="OLD", action="DROPPED"),
        _render_only_side(diff["only_b"], side_label="NEW", action="ADDED"),
    ]
    return "\n\n".join(p for p in parts if p)


# ============================================================
# Snapshot + metrics resolution
# ============================================================

def _pick_snapshot(run_dir, snapshot_label):
    """Return the snapshot dir to compare. Defaults to the last
    ``NN_<label>`` under ``run_dir`` (skipping 00_clean), so the common
    single-subreddit case 'just works'."""
    if snapshot_label is not None:
        candidate = run_dir / snapshot_label
        if not candidate.is_dir():
            raise SystemExit(
                f"snapshot {snapshot_label!r} not found under {run_dir}; "
                f"available: {[p.name for p in sorted(run_dir.iterdir()) if p.is_dir()]}"
            )
        return candidate

    snapshots = sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and re.match(r"^\d{2}_", p.name) and p.name != "00_clean"
    )
    if not snapshots:
        raise SystemExit(f"no NN_<label> snapshot dirs under {run_dir}")
    # Skip the post-sweep snapshot — pre-sweep is the right A/B point
    # for extraction-side changes (sweep adds noise that masks prompt
    # effects and isn't always present).
    pre_sweep = [s for s in snapshots if "post_sweep" not in s.name]
    return pre_sweep[-1] if pre_sweep else snapshots[-1]


def _load_metrics(snapshot_dir):
    """Load metrics.json for one snapshot dir."""
    metrics_path = snapshot_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"missing metrics.json in {snapshot_dir}")
    return json.loads(metrics_path.read_text())


# ============================================================
# Markdown renderers
# ============================================================

def _render_header(run_a, run_b, snap_a, snap_b):
    return (
        "# Run comparison\n\n"
        f"- **A (baseline)**: `{run_a.name}` snapshot `{snap_a.name}`\n"
        f"- **B (new)**: `{run_b.name}` snapshot `{snap_b.name}`\n"
        f"- Cell join key: `(post.permalink, comment.permalink or '__POST_BODY__')`"
    )


def _render_headline_table(a, b):
    """Side-by-side metrics.json headline table with deltas."""
    # Carefully name the rows so a future schema change to metrics.json
    # produces "(missing)" cells rather than crashing the comparison.
    rows = [
        ("Posts",                 _g(a, "totals", "posts"),
                                  _g(b, "totals", "posts")),
        ("Comments",              _g(a, "totals", "comments"),
                                  _g(b, "totals", "comments")),
        ("Pendings persisted",    _g(a, "totals", "pending_painpoints"),
                                  _g(b, "totals", "pending_painpoints")),
        ("Painpoints",            _g(a, "totals", "painpoints"),
                                  _g(b, "totals", "painpoints")),
        ("Multi-source painpoints",
                                  _g(a, "painpoint_merge", "painpoints_with_multi_sources"),
                                  _g(b, "painpoint_merge", "painpoints_with_multi_sources")),
        ("Max signal_count",      _g(a, "painpoint_merge", "max_signal_count"),
                                  _g(b, "painpoint_merge", "max_signal_count")),
        ("In seed category",      _g(a, "categorization", "in_seed_category"),
                                  _g(b, "categorization", "in_seed_category")),
        ("In Uncategorized",      _g(a, "categorization", "in_uncategorized"),
                                  _g(b, "categorization", "in_uncategorized")),
        ("pct_uncategorized",     _g(a, "categorization", "pct_uncategorized"),
                                  _g(b, "categorization", "pct_uncategorized")),
        ("Pending dedup rate",    _g(a, "pending_dedup", "rate"),
                                  _g(b, "pending_dedup", "rate")),
    ]
    out = ["## Headline metrics", "",
           "| Metric | A (old) | B (new) | Δ |", "|---|---:|---:|---:|"]
    for label, av, bv in rows:
        out.append(f"| {label} | {_fmt_num(av)} | {_fmt_num(bv)} | {_fmt_delta(av, bv)} |")
    return "\n".join(out)


def _render_cell_overview(diff):
    return (
        "## Cell-level diff\n\n"
        f"- Common (same Reddit cell in both runs): **{len(diff['common'])}**\n"
        f"- Only in A (dropped under B): **{len(diff['only_a'])}**\n"
        f"- Only in B (newly emitted): **{len(diff['only_b'])}**"
    )


def _render_side_by_side(common_pairs, *, all_common, limit):
    """Render OLD/NEW side-by-side for shared cells.

    Default mode shows only cells where something visibly changed
    (quote, title, severity, or category) since identical cells aren't
    informative. ``all_common`` overrides that filter.
    """
    if not common_pairs:
        return None

    pairs = common_pairs if all_common else [
        (a, b) for a, b in common_pairs if _changed(a, b)
    ]
    pairs.sort(key=lambda pair: -abs(_severity_delta(*pair)))
    rendered = pairs[:limit]
    if not rendered:
        return ("## Side-by-side (changed common cells)\n\n"
                "_No common cells changed quote/title/severity/category._")

    out = [
        "## Side-by-side (changed common cells)",
        "",
        f"Showing {len(rendered)} of {len(pairs)} changed pairs "
        f"(of {len(common_pairs)} total common cells), sorted by "
        f"|Δseverity| descending.",
        "",
    ]
    for old, new in rendered:
        out.extend(_render_pair(old, new))
        out.append("")
    return "\n".join(out)


def _render_pair(old, new):
    """One side-by-side block for a (old, new) pending pair."""
    src_lines = [
        f"### post #{old['post_id']} — {_truncate(old['post_title'], 80)}",
        f"- subreddit: r/{old['subreddit']}",
    ]
    if old.get("comment_body"):
        src_lines.append(
            f"- comment: {_truncate(old['comment_body'], COMMENT_TRUNCATE)}"
        )
    table = [
        "",
        "| | OLD (A) | NEW (B) |",
        "| --- | --- | --- |",
        f"| sev | {old['severity']} | {new['severity']} |",
        f"| cat | {_fmt_cat(old)} | {_fmt_cat(new)} |",
        f"| title | {_truncate(old['title'], TITLE_TRUNCATE)} "
        f"| {_truncate(new['title'], TITLE_TRUNCATE)} |",
        f"| quote | {_truncate(old['quoted_text'], QUOTE_TRUNCATE)} "
        f"| {_truncate(new['quoted_text'], QUOTE_TRUNCATE)} |",
    ]
    return src_lines + table


def _render_only_side(rows, *, side_label, action):
    """Render the ``only_a`` or ``only_b`` list, sorted by severity desc."""
    if not rows:
        return None
    out = [
        f"## {action} pendings (only in {side_label})",
        "",
        f"_{len(rows)} pendings, sorted by severity desc._",
        "",
        "| sev | cat | title | quote | post |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for r in rows:
        out.append(
            f"| {r['severity']} | {_fmt_cat(r)} "
            f"| {_truncate(r['title'], TITLE_TRUNCATE)} "
            f"| {_truncate(r['quoted_text'], QUOTE_TRUNCATE)} "
            f"| {_truncate(r['post_title'], 60)} |"
        )
    return "\n".join(out)


# ============================================================
# Pure helpers
# ============================================================

def _g(d, *path):
    """Safe nested-dict lookup. Returns None on any missing key.
    Used so a future metrics.json schema change shows '(missing)' in
    the comparison rather than crashing it."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _fmt_num(v):
    if v is None:
        return "(missing)"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def _fmt_delta(a, b):
    if a is None or b is None or not isinstance(a, (int, float)) \
            or not isinstance(b, (int, float)):
        return ""
    delta = b - a
    if isinstance(a, float) or isinstance(b, float):
        return f"{delta:+.4f}"
    pct = (100 * delta / a) if a else 0
    return f"{delta:+d} ({pct:+.1f}%)" if a else f"{delta:+d}"


def _fmt_cat(row):
    return row.get("category_name") or "(none)"


def _truncate(text, limit):
    if not text:
        return ""
    flat = " ".join(str(text).split())
    flat = flat.replace("|", "\\|")  # don't break markdown table cells
    if len(flat) <= limit:
        return flat
    return flat[:limit - 1] + "…"


def _changed(a, b):
    return (
        (a["quoted_text"] or "") != (b["quoted_text"] or "")
        or a["title"] != b["title"]
        or a["severity"] != b["severity"]
        or (a.get("category_name") or "") != (b.get("category_name") or "")
    )


def _severity_delta(a, b):
    return (b["severity"] or 0) - (a["severity"] or 0)


if __name__ == "__main__":
    main()
