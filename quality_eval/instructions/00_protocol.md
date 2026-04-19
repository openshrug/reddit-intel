# Pipeline quality evaluation -- protocol

You are the parent **evaluator agent** for one pipeline run. The user
will point you at one specific run directory under
`quality_eval/runs/`, e.g.
`quality_eval/runs/openclaw_claudeai_sideproject_20260419-101530/`.
All paths in this folder are relative to that run directory; treat it
as your working root.

Your final deliverable is a single `report.md` written **inside that
run directory**, alongside the per-stage snapshots, that judges the
four pipeline-quality dimensions and offers cross-dimensional
recommendations. The report must be grounded in concrete examples
copy-pasted out of the snapshot data -- never paraphrased, never
fabricated.

You do **not** need to call the LLM extractor or the OpenAI API. All
the material you need is already on disk: each snapshot directory
contains a `trends.db` SQLite file plus pre-rendered `dump.md` and
`metrics.json`. Use `dump.md` for ready-to-cite examples and
`metrics.json` for population-level numbers; drop into the SQLite file
via the helpers in `quality_eval.inspect_db` when you need anything
that isn't already rendered.

---

## 1. Document map

The instructions are split into one file per concern. Read them in
this order:

| File | Purpose |
| --- | --- |
| `00_protocol.md` (this file) | Inputs, snapshot semantics, snapshot-inspection how-to, threshold-lookup rule, hard rules for citations and the report. |
| `10_extraction.md` | Dimension 1 -- per-subreddit extraction quality. |
| `20_pending_dedup.md` | Dimension 2 -- extract-time pending dedup. |
| `30_pending_merge.md` | Dimension 3 -- pending -> painpoint merging (promote-time + sweep). |
| `40_category.md` | Dimension 4 -- category assignment (promote-time placement + sweep cleanup). |
| `90_synthesis.md` | After the per-dimension work is done: assemble the summary table, write cross-dimensional recommendations, and finalize `report.md`. |

Each per-dimension file is self-contained: it lists the snapshots /
metrics / helpers to use, the source-of-truth functions to read in the
codebase before judging, the criteria, and the citation rules. You can
hand any one of those files to a sub-agent and it will know what to
do.

---

## 2. Execution model: fan out to one sub-agent per dimension

The four dimensions are independent (no shared judgment state), so
**dispatch one sub-agent per dimension and run them in parallel if
your agent harness supports parallel sub-agent dispatch**. This keeps
contexts clean (a dim-4 sub-agent isn't biased by 25 noisy pendings
it just read for dim 1) and cuts wall-clock roughly 4×. If your
harness doesn't support sub-agents at all, fall back to single-agent
sequential evaluation: read the per-dim files yourself in dimension
order. Either mode produces the same final report.

### Dispatch contract

For each dimension `N` in `{1, 2, 3, 4}`, spawn a sub-agent and give
it:

- The **run directory** absolute path (e.g.
  `/abs/path/.../quality_eval/runs/openclaw_..._20260419-101530/`).
- The corresponding **per-dimension file** to follow:
  - dim 1 -> `quality_eval/instructions/10_extraction.md`
  - dim 2 -> `quality_eval/instructions/20_pending_dedup.md`
  - dim 3 -> `quality_eval/instructions/30_pending_merge.md`
  - dim 4 -> `quality_eval/instructions/40_category.md`
- A pointer to **this file** (`00_protocol.md`) for the snapshot-
  inspection helper API, the threshold-lookup rule, and the hard
  rules.

The sub-agent must be able to read code, execute Python helpers
against the snapshot DB, and grep for live constant values -- so a
read-only or pure-text sub-agent profile is too restrictive. Pick
whichever sub-agent type your harness exposes that has those
capabilities.

**Dispatch all four sub-agents in parallel using whatever parallel
sub-agent primitive your harness exposes** (typically: emitting
multiple sub-agent tool calls in a single assistant message). Do not
await one sub-agent before dispatching the next.

### Sub-agent return contract

Every sub-agent must return a single message in this exact shape:

```
META: score=<int 1-5>; verdict=<pass|mixed|fail>; headline=<one sentence>

## Dimension <N> -- <name>

<full markdown body of the dimension section, ready to paste into report.md>
```

- The first line is the `META:` header. Parse it for the summary
  table; never let it leak into the report body.
- Everything after the blank line is the dimension's report section.
  It starts with the `## Dimension N -- ...` heading and contains the
  Numbers, Examples, and Failure modes subsections defined in
  Section 7 below.
- The headline must be a single sentence with no internal `;`
  characters (so the META line stays trivially parseable).

If a sub-agent returns malformed output (missing META, wrong header
line shape) or reports an error, **do not fall back silently**:
re-dispatch with a corrective note, or evaluate that dimension inline
and flag the redo in the final report's Recommendations section.

### After the sub-agents return

1. Parse each `META:` line into one row of the summary table.
2. Concatenate the four section bodies in dimension order (1 -> 4).
3. Read `90_synthesis.md` and run the cross-dimensional protocol
   (compound failure scan + recommendations).
4. Assemble the final `report.md` per the template in
   `90_synthesis.md` and write it to the run directory.

---

## 3. Inputs

```
quality_eval/runs/<sub1>_<sub2>_..._<YYYYMMDD-HHMMSS>/   <- run dir
    00_clean/         trends.db, dump.md, metrics.json   (clean DB)
    01_<sub1>/        ... after analyze(sub1)
    02_<sub2>/        ... after analyze(sub2)
    03_<sub3>/        ... after analyze(sub3)
    04_post_sweep/    ... after one run_sweep()
    report.md         <- you write this when done
```

Snapshots are taken **after** each pipeline stage finishes. Snapshots
1-3 reflect promote-time state only (the category worker has **not**
run yet). Snapshot 4 is post-sweep and is the only one with rows in
`category_events`.

---

## 4. Inspecting a snapshot

From the `reddit-intel/` repo root:

```python
from pathlib import Path
from quality_eval import inspect_db

run = Path("quality_eval/runs/openclaw_claudeai_sideproject_20260419-101530")
snap = run / "03_sideproject" / "trends.db"

with inspect_db.open_snapshot(snap):
    # Existing helpers all work transparently inside this block:
    from db.queries import (
        get_stats, get_top_painpoints, get_painpoint_evidence,
        get_painpoints_by_subreddit, get_subreddit_summary, run_sql,
    )
    from db.categories import get_root_categories, get_all_categories
    from db.posts import get_posts_by_ids, get_comments_for_post

    # Plus the gap-filling helpers in inspect_db itself:
    inspect_db.list_pending_painpoints_for_subreddit("openclaw")
    inspect_db.list_pending_dedup_groups(min_extra_sources=1)
    inspect_db.render_category_tree()
    inspect_db.cross_snapshot_diff(prev_db_path, snap)
```

`run_sql("SELECT ...")` is the escape hatch for arbitrary read-only
queries when no helper fits.

---

## 5. Pipeline thresholds (look them up; don't memorize)

Each per-dimension file references one or more of these constants:

| Constant | Where applied |
| --- | --- |
| `PENDING_MERGE_THRESHOLD` | Pending dedup at extract-time (`db/painpoints.py: save_pending_painpoints_batch`) |
| `MERGE_COSINE_THRESHOLD` | Pending -> painpoint link at promote-time (`db/painpoints.py: promote_pending`) |
| `CATEGORY_COSINE_THRESHOLD` | Below this -> Uncategorized (`db/embeddings.py: find_best_category`) |
| `MIN_SUB_CLUSTER_SIZE` | Min Uncategorized cluster to mint a new category at sweep |

Don't hard-code the numeric values from memory -- they get tuned over
time. Look up the live value of each constant from the codebase before
quoting it in the report (e.g. `rg '^PENDING_MERGE_THRESHOLD' --type py`)
and cite that exact number so your verdict reflects the threshold
actually in force for this run.

---

## 6. Source-of-truth code (read before judging)

Doc prose ages faster than code. Before writing a verdict for any
dimension, **read the actual functions that implement that dimension
end-to-end**. Each per-dimension file lists its targets in a
"Source-of-truth code" subsection. Skim the implementations, then
return to the snapshot evidence with the live algorithm in mind. Do
not paraphrase the code into the report -- use it to ground your
verdict and your failure-mode list.

---

## 7. Per-dimension report shape

Each dimension's section in `report.md` must contain:

1. **Verdict**: `pass` / `mixed` / `fail` plus a 1-5 score.
2. **Population numbers**: pull the matching block out of
   `metrics.json` (cite the snapshot label).
3. **5+ concrete examples** copy-pasted from `dump.md` (or fetched via
   the helpers above). Each example must include the painpoint /
   pending id, the source quote, and a one-sentence reviewer note
   explaining what's right or wrong. Prefer a mix of good and bad
   examples.
4. **Failure modes**: every recurring failure pattern you see (e.g.
   "extractor frequently surfaces pricing-only complaints", "dedup
   collapses superficially-similar but distinct pains").

The exact citation format and the criteria to apply are in each
per-dimension file.

---

## 8. Hard rules

- Every example must carry a real id from the snapshot. **No
  fabrication.**
- Every quoted string must be copy-pasted from `dump.md` or fetched
  via the helpers; **never paraphrase a source quote**.
- Cite snapshot labels (`02_claudeai`, `04_post_sweep`, etc.) so the
  reviewer can re-open the source row.
- If a dimension can't be evaluated (e.g. no dedup groups fired in
  this run), say so explicitly and explain why.
- Cite numeric threshold values you pulled from the source, with the
  symbol name (e.g. ``PENDING_MERGE_THRESHOLD = 0.65``), not naked
  numbers.

The report layout is in `90_synthesis.md`.
