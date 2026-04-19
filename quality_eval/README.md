# Pipeline quality eval

Drive the full pipeline (`scrape -> extract -> promote` per subreddit,
then one `run_sweep`) on a clean DB, snapshot `trends.db` after every
stage, and dump per-stage markdown + metrics so a separate evaluator
agent can produce a grounded quality report.

The runtime modules (`subreddit_pipeline.analyze`,
`category_worker.run_sweep`, `db/*`) are **not modified** -- this folder
only adds a driver, a snapshot helper, and read-only inspection
utilities.

---

## Layout

```
quality_eval/
    run_pipeline.py     # entry point: clean-DB run + snapshots
    snapshot.py         # checkpoint + copy trends.db, render dump + metrics
    inspect_db.py       # open_snapshot() ctx manager + 4 gap-filling helpers
    dump.py             # per-snapshot markdown generator
    metrics.py          # per-snapshot JSON metrics + cross-snapshot deltas
    instructions/       # one file per concern, evaluator agent reads these
        00_protocol.md  # role, inputs, snapshot inspection, threshold rule, hard rules
        10_extraction.md            # dimension 1
        20_pending_dedup.md         # dimension 2
        30_pending_merge.md         # dimension 3
        40_category.md              # dimension 4
        90_synthesis.md             # summary table + cross-dim recommendations
    README.md           # this file
    runs/               # output dir (gitignored), one subdir per invocation
        <sub1>_<sub2>_..._<YYYYMMDD-HHMMSS>/
            00_clean/
            01_<sub1>/
            02_<sub2>/
            03_<sub3>/
            04_post_sweep/
                trends.db
                dump.md
                metrics.json
            report.md   # written by the evaluator agent (see below)
```

Each invocation of `run_pipeline` allocates a fresh run directory whose
name is the slugified subreddit list joined with the start timestamp
(local time, second precision). Re-running the pipeline never
overwrites a previous run; snapshots and the evaluator's report stay
co-located.

---

## Run

From the `reddit-intel/` repo root, with the project's venv activated
and `OPENAI_API_KEY` set in `.env`:

```bash
.venv/bin/python -m quality_eval.run_pipeline
```

The default subreddit list is `openclaw, ClaudeAI, SideProject`. Override with:

```bash
.venv/bin/python -m quality_eval.run_pipeline --subreddits foo bar baz
```

Other flags:

| Flag | Default | Effect |
| --- | --- | --- |
| `--keep-existing-db` | off | Skip the `trends.db` wipe (resume against the live DB). A fresh run dir is still created so prior runs aren't touched. |
| `--skip-sweep` | off | Stop after the last subreddit; don't run `category_worker.run_sweep` |
| `--min-score N` | 0 | Forwarded to `scrape_subreddit_full` |
| `--debug` | off | DEBUG-level logging |

The chosen run directory is logged on the first INFO line (`Run dir: ...`)
and again on completion (`Done. Snapshots in ...`).

**Cost / time**: a real run hits the live Reddit JSON endpoints, the
OpenAI embeddings API, and `gpt-5-nano` / `gpt-4.1-mini` for extraction
and naming. Expect ~3-10 min wall-clock and a few cents of OpenAI spend
per subreddit (matches the figures in
`tests/live/test_e2e_real_subreddits.py`). The clean wipe deletes the
live `trends.db` -- back it up first if you need it.

---

## Snapshot semantics

Each snapshot directory is taken **after** the named stage finishes:

| Dir | Captured state |
| --- | --- |
| `00_clean` | `db.init_db()` only -- seed taxonomy + Uncategorized sentinel, no posts. |
| `01_<sub1>` | After `analyze(sub1)` -- scrape + extract + promote for the first subreddit. |
| `02_<sub2>` | After `analyze(sub2)` -- accumulates over `01_*` (same DB, more rows). |
| `03_<sub3>` | After `analyze(sub3)` -- accumulates over `02_*`. |
| `04_post_sweep` | After one `run_sweep()` -- only snapshot with rows in `category_events`. |

Snapshots 1-3 reflect **promote-time** state (the category worker has
not run yet). They're useful for evaluating extraction + pending dedup
+ promote-time merge. Snapshot 4 is the only one with sweep-time
mutations (Uncategorized clustering, splits, deletes, merges, reroutes,
painpoint_merge) and the final taxonomy.

---

## What's in each snapshot dir

- `trends.db` -- standalone SQLite file, taken after
  `PRAGMA wal_checkpoint(TRUNCATE)`. Open with the project's normal
  helpers via the `inspect_db.open_snapshot(path)` context manager.
- `dump.md` -- six sections aligned with the four quality dimensions
  in `instructions/`: header + totals, new pendings since previous
  snapshot, pending dedup groups, painpoints with multiple sources,
  category tree, audit log.
- `metrics.json` -- numeric aggregates + cross-snapshot deltas
  (`pending_dedup`, `painpoint_merge`, `categorization`, per-subreddit
  totals, `category_events` counts).

---

## Inspecting a snapshot from a Python REPL

```python
from pathlib import Path
from quality_eval import inspect_db

run = Path("quality_eval/runs/openclaw_claudeai_sideproject_20260419-101530")
snap = run / "04_post_sweep" / "trends.db"
with inspect_db.open_snapshot(snap):
    from db.queries import get_top_painpoints, get_painpoint_evidence
    for pp in get_top_painpoints(limit=5):
        print(pp["id"], pp["signal_count"], pp["title"])
        for ev in get_painpoint_evidence(pp["id"]):
            print("   ", ev["pending_id"], ev["pending_title"])
```

`inspect_db.open_snapshot` swaps `db.DB_PATH` for the duration of the
block, so every existing read helper in `db/queries.py`,
`db/categories.py`, and `db/posts.py` works against the snapshot file.
The four new helpers (`list_pending_painpoints_for_subreddit`,
`list_pending_dedup_groups`, `render_category_tree`,
`cross_snapshot_diff`) cover the gaps the existing API doesn't.

---

## Handing the run off to an evaluator agent

Once a run directory under `quality_eval/runs/` is populated:

1. Open a fresh agent session (so it doesn't carry pipeline-debugging
   context).
2. Point it at `quality_eval/instructions/00_protocol.md` and the
   specific run directory (e.g.
   `quality_eval/runs/openclaw_claudeai_sideproject_20260419-101530/`).
3. Ask it to follow the protocol and emit `report.md` *inside* that
   run directory, alongside the per-stage snapshots.

`00_protocol.md` is the entry point. It links to one file per
dimension (`10_extraction.md`, `20_pending_dedup.md`,
`30_pending_merge.md`, `40_category.md`) and a synthesis file
(`90_synthesis.md`) for the summary table + cross-dimensional
recommendations. The split makes each per-dimension file
self-contained so individual dimensions can be re-evaluated in
isolation if needed.
