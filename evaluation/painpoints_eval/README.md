# `painpoints_eval/` — pair-cosine evaluation

Quantitative harness for the two thresholds that govern painpoint
clustering in the engine:

| Threshold | Location | Live value | What it controls |
| --- | --- | ---: | --- |
| `MERGE_COSINE_THRESHOLD`   | `db/embeddings.py` | 0.60 | Pending → painpoint merge at promote time. |
| `PENDING_MERGE_THRESHOLD`  | `db/embeddings.py` | 0.65 | Pending dedup at extract time. |

The fixtures live under [`fixtures/`](fixtures/) and are seeded from
qualitative reports produced by `evaluation/agentic_eval/`. The
seeding protocol — how to lift a `report.md` into a gold-pair YAML —
is in [`SEEDING.md`](SEEDING.md). Do not re-invent it elsewhere; this
doc is the canonical input contract for the harness.

---

## Tools at a glance

| Tool | Purpose | Hits OpenAI? | DB? |
| --- | --- | --- | --- |
| `cosine_lab.py`         | Ad-hoc cosine REPL on arbitrary strings — understand model behaviour outside the fixtures. | yes (cached per session) | no |
| `pair_eval.py`          | Score a fixture at a named threshold; print confusion matrix + each FP/FN. | yes (one batched call) | no |
| `threshold_sweep.py`    | Embed once, replay the score across `[0.55, 0.80]` step `0.025`; emit CSV + ASCII curve. | yes (one batched call) | no |
| `mega_merge_stress.py`  | Open a real snapshot, fetch every pending linked to a target painpoint, greedy-cluster at varying thresholds. | yes (small batch) | yes (read-only via `agentic_eval.inspect_db.open_snapshot`) |

All four are command-line entry points (`python -m
evaluation.painpoints_eval.<tool>`); none mutate any state in
`trends.db` or in the live config.

---

## Run examples

All commands below assume a project venv with `OPENAI_API_KEY` set in
`.env`, run from the `reddit-intel/` repo root.

### 1. Pair eval at the live threshold

```bash
.venv/bin/python -m evaluation.painpoints_eval.pair_eval \
    --fixture evaluation/painpoints_eval/fixtures/painpoint_merge_pairs.yaml
```

Output: confusion matrix, P/R/F1, every FP/FN row with its `cite:`,
plus a JSON dump under `runs/`.

Override the threshold for one run:

```bash
.venv/bin/python -m evaluation.painpoints_eval.pair_eval \
    --fixture .../painpoint_merge_pairs.yaml --threshold 0.70
```

### 2. Threshold sweep + ASCII curve

```bash
.venv/bin/python -m evaluation.painpoints_eval.threshold_sweep \
    --fixture evaluation/painpoints_eval/fixtures/painpoint_merge_pairs.yaml
```

Default range is `[0.55, 0.80]` step `0.025`, which targets the
report.md Recommendation #1 ("raise `MERGE_COSINE_THRESHOLD` to
~0.70"). Tweak with `--start / --stop / --step`.

### 3. Mega-merge stress on `pp #48`

```bash
.venv/bin/python -m evaluation.painpoints_eval.mega_merge_stress \
    --snapshot evaluation/agentic_eval/snapshots/openclaw_claudeai_sideproject/04_post_sweep/trends.db
```

Defaults to painpoint #48 (the Dim 3 mega-merge poster child). Prints
a per-pending component-membership table at every threshold plus an
ASCII trajectory of cluster count vs. threshold — directly answers
"what threshold splits this cluster apart?" and tests Recommendation
#2 ("cap mega-clusters").

### 4. Cosine REPL

```bash
.venv/bin/python -m evaluation.painpoints_eval.cosine_lab --repl
```

Or one-shots:

```bash
.venv/bin/python -m evaluation.painpoints_eval.cosine_lab \
    --pair "API cost barrier" "High API costs for coding assistants"

.venv/bin/python -m evaluation.painpoints_eval.cosine_lab \
    --query "Need persistent local memory" \
    --against "Memory is killer feature" \
              "Add a feature flag" \
              "Cross-machine memory synchronization"
```

---

## Adding new gold pairs

Read [`SEEDING.md`](SEEDING.md) end-to-end. Short version:

1. Pick a fresh `agentic_eval/runs/<run_id>/report.md`.
2. Read Dim 2 / Dim 3 sections — extract the painpoints the report
   labels good (positive pairs) or flags as a failure mode (negative
   pairs).
3. Use `evaluation.agentic_eval.inspect_db.open_snapshot(...)` to
   pull the *verbatim* `pending_painpoints.title` strings out of
   `04_post_sweep/trends.db` for each cited id.
4. Append rows to `fixtures/painpoint_merge_pairs.yaml` (or
   `pending_dedup_pairs.yaml`) with a stable `id`, the verbatim
   strings, the `label`, and a `cite:` to the report section.
5. Run `pair_eval` to confirm the new rows surface as expected
   TPs/TNs (or as honest FPs/FNs that document a real gap).

---

## How this slots into the bigger picture

* This package is *quantitative regression coverage* for the engine's
  pairwise embedding behaviour. It complements but does not replace
  the qualitative `agentic_eval/` runs — embedding cosine alone
  ignores polarity, intent, and the engine's centroid blending.
* Fixture pairs are also useful as a permanent *bug record*: a
  threshold change that "fixes" pp #48 without breaking pp #45's
  positive pairs is provably better than today.
* When the production backend (closed) eventually pulls the engine's
  tuning via the parity contract in
  [`reddit-intel-closed/design_open_closed.md`](../../reddit-intel-closed/design_open_closed.md),
  the threshold values it adopts should be the ones the harness here
  has validated.

A future sibling `evaluation/category_eval/` will cover category-
sweep judge calibration, sibling distinctness, and the
Uncategorized-residue questions Dim 4 raised. It will have its own
`SEEDING.md` and its own (different) utilities — see the promotion
rule in [`evaluation/README.md`](../README.md).
