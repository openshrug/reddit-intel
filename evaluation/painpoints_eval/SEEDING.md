# Gold-pair seeding protocol

How to turn an `evaluation/agentic_eval/` `report.md` into the YAML
fixtures consumed by `pair_eval.py`, `threshold_sweep.py`, and
`mega_merge_stress.py`.

This protocol is **owned by `painpoints_eval/`** (the consumer) — not
by `agentic_eval/`. Agentic_eval emits a generic qualitative report;
this file is how the painpoint-merge harness lifts the cited examples
into a quantitative fixture. A future sibling like `category_eval/`
will have its own `SEEDING.md` for its own (different) fixture shape.

---

## 1. What we are producing

Two YAML files under `evaluation/painpoints_eval/fixtures/`:

| File | Fixture target | Threshold under test |
| --- | --- | --- |
| `painpoint_merge_pairs.yaml` | Dimension 3 — pending → painpoint merge | `MERGE_COSINE_THRESHOLD` (live = 0.60) |
| `pending_dedup_pairs.yaml`   | Dimension 2 — pending dedup at extract time | `PENDING_MERGE_THRESHOLD` (live = 0.65) |

Both share the same schema:

```yaml
threshold_under_test: MERGE_COSINE_THRESHOLD   # name only — pair_eval looks the live value up at runtime
pairs:
  - id: pp45-positive-1                        # short, stable, human-meaningful slug
    a: "Local-first AI architecture (privacy-focused)"
    b: "edge models"
    label: positive                            # positive | negative
    cite: runs/openclaw_claudeai_sideproject_20260419-101530/report.md#dim-3-pp-45
    notes: "pp #45 cluster — 4 pendings articulating run-AI-locally desire across 3 subreddits."
```

Field rules:

* `id` — kebab-case, prefixed with the painpoint id the pair is about
  (`pp45-`, `pp48-`). Must be unique across the whole file. Used as
  the row id in `pair_eval`'s confusion matrix and in
  `threshold_sweep`'s CSV.
* `a` / `b` — the *exact* `title` (and optionally `\n\n` + `description`)
  strings you fetched from the snapshot DB. **Verbatim**, no
  paraphrasing. They are what the live pipeline sees on the
  `text-embedding-3-small` input.
* `label` — `positive` if the two pendings *should* merge (cosine
  expected ≥ threshold), `negative` if they should *not* (cosine
  expected < threshold). No `borderline` value: borderline cases are
  exactly what the threshold sweep is for, so leave them out — adding
  them as either label biases the curve.
* `cite` — relative path from the repo root to the section of
  `report.md` the pair was lifted from, plus a `#dim-N-pp-<id>` anchor.
  This is the *single source of truth* for "why did we add this pair?".
* `notes` — one sentence, optional. For your future self.

---

## 2. Where the gold labels come from

### Positive pairs (should merge)

Read `report.md` Dim 3 examples that are explicitly tagged **good**
or that the report explains as a clean merge. Each linked pending in
that painpoint's evidence list pairs with every other linked pending
to give an `n*(n-1)/2` set of positive pairs.

### Negative pairs (should NOT merge)

Read Dim 3 examples tagged **mixed**, **bad**, or any of the failure
modes called out in the dimension's "Failure modes" subsection
(`mega-merge`, `polarity flip`, `context collision`, `co-occurrence
drift`). The report identifies the specific component pendings that
should not have ended up in the same painpoint — pair *those*
components against each other.

### Pending dedup pairs

Same shape, but read from Dim 2:

* Positive: groups the report flags as **good** (e.g. pp #28 8-extras
  group on persistent local memory). Pair the primary with each
  extra.
* Negative: the **same-post double-extraction** examples are *not*
  good negatives — they're a Dim 1 artifact, not an embedding
  failure. Use the Dim 2 borderline examples explicitly called out
  (e.g. cost-allocation vs. cost-shock) as negatives.

---

## 3. How to fetch the actual title strings

You need the *verbatim* `title` (and ideally `description`) text of
each cited pending so the pair file embeds the same input the live
pipeline would. The agentic_eval snapshot machinery makes this a
two-line lookup.

```python
from pathlib import Path
from evaluation.agentic_eval import inspect_db

run = Path("evaluation/agentic_eval/runs/openclaw_claudeai_sideproject_20260419-101530")
snap = run / "04_post_sweep" / "trends.db"

with inspect_db.open_snapshot(snap):
    from db.queries import run_sql
    rows = run_sql("""
        SELECT id, title, description
        FROM pending_painpoints
        WHERE id IN (52, 55, 60, 62, 264)   -- the pp #48 mega-merge components
        ORDER BY id
    """)
    for r in rows:
        print(r["id"], r["title"])
```

Use `04_post_sweep/trends.db` for any pair you cite from a Dim 3
section (post-sweep is the only snapshot with the merged painpoints
present). Use the per-stage snapshot (`01_<sub>` / `02_<sub>` / etc.)
when the cited row only exists pre-sweep.

For painpoint *titles* (when you want to seed against the painpoint's
own text, not its source pendings), query `painpoints` instead:

```python
run_sql("SELECT id, title, description FROM painpoints WHERE id IN (45, 48, 122)")
```

> **Why fetch from the snapshot, not paraphrase from `report.md`?**
> The report quotes `quoted_text` and post titles for human reading;
> the embedder sees `pending_painpoints.title` (and sometimes
> `description`). Mismatching the two would make the harness measure
> a different distribution from production.

---

## 4. The `cite:` convention

Every pair carries `cite:` so a reader can answer "why is this pair
in the fixture?" in one click. Format:

```
cite: runs/<run_id>/report.md#dim-<N>-pp-<painpoint_id>
```

* `<run_id>` is the sub-dir name (e.g.
  `openclaw_claudeai_sideproject_20260419-101530`).
* `<N>` is the dimension number (2 for dedup, 3 for merge).
* `<painpoint_id>` is the painpoint the section is about (or the
  primary pending id for Dim 2).

The anchors don't actually have to be present headings in the
report — they're a stable shorthand. `pair_eval` doesn't follow
them; humans do, when they want the surrounding evidence.

If a single pair captures a *cross-painpoint* issue (e.g. a Dim 2
borderline that the report discusses without a painpoint id), use
`...#dim-2-pending-<pending_id>` instead.

---

## 5. Dedup against existing pairs

Before adding a pair, grep both YAML files for both ids. If the same
`(a, b)` (in either order, by content match) already exists with the
same label, **skip** it. If it exists with the *opposite* label,
that's a contradiction in your evidence — do not add the new pair;
re-read both `report.md` sections and either remove the older pair or
drop the new one.

The harness intentionally treats every pair as independent (it does
not normalize duplicates), so a duplicate row will silently bias the
P/R/F1 numbers.

---

## 6. Worked example — pp #48 mega-merge

The current `report.md` Dim 3 entry on `pp #48` lists five linked
pendings (pp #52, #55, #60, #62, #264) the report explains are
*distinct* pains incorrectly merged into one painpoint. That gives us
`5 * 4 / 2 = 10` negative pairs for `painpoint_merge_pairs.yaml`.

Step 1: open the snapshot to lift the verbatim titles.

```python
from pathlib import Path
from evaluation.agentic_eval import inspect_db
from db.queries import run_sql  # noqa — imported under the with: block below

run = Path("evaluation/agentic_eval/runs/openclaw_claudeai_sideproject_20260419-101530")
with inspect_db.open_snapshot(run / "04_post_sweep" / "trends.db"):
    from db.queries import run_sql
    for r in run_sql("""
        SELECT id, title FROM pending_painpoints
        WHERE id IN (52, 55, 60, 62, 264) ORDER BY id
    """):
        print(f"pp #{r['id']}: {r['title']}")
```

Step 2: write the 10 negative pairs into
`fixtures/painpoint_merge_pairs.yaml`:

```yaml
- id: pp48-megamerge-52-vs-55
  a: "<title fetched for pp #52>"
  b: "<title fetched for pp #55>"
  label: negative
  cite: runs/openclaw_claudeai_sideproject_20260419-101530/report.md#dim-3-pp-48
  notes: "pp #48 mega-merge — pp #52 is general unreliability, pp #55 is framework misunderstanding."
- id: pp48-megamerge-52-vs-60
  ...
```

Step 3: balance with positives from a clean cluster — pp #45 (4
linked pendings = 6 positive pairs), pp #1 (4 = 6), pp #62 (4 = 6),
pp #122 *security-only* slice (3 = 3 — note: do **not** include
pp #143 there, it's the cost-vs-governance negative).

Total seed: ~10 negatives from pp #48 + ~3 negatives from
pp #81 / pp #56 / pp #122 + ~21 positives from pp #45 / #1 / #62 /
#122 ≈ 25 pairs, which is the order of magnitude `pair_eval` is
designed for.

For `pending_dedup_pairs.yaml` repeat the same flow against Dim 2
(8-extras pp #28 group for positives; pp #143 borderline cost-
allocation vs. cost-shock for the lone negative).

---

## 7. Maintenance

* Seed once per major report.md. Re-seed only when a *new* report
  surfaces a failure mode the existing fixtures don't cover.
* Never delete an old pair just because tuning the threshold "fixed"
  it — the regression value is the whole point. Mark the row instead
  with `notes: "fixed by raising MERGE_COSINE_THRESHOLD to 0.70 — keep as guard."`.
* When `category_eval/` lands, it gets its own `SEEDING.md` for its
  own (different) fixture shape; do not extend this one to cover
  category quality.
