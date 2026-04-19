# `evaluation/` — quality + retrieval harnesses

Two complementary harnesses live under this root:

* **`agentic_eval/`** — *qualitative*. Drives a clean-DB pipeline run,
  snapshots `trends.db` after every stage, and hands the result to an
  LLM evaluator that produces a per-dimension `report.md`.
* **`painpoints_eval/`** — *quantitative*. Pair-cosine harness for the
  two thresholds that govern painpoint clustering
  (`MERGE_COSINE_THRESHOLD`, `PENDING_MERGE_THRESHOLD`). Consumes gold
  pairs lifted from `agentic_eval` reports; produces P/R/F1 numbers,
  a threshold sweep, and a mega-merge stress test.

A future sibling slot, **`category_eval/`**, is reserved for category-
sweep judge calibration / sibling distinctness / runtime-cat
lifecycle. Plan it separately when needed.

---

## Directory map

```
evaluation/
    README.md                                # this file
    __init__.py

    agentic_eval/                            # qualitative — snapshot-driven LLM judge
        README.md                            # how to run a snapshot, hand off to evaluator
        __init__.py
        run_pipeline.py                      # entry point: clean-DB run + per-stage snapshots
        snapshot.py                          # checkpoint + copy trends.db, render dump + metrics
        inspect_db.py                        # open_snapshot() ctx manager + 4 gap-filling helpers
        dump.py                              # per-snapshot markdown generator
        metrics.py                           # per-snapshot JSON metrics + cross-snapshot deltas
        instructions/                        # per-dimension prompts the evaluator agent reads
            00_protocol.md                   # entry point: snapshot inspection + hard rules
            10_extraction.md                 # dimension 1
            20_pending_dedup.md              # dimension 2
            30_pending_merge.md              # dimension 3
            40_category.md                   # dimension 4
            90_synthesis.md                  # summary table + cross-dim recommendations
        runs/                                # gitignored; one subdir per invocation
            <subreddits>_<YYYYMMDD-HHMMSS>/
                {00_clean,01_<sub>,...,04_post_sweep}/
                    trends.db
                    dump.md
                    metrics.json
                report.md                    # written by the evaluator agent

    painpoints_eval/                         # quantitative — pair-cosine harness
        README.md                            # run examples; points at SEEDING.md
        __init__.py
        SEEDING.md                           # protocol for turning agentic_eval reports into gold pairs
        _util.py                             # cosine_sim + YAML load/validate (private)
        cosine_lab.py                        # ad-hoc REPL on arbitrary strings (no DB)
        pair_eval.py                         # load pairs -> embed -> P/R/F1 + confusion matrix
        threshold_sweep.py                   # sweep MERGE threshold, ASCII curve + CSV
        mega_merge_stress.py                 # replay pp #48 cluster against varying thresholds
        fixtures/                            # gold pairs (per SEEDING.md)
            painpoint_merge_pairs.yaml       # ~25 pairs, every entry cited to a report.md
            pending_dedup_pairs.yaml         # ~10 pairs (Dim 2 regression coverage)
        runs/                                # gitignored; pair_eval / threshold_sweep dumps land here

    # category_eval/                         # FUTURE — not in this iteration.
    #     Different-shaped utilities (LLM-judge calibration, sibling
    #     distinctness, Uncategorized residue tracking, runtime-cat
    #     lifecycle). Will get its own SEEDING.md for its own fixture
    #     shape.

    # shared/                                # CREATED LAZILY — see promotion rule below.
    #     Sibling packages start with their own helpers (e.g. cosine_sim
    #     and YAML load live in painpoints_eval/_util.py today). Move a
    #     helper here only when a *second* sibling needs the same code.
```

---

## The qualitative → quantitative loop

```
agentic_eval                            painpoints_eval
─────────────────                       ───────────────────────────────
clean-DB pipeline run                   pair_eval.py    ─┐
   │                                                    │  P/R/F1 +
   ▼                                                    │  confusion
snapshot trends.db per stage            threshold_sweep.py  P/R curve
   │                                                    │
   ▼                                    mega_merge_stress.py
LLM evaluator                                              cluster count
   │   reads dump.md / metrics.json                        vs. threshold
   ▼
report.md                  ── (humans, per SEEDING.md) ──▶  fixtures/*.yaml
   (qualitative verdicts +                                  (gold pairs)
    cited examples)
```

The dotted edge — turning a `report.md` into gold pairs — is
deliberately a *human-in-the-loop* step. Pair labels need judgment;
a pure scrape would produce noisy fixtures. The protocol lives in
[`painpoints_eval/SEEDING.md`](painpoints_eval/SEEDING.md), owned by
the consumer.

---

## Promotion rule for shared helpers

Each sibling package starts with its own helpers. **Do not promote
something to `evaluation/shared/` until a *second* sibling actually
needs it** — premature `shared/` accretes things only one user wants
and locks the API early.

When that second use lands:

1. Create `evaluation/shared/__init__.py`.
2. Move the helper (with its tests) to `evaluation/shared/<name>.py`.
3. Update both consumers to import from `evaluation.shared`.
4. **Never copy-paste between siblings.** If two packages have similar
   but slightly-different helpers, fold them into one `shared/` API
   that covers both shapes — even if it means a small refactor on the
   first user.

Today the only candidate is `cosine_sim` (lives in
[`painpoints_eval/_util.py`](painpoints_eval/_util.py)). It will
graduate to `evaluation/shared/cosine.py` if and when `category_eval/`
also needs it.

---

## Running each harness

Quick links — full instructions live in each sub-package's README:

* `agentic_eval`:
  `python -m evaluation.agentic_eval.run_pipeline`
  (drives a fresh pipeline run on `openclaw, ClaudeAI, SideProject`
  by default — costs a few cents and ~3-10 min/subreddit; the
  resulting run dir is logged on the first INFO line). See
  [`agentic_eval/README.md`](agentic_eval/README.md).

* `painpoints_eval`:
  `python -m evaluation.painpoints_eval.pair_eval --fixture evaluation/painpoints_eval/fixtures/painpoint_merge_pairs.yaml`
  for a one-shot P/R/F1 readout, plus
  `python -m evaluation.painpoints_eval.threshold_sweep ...` and
  `python -m evaluation.painpoints_eval.mega_merge_stress ...` for
  the tuning views. See
  [`painpoints_eval/README.md`](painpoints_eval/README.md) and
  [`painpoints_eval/SEEDING.md`](painpoints_eval/SEEDING.md).
