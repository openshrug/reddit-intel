# Improvements & deferred work

Living doc for design ideas that are **not** worth shipping today but
should be on the radar once the cheap interventions stop carrying us.
Each entry should answer:

- **Why deferred?** What's the simpler fix we tried first, and what
  signal would tell us that fix is no longer enough?
- **Trigger to revisit.** Concrete metric / failure mode that flips
  this from "nice-to-have" to "now."
- **Sketch.** Enough of a design that whoever picks it up doesn't have
  to re-derive the trade-offs.

---

## OpenAI rate limiting — Phase 2: hybrid RPM + TPM token-velocity tracker

### Status

**Deferred.** Phase 1 (per-model concurrency split — see
`OPENAI_COMPLETION_SEMAPHORE` / `OPENAI_EMBEDDING_SEMAPHORE` in
`llm.py`) plus the rate-limit-aware retry layer in
`call_with_openai_retry` should clear the 429 issue we observed in the
`ClaudeAI` quality-eval run (~7 / 66 batches dropped). If a re-run
still drops batches against `gpt-5-nano`, the next intervention is the
token-velocity tracker described below.

### Why deferred

The two interventions we already have address two of the three causes
of 429 loss:

1. **Concurrency contention across models** — fixed by Phase 1.
   Embeddings no longer queue behind 30 in-flight completions for 10s+
   waiting on a slot.
2. **Lack of retry on transient 429s** — fixed by the retry layer.
   `Retry-After` / body-hint-driven sleeps with jitter and per-error
   retry budgets ride out short TPM saturation windows.

The third cause — **issuing a request that would obviously blow the
per-minute TPM budget** — is the one Phase 1 doesn't help with. A
naive concurrency cap approximates TPM only loosely: 30 completions ×
~6.5K avg tokens × 60s/avg-latency-15s ≈ 780K tokens/minute, which is
3.9× over `gpt-5-nano`'s 200K TPM limit. We get away with it today
because actual concurrency rarely sustains 30 in-flight, and because
the retry layer absorbs the overflow. But there's a bias-variance
trade: the more reliably the upstream client respects the bucket, the
fewer 429s OpenAI ever has to send.

A token-velocity tracker eliminates this category of waste entirely
— we'd stop issuing requests *we already know will fail* — at the
cost of a meaningful amount of moving parts (per-model state, header
parsing, blocking semantics, observability).

### Trigger to revisit

Build it when **any one** of the following holds after a representative
quality-eval run with the Phase 1 patch in place:

- ≥ 1 batch fails permanently (post-retries) with 429.
- p99 of `_compute_retry_delay`-driven sleeps exceeds 10s (= we're
  riding out long TPM windows that a tracker would have prevented).
- Cumulative wall-time spent inside `call_with_openai_retry`-induced
  sleep exceeds ~5% of pipeline runtime.
- We add a model whose published RPM/TPM is wildly different from
  `gpt-5-nano` and we don't trust the semaphore size to be in the
  right ballpark for it.

The first three are observable from existing log warnings; the last
one is a code-review signal.

### Sketch

A header-driven, per-model token-velocity tracker. Three responsibilities:

1. **Cheap-side budgeting.** Maintain a per-model rolling window of
   `(timestamp, tokens_used)` covering the last 60s. Before issuing a
   request, estimate its token cost (input prompt length + reserved
   `max_output_tokens`) and check whether the window has room. If not,
   block until the oldest entry expires far enough to make room.
2. **Header-driven correction.** OpenAI returns `x-ratelimit-remaining-
   requests`, `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-
   requests`, `x-ratelimit-reset-tokens` on every response. After each
   call, snap the tracker's "effective remaining" to whichever is
   smaller — our local estimate or the server's reported remaining.
   This converges the tracker to the truth even when our token estimate
   is wrong (e.g. tool-use turns, structured output overhead).
3. **Per-model isolation.** One tracker per `(api_key, model)` pair so
   `gpt-5-nano`'s pressure can't block embedding traffic, and so a
   future addition of `gpt-4.1-mini` or `gpt-5` doesn't share buckets
   with `gpt-5-nano`.

#### Public surface

```python
# llm.py (additive — does not break existing callers)

class _ModelLimiter:
    """Per-model RPM + TPM tracker. Thread-safe."""
    def __init__(self, *, rpm: int, tpm: int): ...
    def acquire(self, *, est_tokens: int): ...     # blocks if needed
    def update_from_headers(self, headers: dict): ...

_LIMITERS: dict[tuple[str, str], _ModelLimiter] = {}

def call_with_openai_retry(do_call, *, label, model: str | None = None,
                           est_tokens: int | None = None,
                           backoff_base=_BACKOFF_BASE_DEFAULT,
                           semaphore=None):
    """If `model` is provided, also gate on the per-model tracker.
    `semaphore` becomes a backstop (mostly redundant when the tracker
    is enforcing both RPM and TPM accurately) but is kept for safety
    against unknown-model paths and tracker bugs."""
```

Embedder + `llm_call` start passing `model=` and a token estimate;
older callers that don't pass `model` keep the current
semaphore-only behaviour.

#### What this subsumes

- **Phase 1 semaphores** become safety nets, not the primary control.
  We could keep them at generous values (e.g. 100/200) since the tracker
  is doing the real budgeting.
- **The body-regex retry hint** stays, but should fire much less often
  because we're no longer over-issuing.

#### What this does NOT solve

- **Multi-process / multi-host coordination.** The tracker is
  in-process. `reddit-intel-closed` runs the engine in one process, so
  this is fine for the open repo. Distributed deployments would need
  to swap the trackers for a Redis-backed implementation (the same
  swap point already documented for the legacy semaphore in
  PIPELINE.md §12).
- **Token estimation accuracy.** We can't perfectly estimate output
  tokens before the call. We rely on (a) reserving `max_output_tokens`
  pessimistically, (b) header-driven correction after the fact. There
  will still be the occasional 429 when our estimate undershoots and
  several requests land in the same window — the existing retry layer
  handles those.

#### Estimated work

Roughly 1 day:

- ~150 LOC for `_ModelLimiter` + window math + header parsing
- ~30 LOC of wiring through `call_with_openai_retry` + the two callers
- ~150 LOC of tests (window expiry, header correction, blocking
  semantics, per-model isolation, race conditions on update vs acquire)
- Doc updates here + PIPELINE.md

#### Worth-doing-with-it

If we end up touching this layer, two adjacent improvements ride free:

- **Surface budget pressure as a metric.** Log
  `tokens_used_in_window / tpm_limit` per call so we can plot it
  against pipeline runtime. The `evaluation/agentic_eval/` snapshot machinery
  already collects programmatic metrics — adding a "rate limit
  pressure" series there would let us validate Phase 2 the same way
  we validated Phase 1.
- **Per-tier auto-sizing.** Read `OPENAI_TIER` env var (or first-call
  headers) and size `rpm` / `tpm` from a small lookup table instead
  of hard-coding. Avoids a class of "I bumped the tier but forgot to
  bump the constant" bugs.

---

## `evaluation/agentic_eval/miners/` — per-snapshot automated failure-mode miners

### Status

**Deferred** until we have 4-5 reports across the new builder-validation corpora (Show-and-tell / Consumer-lifestyle / Vertical-professional / Existing-tool feedback) and can see which qualitative findings recur cross-corpus. Today we only have one report (`evaluation/agentic_eval/snapshots/openclaw_claudeai_sideproject/report.md`); building 8 detectors against one report risks shipping miners for openclaw-specific quirks.

### Why deferred

The cost of having a human evaluator agent re-derive findings per run is one wall-clock hour, not a permanent loss of signal — `dump.md` and `metrics.json` already do most of the grounding. The miners are an optimization on top, and optimizing prematurely against a one-corpus sample biases the detector thresholds to that corpus.

### Trigger to revisit

Build it when **any one** of the following holds after the new agentic_eval runs land:

- 3+ reports surface the same failure mode (mega-merge, polarity-flip, quoted_text mismatch, sweep rubber-stamp, dumping-ground leaf, low-margin reroute, sibling near-duplicate, recall hole) — that's the threshold past which mechanical detection saves real time.
- Engine code in `db/embeddings.py` or `category_worker.py` changes; we need automated regression coverage to know whether the change quietly broke a previously-clean corpus.
- We move agentic_eval from ad-hoc invocation to a CI cadence (e.g. weekly) — at that point manual re-derivation is no longer free.

### Sketch

New package `evaluation/agentic_eval/miners/`, one module per finding, each exposing `def run(snapshot_path: Path) -> dict`. New `miners/__init__.py` aggregates via `run_all(snapshot_path)` returning a dict, written next to `dump.md` / `metrics.json` as `findings.json` by `snapshot.take()`.

Concrete miners (each maps 1:1 to a finding from today's `report.md`, so the existing prose is the spec):

- `quoted_text_mismatch.py` — for every pending, check `quoted_text in (post_title || selftext)` when `comment_id IS NULL`, else in `comment_body`. Recommendation #3 made mechanical. Today: pending #426, #298, #425.
- `mega_merge_detector.py` — for every painpoint with `signal_count >= 8` compute pairwise cosine across linked pendings (reuse `evaluation/painpoints_eval/mega_merge_stress.py`'s `_fetch_pendings` + `cluster_at`); flag if `min_pairwise_cos < 0.45` or `n_components_at_0.65 > 1`. Today: pp #48 sig=20.
- `polarity_flip.py` — batched LLM polarity classifier on multi-source painpoints; flag mixed-polarity clusters. Today: pp #81 (no-code "works" vs "barrier"). ~$0.001/painpoint, cap to top 50.
- `sweep_rubber_stamp.py` — group `category_events` by event_type; flag any with `accept_rate == 1.0 AND proposed >= 10`. Today: 158/158 acceptance.
- `low_margin_reroute.py` — bucket reroute margins; list every painpoint moved with margin < 0.15. Today: events 51/53/136/153 → cat 326 dumping ground.
- `sibling_distinctness.py` — pairwise cosine of children's centroid embeddings per parent; flag pairs > 0.85. Today: cats #322/#323/#324 vs seed #50.
- `empty_or_dumping_leaves.py` — flag `direct_painpoints == 0` and `mean_intra_cos < 0.55 AND direct_painpoints >= 5`. Today: cat #319 (empty) and cat #326 (dumping).
- `recall_holes.py` — per subreddit `pendings_per_post` z-score; flag > 1.5σ below the run's median. Today: ClaudeAI 0.54 vs openclaw 1.12 (TPM rate-limit drops).

Wire-in: extend `snapshot.take()` to call `miners.run_all(snap_db)` and write `findings.json`. Extend `instructions/00_protocol.md` §3 to list `findings.json` as the third input alongside `dump.md` / `metrics.json`.

### Estimated work

~3-4 days. ~80-150 LOC per miner, ~400 LOC tests, ~100 LOC orchestration + protocol updates. `polarity_flip` is the only one that adds OpenAI cost; the rest are pure SQL + cosine math.

### What to do first when you pick this up

Re-read all available `report.md` files and rank findings by recurrence count. Implement miners in descending order of recurrence; ship after the first 2-3 even if the others remain on this list.

---

## `evaluation/category_eval/` — quantitative harness for Dim 4

### Status

**Deferred.** The slot is reserved in `evaluation/README.md` and the rationale is documented; standing it up needs the per-snapshot miners (above) to provide its ground truth and at least 3-4 reports to seed its fixtures.

### Why deferred

Dim 4 in the existing report scored 3/5 (mixed); Dim 3 (mega-merge / polarity flip) scored 2/5 (fail) and `painpoints_eval/` already covers Dim 3 quantitatively. The marginal next dollar of eval investment goes further on Dim 3 fixtures than on standing up a new sibling. Once the painpoints_eval coverage is wide enough that Dim 3 is locked in, Dim 4 becomes the next constraint.

The fixture work also requires gold examples lifted from multiple reports to be comprehensive — building them off one report's findings would over-fit to the 23 runtime cats in the openclaw/ClaudeAI/SideProject taxonomy.

### Trigger to revisit

- 3+ reports show recurring Dim 4 failures (sibling near-duplicates, dumping-ground leaves, low-margin reroutes, sweep rubber-stamping).
- `category_worker.py` is being actively modified; quantitative regression coverage on category placement becomes essential.
- The runtime taxonomy grows past ~50 runtime cats and visual review of `dump.md §5` no longer scales.

### Sketch

New package `evaluation/category_eval/` matching the shape of `painpoints_eval/`:

- `README.md` + `SEEDING.md` — own protocol; the fixture shape differs from `painpoints_eval/` (tuples + LLM-judge replays, not pair-cosines).
- `tree_quality.py` — given a snapshot, emit per-leaf coherence (mean intra-cluster cosine), sibling distinctness, empty leaves, dumping-ground leaves, runtime-vs-seed ratio. Mostly a thin wrapper over the per-snapshot miners — but presented as a single category-level scorecard.
- `proposer_judge_calibration.py` — replays `category_events` rows through a fresh LLM judge (different model than the proposer) and computes Cohen's κ between the live judge and the fresh one. Surfaces the rubber-stamp problem with a number, not just a 158/158 anecdote. Per-event-type κ for `add_category_split`, `painpoint_merge`, `reroute_painpoint`, `delete_category`.
- `fixtures/sibling_pairs.yaml` — gold "should merge" / "should stay separate" sibling category pairs. Seed from cats #322/#323/#324 (should merge into seed #50) plus clearly-distinct sibling pairs as keep-separate negatives. Schema mirrors painpoints_eval YAML; label = `merge` / `keep_separate`.
- `fixtures/placement_pairs.yaml` — gold "(painpoint, category) is correct" pairs. Seed from pp #381 alarm clock under `Sleep & Recovery` (correct) vs `Platform and Abuse Management` (wrong); pp #362 typing extension under `Productivity & Workflow` (correct) vs cat 326 (wrong).
- Per the **Promotion rule** in `evaluation/README.md`, `cosine_sim` (currently in `evaluation/painpoints_eval/_util.py`) graduates to `evaluation/shared/cosine.py` when this package actually uses it. Both consumers update at that move.

### Estimated work

~2-3 days. ~600 LOC including fixtures, ~250 LOC tests.

---

## Cross-run aggregator + committed baselines for `evaluation/`

### Status

**Deferred.** With one historical run today and four arriving in the next sprint, aggregation has limited signal yet — it pays off once we have 5+ runs and can detect recurring vs one-off findings.

### Why deferred

Aggregating across 5 runs gives much better signal than aggregating across 2; the inflection point lands right after the breadth-corpus runs do. Locking in baselines before deciding which per-snapshot miners we're building is also premature — the baseline shape depends on the miner output.

### Trigger to revisit

- 5+ reports exist (likely one sprint after the breadth runs land).
- Engine code starts changing more than weekly; we need a way to attribute "which dimension got better/worse" to a specific commit.
- Threshold tuning lands (`MERGE_COSINE_THRESHOLD` raise from 0.60 → ~0.70 per Recommendation #1 in today's report) — we need before/after numbers to defend the change.

### Sketch

- `evaluation/aggregate.py` — walks `evaluation/agentic_eval/runs/*/`, parses each `report.md` summary table + each `*/findings.json` (when miners exist), emits `evaluation/dashboard.md` with:
  - per-dimension score timeline (1 row per run, 4 columns)
  - per-failure-mode counts across runs (depends on miners landing)
  - per-subreddit difficulty (mean Dim 1 score across runs that used each subreddit)
  - recommendation recurrence (hash each `report.md` recommendation; mark recurring vs resolved)
- Lock today's `runs/pair_eval_painpoint_merge_pairs_20260419-233514.json` and `runs/threshold_sweep_painpoint_merge_pairs_20260419-233526.csv` as `evaluation/painpoints_eval/baselines/` (commit them — they are tiny and pin today's behaviour).
- `evaluation/painpoints_eval/compare_baseline.py` — diff latest pair_eval JSON vs. baseline; non-zero exit on regressions (per-pair: was TP, now FN, etc.).
- Same shape for `evaluation/agentic_eval/compare_baseline.py` once miners exist.

### Estimated work

~1-2 days. Largely glue code over JSON outputs that already exist (or will, once miners land).

### What already exists for cross-run comparison

The pairwise A/B comparison primitive landed early (out of order from this entry's plan) because the `evidence_filter` change needed it:

- `evaluation/agentic_eval/inspect_db.cross_run_pending_diff(snapshot_a_db, snapshot_b_db)` — joins two run snapshots by Reddit identity (`post.permalink`, `comment.permalink`) so synthetic row IDs renumbering between runs doesn't break the join. Returns `{common, only_a, only_b}`.
- `evaluation/agentic_eval/compare_runs.py` — CLI driver that emits a markdown report with headline metric deltas, cell-level overview, side-by-side render of changed common cells (sorted by |Δseverity|), and dropped/added pending tables sorted by severity.

When the aggregator/baseline work above lands, both should reuse `cross_run_pending_diff` rather than re-deriving the join.

---

<!-- Append future deferred-work entries above this line, newest first -->
