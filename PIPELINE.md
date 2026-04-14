# Pipeline

Orientation guide for a new agent picking up this repo. Explains what the
pipeline does, how the pieces fit together, and which function to open
when you need to change something.

---

## 1. What this project does

`reddit-intel` scrapes Reddit subreddits, extracts user "painpoints"
with an LLM, and organises them into a self-maintaining category
taxonomy so they can feed downstream startup-idea generation.

End state in the DB:

- `posts` / `comments` — raw Reddit text.
- `pending_painpoints` — one row per LLM-emitted painpoint observation
  (immutable, append-only).
- `painpoints` — the merged/canonical table; one row per distinct
  painpoint, referencing N pendings via `painpoint_sources`.
- `categories` — a tree of categories seeded from `taxonomy.yaml` and
  mutated at runtime by the category worker. Each row carries cached
  centroid state (`member_emb_sum_blob`, `member_emb_count`) plus
  activity timestamps (`centroid_updated_at`,
  `member_set_last_changed_at`).
- `painpoint_vec` / `category_vec` / `category_anchor_vec` — sqlite-vec
  virtual tables holding 1536-dim OpenAI embeddings (cosine distance).
  `category_vec` is the queryable blend; `category_anchor_vec` is the
  stable per-category name+description embedding.

---

## 2. Stage overview

```
reddit ─► scraper ─► posts/comments ─► extractor (LLM)
                                         │
                                         ▼
                              pending_painpoints
                                         │
                                  promoter (embedding)
                              ┌──────────┴──────────┐
                        link to existing       create new
                         painpoint (≥0.60)      painpoint
                                         │
                                         ▼
                              painpoints (merged)
                                         │
                            category_worker (periodic sweep)
          ┌─────────────┬─────────────┬──────┬──────┬──────┬─────────┐
      Uncategorized  uncat-LLM     split   delete merge  reroute
      clustering    review (Step 1b)
          └─────────────┴─────────────┴──────┴──────┴──────┴─────────┘
```

The **scrape → extract → promote** path runs inline per subreddit. The
**category worker** runs as a separate process (or in the live E2E
test) and is the only component that mutates the taxonomy.

---

## 3. Entry points

| Command                                          | What it does                                                                 |
| ------------------------------------------------ | ---------------------------------------------------------------------------- |
| `python -c "import asyncio, subreddit_pipeline as p; asyncio.run(p.analyze('Foo'))"` | Full scrape→extract→promote for one subreddit.   |
| `python promoter.py`                             | Long-running daemon that drains `pending_painpoints` forever.                |
| `python category_worker.py`                      | One-shot taxonomy sweep.                                                     |
| `pytest tests/test_e2e_real_subreddits.py -v -s` | Live end-to-end on 5 real subreddits with real OpenAI; prints the tree.      |
| `pytest tests/test_painpoint_pipeline.py`        | Hermetic unit/integration tests using `FakeEmbedder` + `FakeNamer`.          |
| `pytest tests/test_category_anchor.py`           | Anchor behaviour: storage, blending, hijack resistance.                      |
| `pytest tests/test_uncat_llm_review.py`          | Uncategorized-singleton LLM review path.                                     |

---

## 4. Stage 1 — Scrape (`reddit_scraper.py`)

`scrape_subreddit_full(subreddit, *, min_score=None)` — async; returns a
list of post dicts, each with `comments` nested. Uses the public Reddit
JSON endpoints (no login). Tunables at the top of the file:
`POSTS_PER_WINDOW`, `POSTS_WITH_COMMENTS`, `COMMENT_DEPTH`.

Posts are persisted by `_persist_scrape` in `subreddit_pipeline.py` via
`db.posts.upsert_post` + `db.posts.upsert_comment`. Upserts are
idempotent on Reddit's `name` (`t3_…`) so re-scrape is safe.

---

## 5. Stage 2 — Extract (`painpoint_extraction/extractor.py`)

`extract_painpoints(post_ids) -> (pending_ids, token_usage)`

Pipeline:
1. `_filter_unextracted` — skip any post with existing
   `pending_painpoints` rows.
2. `_load_posts_with_comments` — fetch from DB.
3. `_build_batches` — greedy bin-pack posts until `BATCH_TOKEN_BUDGET`
   (~2K tokens per batch).
4. `_build_instructions` — inject the live taxonomy into the prompt.
5. `_process_batch` × N in parallel via
   `asyncio.Semaphore(LLM_CONCURRENCY=40)`. Uses `ExtractionResult`
   (Pydantic) as OpenAI structured output with `MODEL = "gpt-4.1-mini"`.
6. `_fix_attribution` — corrects `comment_id` by searching
   `quoted_text` in the source material.
7. `save_pending_painpoints_batch` — persists everything in one
   transaction.

**Extraction prompt scope.** The prompt accepts **any app-addressable
painpoint**, not just technical/developer ones. Consumer frustrations,
dating / relationship friction, lifestyle and habit pains, creator
workflow struggles all qualify if an app / extension / API / service
could plausibly ease them. Skip rules still reject pure opinions,
philosophical takes with no product hook, political drama, and
pricing-only gripes. This expansion turned BuildToAttract and
rSocialskillsAscend from ~20-painpoint slow lanes into 80+ each.

Key schema: `ExtractedPainpoint(title, description, severity 1–10,
quoted_text, category_name, post_id, comment_id?)`.

---

## 6. Stage 3 — Promote (`promoter.py` + `db/painpoints.py`)

### `promoter.run_once(embedder=None) -> {"processed", "linked", "lock_timeouts"}`

1. `pick_unmerged_pending(conn, limit=None)` — drains the **full** queue
   by default (the old 100-row cap left ~40% of pendings orphaned per
   E2E pass once extraction volume grew; callers that want a cap can
   still pass one).
2. Batch-fetch every pending's `title + description` and call
   `embedder.embed_batch(texts)` — one HTTP round-trip per 256
   pendings instead of N.
3. For each pending call `promote_pending(pp_id, embedding=emb)`.
   `TimeoutError` on the merge_lock is counted as a soft skip; those
   pendings stay in the queue for the next pass.

### `db.painpoints.promote_pending(pending_id, *, embedder, embedding)`

Three-step contract:

| Step | Where                 | What                                                                                      |
| ---- | --------------------- | ----------------------------------------------------------------------------------------- |
| 1    | outside lock          | Ensure we have an embedding — `embedder.embed(text)` if not pre-computed by `run_once`.   |
| 2    | inside `merge_lock`   | `find_most_similar_painpoint`. If cosine ≥ `MERGE_COSINE_THRESHOLD` (0.60) → link.         |
| 3    | inside `merge_lock`   | Otherwise `_create_painpoint_from_pending` — places the new painpoint via `find_best_category`. |

No relevance-based drop. Every pending is either linked to an
existing painpoint or spawns a new one. Staleness lives at the
category level (`CATEGORY_STALE_DAYS=30`), not per-pending.

**`_link_pending_to_painpoint`** bumps `signal_count` and inserts a
`painpoint_sources` row. It does **NOT** touch the category centroid —
linking adds a pending source to an existing painpoint, so the
painpoint's embedding and the category's member set are unchanged.
The old code called `update_category_embedding` here unconditionally;
on a batch promote that was O(N²) wasted work against `painpoint_vec`.

**`_create_painpoint_from_pending`** stores the embedding, calls
`add_member_to_centroid(category_id, embedding)` (O(1) update to the
cached sum/count), then `update_category_embedding` to refresh the
category's blended vector.

### Embeddings & centroids (`db/embeddings.py`)

- `OpenAIEmbedder` — `text-embedding-3-small`, 1536-dim, exponential
  retry/backoff. Has `embed(text)` and `embed_batch(texts, batch_size=256)`.
  Empty/whitespace input is sanitized to `" "`.
- `FakeEmbedder` — word-hash→seeded-Gaussian test double, no network.
  Empty input maps to `"__empty__"` so we never produce an all-zero vector.
- `get_painpoint_embedding(conn, pp_id)` — unpacks one painpoint's vec0
  blob into a Python list.
- `find_best_category(conn, emb, embedder=None)` — top-1 above
  `CATEGORY_COSINE_THRESHOLD`, else Uncategorized.
- `find_best_category_ranked(conn, emb, limit=50)` — full top-K for
  reroute logic.

**Category vector is an anchor / mean blend:**

```
category_vec = normalize(ANCHOR_WEIGHT · anchor + (1-ANCHOR_WEIGHT) · mean_of_members)
```

- `anchor` = embedding of `name + description` stored in
  `category_anchor_vec`. Static per category; re-computed only when the
  description actually changes (seed bootstrap, split subcategory create,
  merge description refresh).
- `mean_of_members` = `member_emb_sum_blob / member_emb_count` (both on
  the `categories` row). Maintained **incrementally** by
  `add_member_to_centroid` / `remove_member_from_centroid` — O(1) per
  mutation. `rebuild_centroid_from_members` is the full-scan fallback
  used on bulk moves (delete, merge) and as a legacy-cache repair.
- `ANCHOR_WEIGHT = 0.85` — the anchor dominates. Pure mean-of-members
  (what we had before) drifted: one off-topic member pulled the
  centroid, which made the next off-topic member easier to attract, a
  feedback loop that produced the observed hijacking (e.g. "Generative
  Media" absorbing "Cold Email Deliverability"). Anchoring stops the
  drift; 0.15 member share still lets evidence modulate.

**Anchor helpers:** `store_category_anchor`, `get_category_anchor`,
`delete_category_anchor`. `bootstrap_category_embeddings` seeds the
anchor table in a single `embed_batch` call for every seed category
that doesn't have one yet, then blends into `category_vec`.

**OpenAI concurrency budget.** `OpenAIEmbedder._embed_with_retry`
acquires the global `OPENAI_API_SEMAPHORE` from `llm.py` around every
HTTP call. Embeddings + completions share one process-wide budget of
`OPENAI_CONCURRENCY=10` in-flight calls. Sized for tier-1 RPM limits.

### Relevance (`db/relevance.py`)

Single helper: `per_source_relevance(post, comment, severity) → float`.

```
per_source_relevance = traction × recency
```

- Post-rooted: `traction = log1p(score)·0.5 + log1p(num_comments)·0.8`.
- Comment-rooted: `traction = log1p(comment_score)·0.5`. The post's
  popularity doesn't validate a random comment on it, so the comment
  carries the signal alone.
- `recency = 0.5 ** (age_days / 14)` (half-life 14 days).
- `severity` is accepted for API compatibility but does not affect the
  result (the LLM's 1-10 claim was too subjective to weight on).

All other relevance helpers (`compute_pending_relevance`,
`compute_painpoint_relevance`, the `relevance` / `relevance_updated_at`
columns) were removed — nothing in the live pipeline consumed them
after the promoter stopped filtering on relevance and the delete pass
switched to `CATEGORY_STALE_DAYS`.

---

## 7. Stage 4 — Category worker (`category_worker.py` + `db/category_events.py`)

One sweep acquires `merge_lock` once and runs **six passes in order**:

1. **Uncategorized → cluster → `add_category_new`**
   `propose_uncategorized_events` clusters the Uncategorized bucket via
   `cluster_painpoints` at `MERGE_COSINE_THRESHOLD=0.60`. Clusters of
   size ≥ `MIN_SUB_CLUSTER_SIZE=3` yield an `add_category_new` event
   with the LLM picking the name + parent via
   `namer.name_new_category`.

2. **Uncategorized singletons → LLM review → `add_category_new`** (Step 1b)
   `propose_uncategorized_singleton_events` takes the remaining Uncat
   painpoints that clustering left behind, ranks them by
   `signal_count × severity`, caps at top `MAX_UNCAT_LLM_REVIEWS=50`
   per sweep, and asks `namer.decide_uncategorized` per painpoint (in
   parallel). The LLM returns either `action="keep"` (no event) or
   `action="create"` with a name / description / parent; the latter
   yields an `add_category_new` event with the LLM response
   pre-populated so the apply step skips a redundant naming call.
   Prompt is **conservative** — default is keep, create only for
   distinct actionable concerns with plausible future siblings.

3. **Split crowded categories → `add_category_split`**
   `propose_split_events` skips categories below
   `MIN_CATEGORY_SIZE_FOR_SPLIT=10` or that haven't grown by
   `SPLIT_RECHECK_DELTA=10` since the last check. For the rest it
   clusters members, summarises the top 10 clusters, and asks
   `namer.decide_split` in parallel across all candidates. LLM returns
   `SplitDecision(decision, reason, subcategories)`; split decisions
   yield one `add_category_split` event carrying the sub-category
   payload (name, description, painpoint ids per sub).

4. **Delete stale → `delete_category`**
   `propose_delete_events` proposes delete for any non-Uncategorized
   category whose `member_set_last_changed_at` is older than
   `CATEGORY_STALE_DAYS=30` (dedicated staleness signal —
   `signal_count` bumps don't count as "category activity"). The
   acceptance test re-checks staleness under the lock so a member
   added between propose and apply keeps the category alive.

5. **Merge duplicate siblings → `merge_categories`**
   `propose_merge_events` computes `inter_category_similarity` for
   every sibling pair. Proposes merge if cosine ≥
   `MERGE_CATEGORY_THRESHOLD=0.80`. The pre-apply test rejects pairs
   where either side was already merged away earlier in the sweep
   (cascade safety).

6. **Reroute stragglers → `reroute_painpoint`**
   `propose_reroute_events` finds painpoints whose best-matching
   category beats their current category by at least `REROUTE_MARGIN=0.08`
   (leave-one-out centroid for current fit — otherwise a pp's own
   embedding dominates its category centroid and artificially inflates
   its current-sim).

   **Skip logic:** a painpoint is safe to skip if it has been
   reroute-checked before (`painpoints.reroute_checked_at` is set),
   its own state hasn't moved since that check, AND no category's
   centroid has moved globally since that check
   (`max(categories.centroid_updated_at)`). Stamps
   `reroute_checked_at = now()` on every painpoint we evaluated this
   sweep. Turns O(N_pps × K_cats) into O(changed_pps × K_cats).

### Event model (`CategoryEvent`)

```
event_type:      add_category_new | add_category_split |
                 delete_category  | merge_categories |
                 reroute_painpoint
payload:         dict (event-specific)
target_category: int (the category this acts on)
triggering_pp:   optional int
metric_name/value/threshold: audit fields
llm_result:      optional pre-fetched LLM response
```

`apply_with_test(conn, event, namer, embedder=None)` is the uniform
runner: run acceptance test → on pass, open a SAVEPOINT, call
`_apply_<event_type>`, log to `category_events` table. Every sweep
mutation — accepted or rejected — lands in that audit log. `embedder`
is threaded through so `_apply_add_category_new` / `_apply_add_category_split`
can write an anchor for each new subcategory, and
`_apply_merge_categories` can re-anchor the survivor when the
description refresh mints a new description.

### `ACCEPT` in the audit log

Three gates to an `ACCEPT`:
1. Proposed — the propose_* generator emitted the event because its
   metric cleared the threshold.
2. Acceptance test passed — cascade-safety checks under the lock
   (categories still exist, painpoint still in its from_cat, etc.).
3. Apply succeeded — the mutation ran inside a SAVEPOINT without
   raising.

`REJECT` lines carry a reason in the audit log so you can tell whether
it was "cascade from earlier merge" vs "apply error: foo".

### Parallel LLM helpers

- `parallel_namer_calls(specs, max_workers=_LLM_PARALLEL_WORKERS=5)` —
  fans out arbitrary LLM calls across a thread pool. Returns
  `(results, errors)` so callers distinguish "LLM returned None" from
  "LLM call raised".
- `prefetch_llm_batch(conn, events, namer)` — runs LLM calls
  concurrently before the serial `apply_with_test` loop. Results
  attach to `event.llm_result`; apply functions use the pre-fetched
  response or fall back to an inline call.
- Used by: Step 1b (decide_uncategorized), Step 3 (decide_split), and
  `add_category_new` / `merge_categories` prefetch.

### LLM naming (`db/llm_naming.py`)

- `LLMNamer` uses `gpt-4.1-mini` (`_STRUCTURED_MODEL`). Reasoning
  models return `None` for structured output (they burn the token
  budget on reasoning).
- `name_new_category(titles, descriptions, existing_taxonomy)` —
  returns `{"name", "description", "parent" or null}`.
- `decide_split(name, desc, total, clusters)` — returns
  `SplitDecision` (Pydantic).
- `decide_uncategorized(title, desc, signal, severity, taxonomy)` —
  returns `UncatDecision(action, reason, name?, description?, parent?)`.
  Default-conservative (favours keep).
- `describe_merged_category(survivor, loser, sample_titles)` —
  returns `{"description"}`, used to refresh the survivor's
  description post-merge.
- `FakeNamer` — test double; `decide_uncategorized` always returns
  `keep` so hermetic tests don't spuriously spawn categories.

---

## 8. Data model conventions

- **3 painpoint tables, not 2**: `pending_painpoints` is append-only
  raw. Merged rows live in `painpoints`, linked via
  `painpoint_sources(painpoint_id, pending_painpoint_id)`. The
  promoter never deletes pendings.
- **`pending_painpoint_all_sources`** is a VIEW unioning
  `pending_painpoints.(post_id, comment_id)` (primary) with
  `pending_painpoint_sources` (extras from batched extraction).
- **Uncategorized** — id-stable sentinel, never deleted. Painpoints
  land here when no category anchor matches above
  `CATEGORY_COSINE_THRESHOLD=0.35`.
- **Category centroid state** — three pieces per category:
  - `category_anchor_vec.embedding` — static name+description vector.
  - `categories.member_emb_sum_blob` + `member_emb_count` — raw sum
    and count of member embeddings; maintained incrementally.
  - `category_vec.embedding` — the blended vector used for matching.
    Recomputed cheaply from the two above on every mutation.
  - `categories.centroid_updated_at` / `member_set_last_changed_at` —
    timestamps feeding reroute-skip and staleness-delete respectively.
    `member_set_last_changed_at` fires only on real add/remove;
    `centroid_updated_at` also fires on cache repair.
  - Uncategorized is centroid-exempt.
- **Bootstrap centroids** (`bootstrap_category_embeddings`): seeded
  taxonomy categories get an anchor from `name + description` in one
  `embed_batch` call. Eagerly triggered from `init_db()` when an
  `OPENAI_API_KEY` is present, otherwise lazy on first
  `find_best_category`.
- **`merge_lock`** (`db/locks.py`) — coarse advisory lock held for
  every mutation path. One sqlite writer anyway, but the explicit
  lock makes the boundaries obvious and protects savepoint semantics.
- **`in_clause_placeholders(n)`** (`db/__init__.py`) — centralised
  helper for SQL `IN (?,?,…)` clauses. Used by the few queries that
  pass a Python list of IDs.

---

## 9. Thresholds & tunables (quick reference)

| Name                          | Where                | Value | Purpose                                     |
| ----------------------------- | -------------------- | ----- | ------------------------------------------- |
| `MERGE_COSINE_THRESHOLD`      | `db/embeddings.py`   | 0.60  | Promote link + sweep clustering.            |
| `CATEGORY_COSINE_THRESHOLD`   | `db/embeddings.py`   | 0.35  | Below this → Uncategorized.                 |
| `ANCHOR_WEIGHT`               | `db/embeddings.py`   | 0.85  | Anchor share of the blended category_vec.   |
| `RELEVANCE_HALF_LIFE_DAYS`    | `db/relevance.py`    | 14    | Recency decay.                              |
| `MIN_SUB_CLUSTER_SIZE`        | `db/category_events` | 3     | Min cluster size for a new category.        |
| `MIN_CATEGORY_SIZE_FOR_SPLIT` | `db/category_events` | 10    | Won't split categories smaller than this.   |
| `SPLIT_RECHECK_DELTA`         | `db/category_events` | 10    | Growth needed before re-asking LLM to split.|
| `MERGE_CATEGORY_THRESHOLD`    | `db/category_events` | 0.80  | Sibling-merge threshold.                    |
| `CATEGORY_STALE_DAYS`         | `db/category_events` | 30    | Member-set idle → propose delete.           |
| `REROUTE_MARGIN`              | `db/category_events` | 0.08  | Min cosine gap to propose reroute.          |
| `MAX_UNCAT_LLM_REVIEWS`       | `db/category_events` | 50    | Per-sweep cap on Uncat singleton reviews.   |
| `_LLM_PARALLEL_WORKERS`       | `db/category_events` | 5     | Parallel LLM fan-out (below OPENAI_CONCURRENCY). |
| `OPENAI_CONCURRENCY`          | `llm.py`             | 10    | Process-wide cap on in-flight OpenAI calls. |
| `BATCH_TOKEN_BUDGET`          | `extractor.py`       | 2_000 | Approx tokens per LLM extraction batch.     |
| `LLM_CONCURRENCY`             | `extractor.py`       | 40    | Parallel extraction batches.                |

---

## 10. Known quirks & gotchas

- **Seed taxonomy coverage.** The seed in `taxonomy.yaml` is tech-heavy
  (AI/ML, Creator & Content, Fintech, Productivity). With the relaxed
  extraction prompt the scraper now emits dating / relationships /
  social skills / fitness painpoints that have no seed home. Step 1b
  (uncat LLM review) bootstraps new roots for those over time; if you
  want faster coverage, extend `taxonomy.yaml` with consumer / dating /
  health / lifestyle roots.
- **Residual AI-sibling mis-routing.** Sub-categories of AI/ML
  ("AI Coding Tools", "AI Agent Workflow", etc.) share a lot of
  vocabulary in their anchors. A generic AI-business painpoint can
  match several anchors at similar cosine and land in the wrong
  sibling. Iterating sweeps doesn't help (the blend is 85% anchor, so
  centroids barely move between passes). The real fix is enriching
  anchors with exemplar titles; deferred.
- **OpenAI latency variability.** Live E2E runtimes have been observed
  to fluctuate 3× between otherwise identical runs (173 s → 775 s)
  purely from OpenAI-side latency. Zero retries in the log + same
  workload = not a code regression. If a run is suspiciously slow,
  rerun before investigating the code.
- **Structured output + reasoning models.** `gpt-5-nano` returns
  `None` for structured output because reasoning tokens exhaust the
  budget. Use `gpt-4.1-mini` (the default).
- **OpenAI json_object mode.** `_call` appends "Return your answer as
  a single JSON object" to the user message — the API errors if the
  literal word "json" isn't in the input.
- **Merge cascade.** `propose_merge_events` emits every candidate pair
  up front; earlier merges in the batch can delete one side of a later
  pair. `_test_merge_categories` guards against this.
- **FK on hallucinated post_ids.** The LLM sometimes emits a `post_id`
  that was never in the input batch. `save_pending_painpoint(s)[_batch]`
  validates existence before insert and skips silently with a warning.
- **Pytest output buffering.** The E2E test prints a lot; output only
  flushes when the test completes unless you pass `flush=True`.
  Don't kill a run assuming it hung — it's probably waiting on OpenAI.

---

## 11. Where things live

| Concern                          | Path                                    |
| -------------------------------- | --------------------------------------- |
| Pipeline orchestration           | `subreddit_pipeline.py`                 |
| Scraper                          | `reddit_scraper.py`                     |
| LLM extraction                   | `painpoint_extraction/extractor.py`     |
| Promoter loop                    | `promoter.py`                           |
| Promote core                     | `db/painpoints.py`                      |
| Embeddings, anchors, incremental centroid | `db/embeddings.py`             |
| Relevance                        | `db/relevance.py`                       |
| Category worker driver           | `category_worker.py`                    |
| Category events / proposers / appliers / tests | `db/category_events.py`   |
| LLM-naming wrappers              | `db/llm_naming.py`                      |
| Sweep clustering helpers         | `db/category_clustering.py`             |
| Category CRUD helpers            | `db/categories.py`                      |
| Taxonomy seed                    | `taxonomy.yaml`, `db/seed.py`           |
| Schema + migrations              | `db/schema.sql`, `db/__init__.py`       |
| Merge lock                       | `db/locks.py`                           |
| Live E2E test                    | `tests/test_e2e_real_subreddits.py`     |
| Hermetic pipeline test           | `tests/test_painpoint_pipeline.py`      |
| Anchor tests                     | `tests/test_category_anchor.py`         |
| Uncat LLM-review tests           | `tests/test_uncat_llm_review.py`        |
| OpenAI shared utilities + semaphore | `llm.py` (`OPENAI_API_SEMAPHORE`, `llm_call`, `TokenCounter`) |

---

## 12. First steps for a new agent

1. Skim `subreddit_pipeline.analyze` — ~70 lines, the canonical "what
   does one run look like" reference.
2. Read `db/category_events.py` top-to-bottom — it's the most complex
   file and the one you'll touch most.
3. Read `db/embeddings.py` sections on anchors + incremental centroid
   — the blend semantics are load-bearing and not obvious from the
   function names alone.
4. Run the hermetic tests: `pytest tests/test_painpoint_pipeline.py
   tests/test_category_anchor.py tests/test_uncat_llm_review.py`.
   They use `FakeEmbedder` + `FakeNamer` so no API keys required.
5. If the task touches taxonomy behaviour, run
   `pytest tests/test_e2e_real_subreddits.py -v -s` and read the
   printed tree. Needs `OPENAI_API_KEY`, costs cents, takes 3-6 min.
