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
  centroid state (`member_emb_sum_blob`, `member_emb_count`), activity
  timestamps (`centroid_updated_at`, `member_set_last_changed_at`),
  and an `is_seed` flag distinguishing human-curated seed categories
  from LLM-minted runtime categories (drives per-category anchor
  weight — see §6).
- `painpoint_vec` / `category_vec` / `category_anchor_vec` — sqlite-vec
  virtual tables holding 1536-dim OpenAI embeddings (cosine distance).
  `category_vec` is the queryable blend; `category_anchor_vec` is the
  stable per-category name+description embedding.
- `category_fts` — SQLite FTS5 virtual table over
  `categories.(name, description)`, kept in lock-step with every
  category write. Backs the BM25 side of the hybrid (BM25 + dense)
  retrieval used by the split gate, uncat-review gate, and reroute —
  see `db/category_retrieval.py` and §7.

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

**Category vector is an anchor / mean blend, weighted by seed-status:**

```
category_vec = normalize(w · anchor + (1-w) · mean_of_members)
  where w = ANCHOR_WEIGHT_SEED    (0.85)  if categories.is_seed = 1
        w = ANCHOR_WEIGHT_RUNTIME (0.60)  if categories.is_seed = 0
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
- **Seed categories** have human-curated anchor text from
  `taxonomy.yaml`; the 0.85 weight keeps the declared intent in charge
  and prevents one off-topic member from dragging the centroid (the
  old pure-mean blend exhibited hijacking — e.g. "Generative Media"
  absorbing "Cold Email Deliverability"). 0.15 member share still lets
  evidence modulate.
- **Runtime-minted categories** have LLM-synthesized anchor text of
  variable quality, so 0.60 gives members a stronger say. Dropping to
  0.60 (from the seed 0.85) corrects for weak anchors without flipping
  to pure-mean, keeping drift bounded.
- `ANCHOR_WEIGHT` is kept as a backwards-compat alias pointing at
  `ANCHOR_WEIGHT_SEED` for older callers/tests that referenced it as
  a single constant.

**Anchor helpers:** `store_category_anchor`, `get_category_anchor`,
`delete_category_anchor`. `bootstrap_category_embeddings` seeds the
anchor table in a single `embed_batch` call for every seed category
that doesn't have one yet, then blends into `category_vec`.

**OpenAI concurrency budget.** Per-model: `OpenAIEmbedder._embed_with_retry`
acquires `OPENAI_EMBEDDING_SEMAPHORE` (default 80 slots) and `llm_call`
acquires `OPENAI_COMPLETION_SEMAPHORE` (default 30 slots). Both buckets
live in `llm.py` and route through the shared `call_with_openai_retry`
helper. The split (Phase 1) prevents completion fan-out — which holds
slots 5-10x longer than embedding calls — from starving embedding
callers; sizing reflects the ~5x RPM/TPM headroom embeddings get over
`gpt-5-nano` at tier 1. Phase 2 (token-velocity tracker) is documented
in `docs/IMPROVEMENTS.md`. The legacy `OPENAI_API_SEMAPHORE` /
`OPENAI_CONCURRENCY` symbols are kept as deprecated aliases of the
completion bucket for downstream consumers (see §12).

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
   painpoints that clustering left behind, **pre-filters** them by
   nearest-category cosine (skip if >= `CATEGORY_COSINE_THRESHOLD` —
   reroute will move those on its own), ranks the remainder by
   `signal_count × severity`, and asks `namer.decide_uncategorized`
   per painpoint (in parallel). The call gets a `nearest_hint=(name,
   cosine)` tuple so the LLM can see what the closest existing
   category is even though it's below the routing floor. The LLM
   returns either `action="keep"` (no event) or `action="create"` with
   a name / description / parent; the latter yields an
   `add_category_new` event with the LLM response pre-populated so the
   apply step skips a redundant naming call. Prompt is
   **default-keep**, creates only when (a) no existing branch is a
   reasonable home, (b) 3+ similar painpoints are plausibly incoming,
   and (c) the proposed name isn't a near-synonym of an existing
   entry. The pre-filter + tightened prompt together drove the
   `add_category_new` ACCEPT rate from 170+/sweep to near zero in the
   live E2E — reroute now handles what used to be the LLM's job for
   painpoints with a plausible existing home.

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
   category beats their current category by at least `REROUTE_MARGIN=0.10`
   (leave-one-out centroid for current fit — otherwise a pp's own
   embedding dominates its category centroid and artificially inflates
   its current-sim).

   **Hybrid retrieval** (`find_hybrid_candidates`): the best-other
   candidate is the top result of RRF-fused BM25 + dense, not pure
   dense top-K. BM25 over `category_fts` recalls keyword-rich
   candidates (rare-token and product-name matches like "Stripe",
   "ATS", "OAuth") that pure dense ranking was missing at rank 50+.
   The margin test still compares dense cosines, so
   REROUTE_MARGIN keeps its semantic meaning — hybrid only changes
   which candidate gets tested. Live E2E: this collapsed
   "Dating > Marketing and User Acquisition Challenges"-style
   misroutings into their natural homes in `App Business`.

   **Skip logic:** a painpoint is safe to skip if it has been
   reroute-checked before (`painpoints.reroute_checked_at` is set),
   its own state hasn't moved since that check, AND no category's
   centroid has moved globally since that check
   (`max(categories.centroid_updated_at)`). Stamps
   `reroute_checked_at = now()` on every painpoint we evaluated this
   sweep. Turns O(N_pps × K_cats) into O(changed_pps × K_cats).

### Creation gate + cross-parent replant (`db/category_retrieval.py`)

Every `_apply_add_category_new` and `_apply_add_category_split` call
runs proposal text through **hybrid retrieval** before minting a new
`categories` row:

```
hybrid = RRF_K(60)-fuse( dense top-K over category_vec,
                         BM25  top-K over category_fts )
```

Three outcomes per proposed sub-category (see `_decide_split_sub_fate`
for the split path; `_maybe_route_to_similar` for add_category_new):

- **Absorb** — top candidate's dense cosine >=
  `SIMILAR_CATEGORY_THRESHOLD=0.60` → route pp_ids into the existing
  category, skip the INSERT. Catches same-parent duplicates the split
  pass would otherwise produce (e.g., split of `Video Tools` proposing
  a sub called "Video Editing and Clipping" at cos 0.75 to the source).

- **Replant** (split only) — top candidate's dense cos in
  `[CROSS_PARENT_REPARENT_MIN_COS=0.45, SIMILAR_CATEGORY_THRESHOLD)`
  AND it lives under a *different* root than the split source, AND
  its dense cos beats the source's parent by >=
  `CROSS_PARENT_REPARENT_MARGIN=0.10` → create the new sub under the
  candidate's root instead of blindly inheriting the split source's
  parent. The split LLM only sees the source's immediate context, so
  a cluster of marketing-about-AI painpoints ends up proposed under
  `AI/ML` even though it belongs under `App Business`. Replant
  corrects the parent choice without a second LLM call.

- **Create** — neither absorb nor replant fires → mint the new
  category under the default parent (source's parent for split, LLM's
  proposed parent for add_category_new).

The accept gate uses **dense cosine** (not RRF score) because cosine
has a measurable semantic band from probing the live tree (noise
floor 0.21–0.32, distinct-siblings 0.34–0.53, duplicates 0.58–0.75);
RRF scores have no intrinsic scale. BM25 contributes recall — it
surfaces keyword-matching candidates dense top-K would have missed —
but the final accept is a dense-cos check.

`find_hybrid_candidates(conn, embedding, query_text, *, exclude_ids,
top_k)` is the reusable primitive — also called by
`propose_reroute_events` to pick the best-other candidate per
painpoint.

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

| Name                              | Where                    | Value | Purpose                                     |
| --------------------------------- | ------------------------ | ----- | ------------------------------------------- |
| `MERGE_COSINE_THRESHOLD`          | `db/embeddings.py`       | 0.60  | Promote link + sweep clustering.            |
| `PENDING_MERGE_THRESHOLD`         | `db/embeddings.py`       | 0.65  | Pending-stage (observation-level) dedup.    |
| `CATEGORY_COSINE_THRESHOLD`       | `db/embeddings.py`       | 0.35  | Below this → Uncategorized; also the Step 1b pre-filter floor. |
| `ANCHOR_WEIGHT_SEED`              | `db/embeddings.py`       | 0.85  | Anchor share of blended category_vec for seed categories. |
| `ANCHOR_WEIGHT_RUNTIME`           | `db/embeddings.py`       | 0.60  | Anchor share for LLM-minted runtime categories. |
| `SIMILAR_CATEGORY_THRESHOLD`      | `db/category_retrieval`  | 0.60  | Hybrid-gate absorb threshold (dense cos).   |
| `CROSS_PARENT_REPARENT_MIN_COS`   | `db/category_retrieval`  | 0.45  | Minimum dense cos before replant fires.     |
| `CROSS_PARENT_REPARENT_MARGIN`    | `db/category_retrieval`  | 0.10  | Top-candidate must beat source's parent by this. |
| `RRF_K`                           | `db/category_retrieval`  | 60    | Reciprocal Rank Fusion rank constant.       |
| `RELEVANCE_HALF_LIFE_DAYS`        | `db/relevance.py`        | 14    | Recency decay.                              |
| `MIN_SUB_CLUSTER_SIZE`            | `db/category_events`     | 3     | Min cluster size for a new category.        |
| `MIN_CATEGORY_SIZE_FOR_SPLIT`     | `db/category_events`     | 10    | Won't split categories smaller than this.   |
| `SPLIT_RECHECK_DELTA`             | `db/category_events`     | 10    | Growth needed before re-asking LLM to split.|
| `MERGE_CATEGORY_THRESHOLD`        | `db/category_events`     | 0.80  | Sibling-merge threshold (non-root).         |
| `MERGE_ROOT_CATEGORY_THRESHOLD`   | `db/category_events`     | 0.70  | Root-level merge threshold (looser).        |
| `CATEGORY_STALE_DAYS`             | `db/category_events`     | 30    | Member-set idle → propose delete.           |
| `REROUTE_MARGIN`                  | `db/category_events`     | 0.10  | Min cosine gap to propose reroute.          |
| `MAX_UNCAT_LLM_REVIEWS`           | `db/category_events`     | None  | Per-sweep cap on Uncat singleton reviews (unbounded). |
| `_LLM_PARALLEL_WORKERS`           | `db/category_events`     | 30    | Parallel LLM fan-out (≤ OPENAI_COMPLETION_CONCURRENCY). |
| `OPENAI_COMPLETION_CONCURRENCY`   | `llm.py`                 | 30    | Process-wide cap on in-flight OpenAI completion calls (gpt-5-nano / gpt-4.1-mini). |
| `OPENAI_EMBEDDING_CONCURRENCY`    | `llm.py`                 | 80    | Process-wide cap on in-flight OpenAI embedding calls (text-embedding-3-small). |
| `BATCH_TOKEN_BUDGET`              | `extractor.py`           | 2_000 | Approx tokens per LLM extraction batch.     |
| `LLM_CONCURRENCY`                 | `extractor.py`           | 40    | Parallel extraction batches.                |

---

## 10. Known quirks & gotchas

- **Seed taxonomy coverage.** The seed in `taxonomy.yaml` covers the
  tech stack (AI/ML, Developer Tools, Cloud, Data, Security, Fintech,
  Hardware) plus consumer / lifestyle roots (`Dating & Relationships`,
  `Health & Lifestyle`, `App Business`, `Business & Sales`,
  `Creator & Content`, `Productivity & Workflow`). The 3 consumer
  roots were added after observing the extractor emitting painpoints
  that had no seed home. If you add new roots or children, `seed_taxonomy`
  is idempotent (INSERT OR IGNORE) so new yaml entries propagate into
  existing DBs on the next `init_db()`. Run `python check_taxonomy.py`
  (needs `OPENAI_API_KEY`) to verify no pair of anchors embeds >=
  0.70 — failing pairs indicate descriptions that will routinely
  steal painpoints from each other at routing time.
- **Residual AI-sibling mis-routing is now mostly handled.** The
  creation gate + cross-parent replant (`_decide_split_sub_fate`)
  catches two of the three failure modes that used to cause this:
  (a) split minting a sub that's a near-duplicate of a same-parent
  sibling → absorbed; (b) split minting a sub whose content belongs
  under a different root → replanted. The third mode (same-parent
  siblings at cosine 0.55–0.75, below the 0.60 absorb threshold but
  above the 0.80 sibling-merge threshold) is the only residual case
  — the post-split sweep merge pass at 0.80 won't catch them and one
  sweep doesn't leave time for iteration. Two-threshold fix (drop
  sibling-merge to ~0.65 for same-parent pairs, drop root-merge to
  ~0.60) collapses the remaining cases.
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
| Hybrid (BM25 + dense) retrieval, FTS5 sync, creation gate thresholds | `db/category_retrieval.py` |
| Taxonomy anchor-distinguishability check | `check_taxonomy.py`             |
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
| Creation-gate + replant tests    | `tests/test_creation_gate.py`           |
| OpenAI shared utilities + per-model semaphores | `llm.py` (`OPENAI_COMPLETION_SEMAPHORE`, `OPENAI_EMBEDDING_SEMAPHORE`, `call_with_openai_retry`, `llm_call`, `TokenCounter`) |

---

## 12. Public API (for downstream consumers)

This repo is installable as a library (`pip install -e .`). Downstream
consumers — specifically the closed `reddit-intel-closed` backend — import
a small, stable surface from here. Treat these symbols as a
**semver-stable public API**:

| Symbol                                      | Used for                                  |
| ------------------------------------------- | ----------------------------------------- |
| `reddit_scraper.scrape_subreddit_full`      | Own-API Reddit scrape (OAuth, 60 rpm)     |
| `painpoint_extraction.extract_painpoints`   | LLM extraction with engine SQLite I/O (solo-agent path) |
| `painpoint_extraction.extract_painpoints_from_posts` | Pure dict-in / dict-out extraction (no DB) — for consumers with their own storage |
| `db.embeddings.OpenAIEmbedder`              | 1536-dim embeddings, `embed` + `embed_batch` |
| `llm.OPENAI_COMPLETION_SEMAPHORE` / `llm.OPENAI_EMBEDDING_SEMAPHORE` | Per-model concurrency caps — downstream may swap either for a distributed implementation at process boot. Legacy `llm.OPENAI_API_SEMAPHORE` is a deprecated alias of the completion bucket and will be removed in a future major bump. |
| Pending-painpoint dict shape (fields: `title`, `description`, `severity`, `quoted_text`, `category_name`, `post_id`, `comment_id?`) | Data handoff from `extract_painpoints` |

**Stability rules:**

- Rename / remove / break the shape of any of the above → **major**
  version bump.
- Additive change (new field on the pending dict, new helper) → **minor**
  bump. Consumers ignore unknown fields.
- Any other internal refactor → **patch** bump.

**Everything else is internal.** `db/painpoints.py`, `db/category_events.py`,
`db/categories.py`, `db/category_clustering.py`, `db/relevance.py`,
`db/locks.py`, `db/queries.py`, `promoter.py`, `category_worker.py`
are NOT part of the public API. They're free to refactor at will; the
closed backend reimplements the merge + taxonomy logic on Postgres
rather than importing these.

---

## 13. First steps for a new agent

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
