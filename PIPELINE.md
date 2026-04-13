# Pipeline

Orientation guide for a new agent picking up this repo. Explains what the
pipeline does, how the pieces fit together, and which function to open
when you need to change something.

---

## 1. What this project does

`reddit-intel` scrapes developer-oriented subreddits, extracts user
"painpoints" with an LLM, and organises them into a self-maintaining
category taxonomy so they can feed downstream startup-idea generation.

End state in the DB:

- `posts` / `comments` — raw Reddit text.
- `pending_painpoints` — one row per LLM-emitted painpoint observation
  (immutable, append-only).
- `painpoints` — the merged/canonical table; one row per distinct
  painpoint, referencing N pendings via `painpoint_sources`.
- `categories` — a tree of categories seeded from `taxonomy.yaml` and
  mutated at runtime by the category worker.
- `painpoint_vec` / `category_vec` — sqlite-vec virtual tables holding
  1536-dim OpenAI embeddings (cosine distance).

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
                         ┌──────┬───────┬──────┬──────┐
                    Uncategorized  split  delete  merge
                         └──────┴───────┴──────┴──────┘
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
| `python category_worker.py`                      | One-shot taxonomy sweep (Uncategorized/split/delete/merge).                  |
| `pytest tests/test_e2e_real_subreddits.py -v -s` | Live end-to-end on 5 real subreddits with real OpenAI; prints the tree.     |
| `pytest tests/test_painpoint_pipeline.py`        | Hermetic unit/integration tests using `FakeEmbedder` + `FakeNamer`.         |

---

## 4. Stage 1 — Scrape (`reddit_scraper.py`)

`scrape_subreddit_full(subreddit, *, min_score=None)` — async; returns a
list of post dicts, each with `comments` nested. Uses the public Reddit
JSON endpoints (no login). Tunables live at the top of the file:
`POSTS_PER_WINDOW`, `POSTS_WITH_COMMENTS`, `COMMENT_DEPTH`.

Posts are then persisted by `_persist_scrape` in
`subreddit_pipeline.py` via `db.posts.upsert_post` +
`db.posts.upsert_comment`. Upserts are idempotent on Reddit's `name`
(`t3_…`) so re-scrape is safe.

---

## 5. Stage 2 — Extract (`painpoint_extraction/extractor.py`)

`extract_painpoints(post_ids) -> (pending_ids, token_usage)`

Pipeline:
1. `_filter_unextracted` — skip any post that already has rows in
   `pending_painpoints`.
2. `_load_posts_with_comments` — fetch from DB.
3. `_build_batches` — greedy bin-pack posts until `BATCH_TOKEN_BUDGET`
   (~2K tokens per batch); this keeps the LLM input within a single
   response and lets us parallelize.
4. `_build_instructions` — inject the live taxonomy into the prompt
   (`db.categories.get_category_list_flat()`). The LLM is asked to pick
   a `category_name` from this list or return `"Uncategorized"`.
5. `_process_batch` × N in parallel via `asyncio.Semaphore(LLM_CONCURRENCY=40)`.
   Uses `ExtractionResult` (Pydantic) as OpenAI structured output with
   `MODEL = "gpt-4.1-mini"` (non-reasoning — faster and reliable for
   this shape of task).
6. `_fix_attribution` — corrects `comment_id` by searching
   `quoted_text` in the source material (the LLM sometimes attributes
   to the wrong comment).
7. `save_pending_painpoints_batch` — persists everything in one
   transaction.

Key schema: `ExtractedPainpoint(title, description, severity 1–10,
quoted_text, category_name, post_id, comment_id?)`.

---

## 6. Stage 3 — Promote (`promoter.py` + `db/painpoints.py`)

### `promoter.run_once(embedder=None) -> {"processed", "linked"}`

1. `pick_unmerged_pending(conn, limit=100)` — pendings with no
   `painpoint_sources` row yet. **NOTE**: hardcoded 100-row cap; the
   pipeline calls this once per subreddit so a large backlog needs
   multiple invocations (known gotcha — see §10).
2. Batch-fetch every pending's `title + description` and call
   `embedder.embed_batch(texts)` — one HTTP round-trip per 256
   pendings instead of N.
3. For each pending call `promote_pending(pp_id, embedding=emb)`.

### `db.painpoints.promote_pending(pending_id, *, embedder, embedding, now)`

Three-step contract:

| Step | Where                 | What                                                                                      |
| ---- | --------------------- | ----------------------------------------------------------------------------------------- |
| 1    | outside lock          | Ensure we have an embedding — `embedder.embed(text)` if not pre-computed by `run_once`.   |
| 2    | inside `merge_lock`   | `find_most_similar_painpoint`. If cosine ≥ `MERGE_COSINE_THRESHOLD` (0.60) → link.         |
| 3    | inside `merge_lock`   | Otherwise `_create_painpoint_from_pending` — places the new painpoint via `find_best_category`. |

The pending → drop step (compute_pending_relevance against
`MIN_RELEVANCE_TO_PROMOTE`) was removed: the LLM extractor is the
filter, not relevance. Every pending is linked or creates a new
painpoint. Staleness is handled at the category level in the worker
(`CATEGORY_STALE_DAYS`), not per-pending.

Linking (`_link_pending_to_painpoint`) bumps `signal_count`, inserts a
`painpoint_sources` row, and refreshes the target category's
centroid via `update_category_embedding`.

Creating a new painpoint also stores its embedding in `painpoint_vec`
and refreshes its category's centroid.

### Embedder (`db/embeddings.py`)

- `OpenAIEmbedder` — `text-embedding-3-small`, 1536-dim, with
  exponential retry/backoff. Has both `embed(text)` and
  `embed_batch(texts, batch_size=256)`. Empty/whitespace input is
  sanitized to `" "` so the API never errors.
- `FakeEmbedder` — word-hash→seeded-Gaussian test double, no network.
  Empty input maps to `"__empty__"` so we never produce an all-zero
  vector that would collide with everything at cosine_sim=0.
- `get_painpoint_embedding(conn, pp_id)` — unpacks one painpoint's
  vec0 blob into a Python list (returns None on size mismatch).
- `find_best_category_ranked(conn, embedding, limit=50)` — like
  `find_best_category` but returns the full top-K ranked by cosine
  sim (no Uncategorized fallback, no bootstrap). Used by callers that
  compare across multiple candidates (e.g. reroute logic).

**OpenAI concurrency budget.** `OpenAIEmbedder._embed_with_retry`
acquires the global `OPENAI_API_SEMAPHORE` from `llm.py` around every
HTTP call. Same semaphore is acquired by `llm_call` in `llm.py`. So
embeddings + completions share one process-wide budget of
`OPENAI_CONCURRENCY=10` in-flight calls. Sized for tier-1 RPM limits;
adjust upward if you have higher tier or hit it as a bottleneck.

sqlite-vec virtual tables are created in `init_vec_tables`; vectors are
packed with `struct.pack("{n}f", …)`. Distance metric is cosine, so
`cosine_sim = 1 - distance`.

### Relevance (`db/relevance.py`)

`per_source_relevance = traction × recency` where:

- **Post-rooted** (no comment): `traction = log1p(score)·0.5 + log1p(num_comments)·0.8`
- **Comment-rooted** (has comment): `traction = log1p(comment_score)·0.5` —
  the post's popularity doesn't validate a random comment on it, so the
  comment carries the signal alone.
- `recency = 0.5 ** (age_days / 14)` (half-life 14 days).

Severity (the LLM's 1–10 claim) was removed from the formula: too
subjective to weight on. Severity is still stored on
`pending_painpoints.severity` for downstream display / filtering.

`per_source_relevance` is the only helper in `relevance.py`. The old
aggregation / cache helpers (`compute_pending_relevance`,
`compute_painpoint_relevance`, `cache_painpoint_relevance`,
`get_or_compute_painpoint_relevance`) and the `painpoints.relevance` /
`relevance_updated_at` columns were deleted — nothing in the live
pipeline consumed them after the promoter stopped filtering on
relevance and the delete pass switched to `CATEGORY_STALE_DAYS`.

---

## 7. Stage 4 — Category worker (`category_worker.py` + `db/category_events.py`)

Runs in a separate process — the live test invokes it once. One sweep
acquires `merge_lock` once, then runs four passes **in order**:

1. **Uncategorized → add_category_new**
   `propose_uncategorized_events` clusters the Uncategorized bucket via
   `cluster_painpoints` at `MERGE_COSINE_THRESHOLD=0.60` — the same
   threshold the promoter uses (the dual `SWEEP_CLUSTER_THRESHOLD` was
   collapsed; 0.05 below 0.60 was noise not signal). Clusters of size
   ≥ `MIN_SUB_CLUSTER_SIZE` (5) yield an `add_category_new` event. The
   apply function calls `namer.name_new_category(..., existing_taxonomy=flat)`
   so the LLM picks a parent from the current tree.

2. **Split crowded categories → add_category_split**
   `propose_split_events` skips anything smaller than
   `MIN_CATEGORY_SIZE_FOR_SPLIT=10` or that hasn't grown by
   `SPLIT_RECHECK_DELTA=10` since last check. For the rest it clusters,
   summarises the top 10 clusters, and asks the LLM via
   `namer.decide_split` → `SplitDecision(decision, reason, subcategories)`.
   The prompt deliberately tells the LLM to ignore cluster-size balance
   and use semantic judgement (big topics hijacking a category matter
   more than cluster sizes).

   **Parallel decide_split.** Multiple split candidates in one sweep
   used to be serial (one LLM call at a time). Now `propose_split_events`
   collects all candidates in phase 1 (sequential under the lock),
   fans out the `decide_split` LLM calls via `parallel_namer_calls`
   in phase 2, and yields events in phase 3. Reduces LLM wall-clock
   from O(N×3s) to O(N/5×3s) when N candidates trigger in one sweep.

3. **Delete dead → delete_category**
   `propose_delete_events` proposes delete for any category whose most
   recently updated member is older than `CATEGORY_STALE_DAYS=30`. This
   replaces the old mass-based threshold (`MIN_CATEGORY_RELEVANCE`), which
   was a size test dressed up as a quality test — small-but-valid niche
   categories were getting culled. The acceptance test re-checks
   staleness under the lock so a member added between propose and apply
   keeps the category alive.

4. **Merge duplicate siblings → merge_categories**
   `propose_merge_events` computes `inter_category_similarity` for every
   sibling pair under a common parent. Proposes merge if cosine ≥
   `MERGE_CATEGORY_THRESHOLD=0.80` (high — we only want near-duplicates;
   the threshold was raised from 0.65 after observing over-merging that
   undid legitimate splits). The pre-apply test rejects pairs where
   either category was already merged away earlier in the same sweep
   (cascade safety).

### Event model (`CategoryEvent`)

```
event_type:      add_category_new | add_category_split |
                 delete_category | merge_categories
payload:         dict (event-specific)
target_category: int (the category this acts on)
metric_name/value/threshold: audit fields
llm_result:      optional pre-fetched LLM response
```

`apply_with_test(conn, event, namer)` in `db/category_events.py` is the
uniform runner: run acceptance test → if pass, open a SAVEPOINT, call
`_apply_<event_type>`, log to `category_events` table. Every sweep
mutation ends up in that audit log (accepted or rejected, with reason).

### LLM prefetch

`prefetch_llm_batch(conn, events, namer, max_workers=5)` runs LLM
calls concurrently in a `ThreadPoolExecutor` before the serial
`apply_with_test` loop — turns N × 2–3 s serial calls into ~N/5
wall-clock. Results attach to `event.llm_result`; the apply function
uses the pre-fetched response or falls back to an inline call if
prefetch failed.

`parallel_namer_calls(call_specs, max_workers=5)` is the underlying
helper — takes `(key, callable)` pairs and returns `{key: result_or_None}`
with failures swallowed. Used by both `prefetch_llm_batch` (apply-phase
LLM calls bound to events) and `propose_split_events` (decide_split
calls per candidate). Process-wide concurrency is also bounded by
the OpenAI semaphore in `llm.py` regardless of how many pools exist.

### LLM naming (`db/llm_naming.py`)

- `LLMNamer` uses `gpt-4.1-mini` for everything
  (`_STRUCTURED_MODEL = "gpt-4.1-mini"`). Reasoning models (gpt-5-nano)
  were tried and rejected because they burn the token budget on
  reasoning and return `None` for structured output.
- `name_new_category(titles, descriptions, existing_taxonomy)` — returns
  `{"name", "description", "parent" or null}`.
- `decide_split(name, desc, total, clusters)` — returns `SplitDecision`
  (Pydantic).
- `describe_merged_category(survivor, loser, sample_titles)` — returns
  `{"description"}`, used to refresh the survivor's description post-merge.
- `FakeNamer` — test double with deterministic outputs.

---

## 8. Data model conventions

- **3 tables, not 2**: `pending_painpoints` is append-only raw; it is
  *never deleted by the promoter* — every pending becomes either a new
  merged painpoint or a link to an existing one. Merged rows live in
  `painpoints`, linked via `painpoint_sources(painpoint_id,
  pending_painpoint_id)`.
- **`pending_painpoint_all_sources`** is a VIEW unioning primary
  (`pending_painpoints.post_id`) with extras (`pending_painpoint_sources`).
  Any relevance / source-walking code reads through this view.
- **Uncategorized** is a sentinel, id-stable, never deleted. Everything
  lands here initially if no category exceeds
  `CATEGORY_COSINE_THRESHOLD=0.45`.
- **Category centroid**: mean of member painpoint embeddings
  (`update_category_embedding`). Implemented as a single JOIN against
  `painpoint_vec` (`pv.rowid = painpoints.id WHERE category_id = ?`)
  rather than the previous N+1 per-member SELECT — scales linearly
  with member count instead of round-tripping per member. Uncategorized
  is centroid-exempt (it's a dumping ground with no semantic coherence).
- **Bootstrap centroids** (`bootstrap_category_embeddings`): seeded
  taxonomy categories get an initial embedding from `name + description`
  before any painpoint lands in them. One `embed_batch` call covers
  every category in a single API round-trip — drops cold-start
  lock-hold from O(N×200ms) to O(300ms).
- **`merge_lock`** (`db/locks.py`) — coarse advisory lock held for
  every mutation path (promote + sweep). A single sqlite writer anyway,
  but the explicit lock makes the boundaries obvious and protects
  save-point semantics.

---

## 9. Thresholds (quick reference)

| Name                        | Where                | Value | Purpose                                     |
| --------------------------- | -------------------- | ----- | ------------------------------------------- |
| `MERGE_COSINE_THRESHOLD`    | `db/embeddings.py`   | 0.60  | Promote link + sweep clustering (single source of truth). |
| `CATEGORY_COSINE_THRESHOLD` | `db/embeddings.py`   | 0.45  | Below this → Uncategorized (was 0.30; raised above noise floor). |
| `RELEVANCE_HALF_LIFE_DAYS`  | `db/relevance.py`    | 14    | Recency decay.                              |
| `MIN_SUB_CLUSTER_SIZE`      | `db/category_events` | 5     | Min cluster size for a new category.        |
| `MIN_CATEGORY_SIZE_FOR_SPLIT` | `db/category_events`| 10   | Won't split categories smaller than this.   |
| `SPLIT_RECHECK_DELTA`       | `db/category_events` | 10    | Growth needed before re-asking LLM to split.|
| `MERGE_CATEGORY_THRESHOLD`  | `db/category_events` | 0.80  | Sibling-merge threshold (near-duplicates).  |
| `CATEGORY_STALE_DAYS`       | `db/category_events` | 30    | Category with no member activity in N days → propose delete. |
| `REROUTE_MARGIN`            | `db/category_events` | 0.08  | Step-5 reroute margin (in-progress; reroute fires only when alt cosine sim exceeds current by ≥ this). |
| `OPENAI_CONCURRENCY`        | `llm.py`             | 10    | Process-wide cap on in-flight OpenAI calls (embeddings + completions share). |
| `BATCH_TOKEN_BUDGET`        | `extractor.py`       | 2_000 | Approx tokens per LLM extraction batch.     |
| `LLM_CONCURRENCY`           | `extractor.py`       | 40    | Parallel extraction batches.                |

---

## 10. Known quirks & gotchas

- **Promoter drain limit.** `pick_unmerged_pending` defaults to
  `LIMIT 100`. The inline pipeline calls `run_once` once per subreddit;
  a very productive subreddit can leave orphan pendings behind. Loop
  `run_once` until `processed == 0` if you need a hard drain.
- **OpenAI concurrency ceiling.** Every API call (embeddings via
  `OpenAIEmbedder`, completions via `llm_call`) acquires the same
  `OPENAI_API_SEMAPHORE` in `llm.py` (default 10 in-flight). If you
  see embedding latency spike when the worker is also calling the LLM,
  it's the semaphore queueing — bump `OPENAI_CONCURRENCY` if your
  OpenAI tier supports more RPM/RPS.
- **Deep hijacking.** A single sweep does **not** fully rescue a mega
  category that's been semantically hijacked. The LLM splitter groups
  clusters but does not reroute individual outliers within the
  surviving bucket. Either iterate sweeps or add a per-painpoint
  re-routing pass.
- **Structured output + reasoning models.** `gpt-5-nano` returns
  `None` for structured output because reasoning tokens exhaust the
  budget. Use `gpt-4.1-mini` (the default) for anything going through
  `_call_structured`.
- **OpenAI json_object mode.** `_call` appends "Return your answer as
  a single JSON object" to the user message — the API errors if the
  literal word "json" isn't in the input.
- **Merge cascade.** `propose_merge_events` emits every candidate pair
  up front; earlier merges in the batch can delete one side of a later
  pair. `_test_merge_categories` guards against this.
- **Pytest output buffering.** The E2E test prints a lot; output only
  flushes when the test completes unless you explicitly pass
  `flush=True`. Don't kill a run assuming it hung — it's probably
  waiting on OpenAI.
- **FK on hallucinated post_ids.** The LLM sometimes emits a
  `post_id` that was never in the input batch.
  `save_pending_painpoints_batch` validates post existence before
  insert and skips silently with a warning.

---

## 11. Where things live

| Concern                          | Path                                    |
| -------------------------------- | --------------------------------------- |
| Pipeline orchestration           | `subreddit_pipeline.py`                 |
| Scraper                          | `reddit_scraper.py`                     |
| LLM extraction                   | `painpoint_extraction/extractor.py`     |
| Promoter loop                    | `promoter.py`                           |
| Promote core                     | `db/painpoints.py`                      |
| Embeddings + sqlite-vec          | `db/embeddings.py`                      |
| Relevance                        | `db/relevance.py`                       |
| Category worker driver           | `category_worker.py`                    |
| Category events / proposers /    |                                         |
| appliers / acceptance tests      | `db/category_events.py`                 |
| LLM-naming wrappers              | `db/llm_naming.py`                      |
| Sweep clustering helpers         | `db/category_clustering.py`             |
| Category CRUD helpers            | `db/categories.py`                      |
| Taxonomy seed                    | `taxonomy.yaml`, `db/seed.py`           |
| Schema                           | `db/schema.sql`, `db/__init__.py` (migrations) |
| Merge lock                       | `db/locks.py`                           |
| Live E2E test                    | `tests/test_e2e_real_subreddits.py`     |
| Hermetic pipeline test           | `tests/test_painpoint_pipeline.py`      |
| Design docs (tracked)            | `docs/PAINPOINT_INGEST_PLAN.md`, `docs/SIGNAL_SCORING_PLAN.md` |
| OpenAI shared utilities + semaphore | `llm.py` (`OPENAI_API_SEMAPHORE`, `llm_call`, `TokenCounter`) |

---

## 12. First steps for a new agent

1. Skim `subreddit_pipeline.analyze` — it's 70 lines and is the
   canonical "what does one run look like" reference.
2. Read `db/category_events.py` top-to-bottom — it's the most complex
   file and the one you'll touch most.
3. Run the hermetic tests: `pytest tests/test_painpoint_pipeline.py`.
   They use `FakeEmbedder`+`FakeNamer` so no API keys required.
4. If the task touches taxonomy behaviour, run
   `pytest tests/test_e2e_real_subreddits.py -v -s` and read the
   printed tree. Needs `OPENAI_API_KEY`, costs cents, takes minutes.
5. Refer to `docs/PAINPOINT_INGEST_PLAN.md` (§-numbers in code
   comments point into it) for the design intent behind the current
   structure.
