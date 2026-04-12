# Dataflow Overview

End-to-end pipeline: user picks a subreddit, we scrape Reddit, extract
painpoints with an LLM, deduplicate them, and evolve the category taxonomy
automatically.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER REQUEST                              │
│                     "Analyze r/ExperiencedDevs"                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. SCRAPE  (async httpx, semaphore-limited concurrency)            │
│                                                                    │
│  Phase 1: Fetch top-K posts across week/month/year  (3 reqs)      │
│  Phase 2: Dedup by reddit fullname → ~180 unique posts             │
│  Phase 3: Fetch comments for top N by engagement   (60 reqs)      │
│                                                                    │
│  ~63 requests total, Semaphore(10) + 429 retry                     │
│                                                                    │
│  Store every post  → posts table     (dedup by reddit_id)          │
│  Store every comment → comments table (dedup by reddit_id)         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. EXTRACT (LLM)                                                  │
│                                                                    │
│  Batch the scraped posts+comments and send to an LLM.              │
│  The prompt includes the full category taxonomy so the model can   │
│  assign each painpoint to an existing category.                    │
│                                                                    │
│  Input:  posts with their comments, category list                  │
│  Output: list of pending painpoints, each with:                    │
│          - title, description, severity (1-10)                     │
│          - quoted_text (verbatim evidence from the source)         │
│          - category_name (from the taxonomy)                       │
│          - source references (post_id, comment_id)                 │
│                                                                    │
│  Saved via db.painpoints.save_pending_painpoints_batch()           │
│  → pending_painpoints table                                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. PROMOTE  (promoter.py)                                         │
│                                                                    │
│  Drains pending_painpoints that haven't been linked to a merged    │
│  painpoint yet. For each pending painpoint:                        │
│                                                                    │
│  3a. Relevance gate                                                │
│      Compute relevance = traction × recency × severity.            │
│      If below MIN_RELEVANCE_TO_PROMOTE → drop the pending row.    │
│                                                                    │
│  3b. Dedup — Layer A (source overlap)                              │
│      Check if any existing painpoint shares the same Reddit        │
│      post/comment sources. If one match → link. If several →      │
│      merge those painpoints into one, then link.                   │
│                                                                    │
│  3c. Dedup — Layer B (text similarity)                             │
│      MinHash LSH (threshold 0.65) over title+description.          │
│      If match → link to best Jaccard candidate.                   │
│      If no match → create new painpoint in Uncategorized.         │
│                                                                    │
│  Result: pending painpoints are either dropped, linked to an       │
│  existing painpoint (bumping signal_count), or created as new.     │
│                                                                    │
│  painpoints table           (merged, canonical painpoints)         │
│  painpoint_sources table    (pending → painpoint junction)         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. CATEGORY SWEEP  (category_worker.py)                           │
│                                                                    │
│  Periodic taxonomy maintenance, four passes under merge lock:      │
│                                                                    │
│  4a. Uncategorized → assign painpoints to real categories          │
│      (LLM names new categories when needed)                        │
│  4b. Split crowded categories into subcategories                   │
│  4c. Delete dead categories (low relevance mass)                   │
│  4d. Merge duplicate sibling categories                            │
│                                                                    │
│  Every proposal goes through an acceptance test before mutation.   │
│  All events are logged to category_events for audit.               │
└────────────────────────────────────────────────────────────────────┘
```

## Database schema

Six core tables plus pipeline extensions:

```
posts ──────────────┐
  id, reddit_id,    │
  subreddit, title, │     pending_painpoints ──────────┐
  selftext, score,  │       id, post_id (FK),          │
  num_comments,     │       comment_id (FK),           │
  permalink,        ├──────►category_id (FK),           │
  created_utc,      │       title, description,        │
  signal_score      │       quoted_text, severity,     │
                    │       extracted_at               │
comments ───────────┘                                  │
  id, reddit_id,         pending_painpoint_sources     │
  post_id (FK),            (extra sources per pending   │
  body, score,              beyond the primary one)     │
  permalink,                                           │
  created_utc                                          │
                                                       │
                         painpoint_sources ◄────────────┘
categories                 painpoint_id (FK) ──► painpoints
  id, name,                pending_painpoint_id    id, title,
  parent_id (FK),                                  description,
  description              ┌───────────────────── category_id (FK),
  last_split_check_at      │                       severity,
                           │                       signal_count,
                           │                       relevance,
category_events ◄──────────┘                       first_seen,
  event_type, payload,                             last_updated
  metric, threshold,
  accepted, reason
```

## Step 1 — Scrape in detail

Entry point: `scrape_subreddit_full(subreddit)` in `reddit_scraper.py`.
Fully async (httpx + asyncio) with `Semaphore(10)` for concurrency
control. Reddit enforces 60 req/min server-side; the scraper handles
429 responses with a sleep + retry.

### Time windows

Three parallel listing fetches capture both recent and chronic pain:

| Sort   | Time filter | Purpose                          |
|--------|-------------|----------------------------------|
| `top`  | `week`      | What's hot right now             |
| `top`  | `month`     | Sustained recent problems        |
| `top`  | `year`      | Chronic, long-standing issues    |

### Dedup and comment budget

Posts across the three windows overlap heavily (week's top is a subset
of month's top, which overlaps with year's top). After fetching, posts
are deduplicated by Reddit fullname (`name` field) and ranked by
engagement (`score + num_comments`).

Comments are the expensive part (1 request per post, no batch API).
Only the top `posts_with_comments` posts by engagement get their comments
fetched. An optional `min_score` threshold further filters eligibility.

### Request budget (defaults)

```
Listings:  3 requests   (1 per window, up to 100 posts each)
Dedup:     ~180 unique posts from ~300 raw
Comments:  60 requests  (top 60 by engagement)
─────────────────────────
Total:     63 requests
Concurrency: Semaphore(10) + 429 retry with backoff
```

### Persistence

All posts and comments are persisted via `db.posts.upsert_post` and
`db.posts.upsert_comment` (dedup on `reddit_id`). The internal `post.id`
and `comment.id` are used as foreign keys throughout the pipeline.

### API surface

`reddit_scraper.py` provides:
- `scrape_subreddit_full(sub, ...)` — high-level: 3-window fetch, dedup,
  comment budget (the primary entry point)
- `scrape_subreddit(client, sem, rate, sub, ...)` — single listing fetch
- `scrape_comments(client, sem, rate, permalink, ...)` — top comments
- `search_reddit(client, sem, rate, query, ...)` — search endpoint
- `_dedup_and_rank(batches)` — merge + sort by engagement

## Batching strategy for LLM extraction

The scraped data must be chunked to fit within the LLM context window.
Target model: **GPT-5.4-nano** (400K context, 128K max output,
$0.20/M input tokens, $1.25/M output tokens).

### Payload size estimates

Using ~4 chars/token for English text with Reddit formatting.

| Component         | Count (defaults)        | Avg chars/item | Worst chars/item |
|-------------------|-------------------------|----------------|------------------|
| Posts              | ~200 (after dedup)      | 1,300          | 10,300           |
| Comments           | ≤1,500 (60 posts × 25) | 550            | 5,150            |

| Scenario     | Total chars | Total tokens | Fits in one 350K-usable call? |
|-------------|-------------|--------------|-------------------------------|
| **Average**  | ~1.09M      | ~271K        | Yes                           |
| **Worst**    | ~9.79M      | ~2.45M       | No (7 batches)                |

### Long-context recall degradation

GPT-5.4-nano's 400K context window does not guarantee uniform attention
across the full input. Benchmarks show meaningful recall loss at high
token counts:

- **MRCR v2 (multi-needle retrieval) at 64K-128K: 44.2%** — the model
  misses scattered details in large contexts.
- **LCR (long-context reasoning): 66%** — decent but not strong.
- A 2026 study on the GPT-5 family found accuracy dropped to 50-53% on
  exhaustive comprehension tasks at 70K+ tokens, while precision stayed
  high (~95%). The model becomes more selective: what it finds is
  accurate, but it skips more items ("lost in the middle" effect).

Pain-point extraction is an **exhaustive sweep** — we need every post
and comment examined, not just the ones the model happens to attend to.
At ~271K tokens (our average payload), the model uses 68% of its
context window, and estimated recall drops to ~70-80%. Splitting into
three ~78K-token batches keeps each call well within the reliable zone
(≤100K tokens) and brings estimated recall to ~90-95%, at negligible
extra cost (~$0.01).

| Strategy          | Tokens/batch | Est. recall | Batches | Cost   |
|-------------------|-------------|-------------|---------|--------|
| Single call       | ~271K       | ~70-80%     | 1       | ~$0.07 |
| Time-window split | ~78K each   | ~90-95%     | 3       | ~$0.08 |

**Recommendation:** default to the time-window split for extraction
quality; only use a single call when speed matters more than coverage
(e.g. interactive preview).

### Batching approach

1. **Time-window split (default)** — split posts by `created_utc` into
   the original time windows (week / month / year). Each window has at
   most ~100 posts + comments, yielding ~78K tokens on average —
   comfortably within nano's reliable recall zone. Posts that appear in
   multiple windows are assigned to the earliest.

2. **Single call (fast path)** — if total tokens ≤ 100K, send everything
   in one request. Useful for small subreddits or interactive previews
   where speed matters more than exhaustive coverage.

3. **Engagement-rank split (extreme fallback)** — if a single window
   still exceeds ~100K tokens (e.g. a subreddit with very long posts),
   split within the window by engagement rank (top half / bottom half).

This gives at most 3 batches in the normal case and 6 in the extreme
case, with each batch carrying temporally coherent posts — which helps
the LLM detect patterns like "multiple people hit X this week".

Cross-batch pattern detection is handled downstream by the **promoter**,
which merges `pending_painpoints` by source overlap and text similarity
regardless of which batch produced them.

### Cost per subreddit (GPT-5.4-nano)

| Scenario | Batches | Input cost | Output cost* | Total   |
|----------|---------|------------|--------------|---------|
| Average  | 3       | $0.055     | $0.038       | ~$0.09  |
| Worst    | 7       | $0.490     | $0.088       | ~$0.58  |

\* Output estimate: ~50 painpoints × ~200 tokens each per batch.

### In-memory footprint

| Scenario | Posts + comments in RAM |
|----------|------------------------|
| Average  | ~2.2 MB                |
| Worst    | ~10.9 MB               |

Memory is not a constraint for any practical deployment.

## Step 2 — LLM extraction in detail

Posts and comments are formatted into batches and sent to an LLM with a
system prompt containing:

1. The full category taxonomy (from `db.categories.get_category_list_flat`)
2. Instructions to extract painpoints with evidence

For each painpoint the LLM must return:
- **title** — concise name for the pain (e.g. "Cursor AI suggests wrong imports")
- **description** — 1-2 sentence explanation
- **severity** — 1 (minor annoyance) to 10 (blocking/critical)
- **quoted_text** — verbatim quote from the post or comment
- **category_name** — must match an existing category from the taxonomy
- **post_id** — which post this was extracted from
- **comment_id** — which comment (null if extracted from the post body)

The LLM may extract zero, one, or many painpoints from a single post.
Different comments on the same post may yield different painpoints.

Results are saved via `db.painpoints.save_pending_painpoints_batch()`.

## Step 3 — Promote in detail

The promoter (`promoter.py`) is a loop that drains unmerged pending
painpoints. Each one passes through a three-stage funnel:

```
pending_painpoint
       │
       ▼
 ┌─ relevance ≥ 0.5? ──── no ──► DROP (delete row)
 │     yes
 │      │
 │      ▼
 ├─ Layer A: same sources? ──── 1 match ──► LINK to existing painpoint
 │      │                  ──── N matches ► MERGE N → 1, then LINK
 │      │ 0 matches
 │      ▼
 └─ Layer B: MinHash ≥ 0.65? ── match ──► LINK to best Jaccard hit
                             ── no match ► CREATE new in Uncategorized
```

**Relevance** = max over sources of `traction × recency × severity_mult`
- traction: `log1p(score) × 0.5 + log1p(num_comments) × 0.8` (or `signal_score` if computed)
- recency: exponential decay with 14-day half-life
- severity_mult: `0.5 + 0.1 × severity` (range 0.6 – 1.5)

**Layer A** (SQL): check if any existing merged painpoint already cites
the same `(post_id, comment_id)` source. Fast, exact, zero false positives.

**Layer B** (MinHash LSH): 128-perm MinHash over title+description shingles.
Query the LSH index at Jaccard threshold 0.65. Picks the single best
match by exact Jaccard when multiple candidates return.

Writes go to `painpoints` (signal_count bumped) and `painpoint_sources`
(junction linking merged ↔ pending). All writes happen under a
`BEGIN IMMEDIATE` merge lock.

## Step 4 — Category sweep in detail

The category worker runs periodically (not per-painpoint). It acquires
the same merge lock as the promoter, then runs four passes:

1. **Uncategorized pass** — painpoints sitting in the Uncategorized
   bucket are clustered (MinHash at threshold 0.40) and assigned to
   existing or LLM-named new categories.

2. **Split pass** — categories with too many painpoints are split into
   subcategories. The LLM names the new subcategories. A trigger-
   discipline mechanism prevents re-splitting too soon.

3. **Delete pass** — categories whose total relevance mass falls below
   `MIN_CATEGORY_RELEVANCE` are deleted (painpoints move to
   Uncategorized for reassignment on the next sweep).

4. **Merge pass** — sibling categories with high inter-category MinHash
   similarity (threshold from `MERGE_CATEGORY_THRESHOLD`) are merged.

Every event is logged to `category_events` with the metric that triggered
it, the threshold it was compared against, and whether it was accepted.

## Module responsibility map

| Module               | Role                                          |
|----------------------|-----------------------------------------------|
| `reddit_scraper.py`  | Async Reddit OAuth client (httpx), rate limiter, scrape_subreddit_full |
| `db/posts.py`        | Persist posts and comments                     |
| `db/painpoints.py`   | Pending painpoint CRUD + promote pipeline      |
| `db/relevance.py`    | Relevance scoring (traction × recency × sev)  |
| `db/similarity.py`   | MinHash LSH index for text dedup               |
| `db/categories.py`   | Category lookups                               |
| `db/category_events.py` | Category mutation proposals + acceptance    |
| `db/category_clustering.py` | Intra/inter-category clustering          |
| `db/llm_naming.py`   | LLM-powered category name generation           |
| `db/locks.py`        | `BEGIN IMMEDIATE` merge lock                   |
| `db/queries.py`      | Read-only queries for consumers                |
| `promoter.py`        | Pending → merged painpoint loop               |
| `category_worker.py` | Periodic taxonomy sweep                        |
| `llm.py`             | Shared OpenAI helpers (llm_call, web_search)   |
| `ingest.py`          | Pipeline orchestrator (TO BE REWRITTEN)        |
| `main.py`            | CLI entry point (TO BE REWRITTEN)              |
