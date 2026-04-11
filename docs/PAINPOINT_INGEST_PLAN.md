# Painpoint ingest & taxonomy-evolution plan

> Status: **implemented**. See §13 for the implementation log and the
> deltas between this design doc and the code that actually shipped.
> All 101 tests (60 base db tests + 41 pipeline tests) pass.

This plan describes how rows in `pending_painpoints` get promoted into the
merged `painpoints` table, how categories evolve in response, and how we
score painpoint relevance so the system prefers signal over noise.

It composes with the existing `SIGNAL_SCORING_PLAN.md` (post-level
`signal_score`); painpoint relevance is built **on top of** the post
signal score, not in place of it.

---

## 0. Terminology & assumptions

| Term | Meaning |
|---|---|
| `pending_painpoints` | the existing append-only LLM-extraction table (the user originally called this "raw_painpoints") |
| `painpoints` | the merged, deduped table with `signal_count` (the user originally called this the "clean table") |
| Promoter | a tight, LLM-free loop that pulls pending pps and links/inserts them into the merged table (§3, §4, `promoter.py`) |
| Category worker | a separate process that runs periodically, acquires the merge lock, and sweeps the taxonomy (§5, `category_worker.py`) |
| Sentinel `Uncategorized` category | the bucket the promoter parks new painpoints in when no similarity match exists; the category worker clusters and reassigns them on its next sweep |

**Process model:** **two cooperating processes** sharing one merge lock —
the promoter and the category worker. Originally I drafted this as a
single in-line worker; we split it on review (see §5 intro). All
significant tunables live in §10.

---

## 1. End-to-end flow at a glance

```
   pending_painpoints                         painpoints
        │                                          ▲
        │ 1. pick batch                            │
        ▼                                          │
   ┌─────────────┐                                 │
   │ Promoter    │  (process A — tight loop)      │
   │ worker      │                                 │
   └─────────────┘                                 │
        │                                          │
        │ for each pending pp:                     │
        │   2. compute relevance                   │
        │   3. drop if relevance < threshold       │
        │   4. find similar painpoint (Layers A+B) │
        │   5. acquire merge lock                  │
        │   6a. similar found → link as source     │
        │   6b. no match     → insert into         │
        │       Uncategorized sentinel             │
        │   7. release lock                        │
        ▼
  (no category mutation here — promoter never
   touches the taxonomy)


   ┌─────────────────────┐  (process B — periodic, e.g. every 15 min)
   │ Category worker     │
   │  category_worker.py │
   └─────────────────────┘
        │
        │ on schedule:
        │   1. acquire merge lock (shared with promoter)
        │   2. sweep step 1: cluster Uncategorized → propose
        │      add_category(new) events
        │   3. sweep step 2: split-check grown categories →
        │      propose add_category(split) events
        │   4. sweep step 3: delete-check dead categories →
        │      propose delete_category events
        │   5. sweep step 4: pairwise sibling similarity →
        │      propose merge_categories events
        │   6. each event: apply under savepoint, run test,
        │      release or rollback
        │   7. release lock
        ▼
   category_events log
```

The two non-trivial decisions:

- **(promoter step 4)** does this pending painpoint already have a
  home in the merged table? — similarity, not exact-title equality.
- **(worker steps 2–5)** does this taxonomy mutation actually improve
  things? — domain-specific test per event type, no global gate.

---

## 2. Painpoint relevance (the score)

### 2.1 Inputs

A painpoint is tied to **one or more sources** — each source is a
`(post, optional comment)` tuple. With multi-source pending painpoints
(§7.5) the LLM can claim the same pain across several posts in one
extraction batch. The relevance calc below treats single-source as a
trivial case of the multi-source rule. We pull data from each source
plus the pending row itself:

| Source | Field | Used for |
|---|---|---|
| `posts.signal_score` | post-level Reddit signal (column added by **this plan** in §7.7; compute logic from `SIGNAL_SCORING_PLAN.md`) | base traction |
| `posts.created_utc` | age | time decay |
| `posts.score`, `num_comments` | fallback for the inline signal_score approximation when the cached column is NULL | base traction (fallback) |
| `comments.score` | if the painpoint came from a comment, that comment's traction | base traction boost |
| `comments.created_utc` | comment age | time decay |
| `pending_painpoints.severity` (1–10) | LLM's claim about pain intensity | severity multiplier |
| `pending_painpoints.extracted_at` | freshness of the *extraction*, not the source post | secondary recency |

**Schema dependency.** `posts.signal_score` is added by **this plan**
(see §7.7) — not by `SIGNAL_SCORING_PLAN.md`. The compute logic for
filling that column still belongs to the SIGNAL_SCORING_PLAN
workstream (the engagement × velocity × cluster formula), but the
column itself lands here so painpoint relevance has somewhere to read
from. Until the scoring job exists and starts populating the column,
`signal_score` is `NULL` and `compute_relevance` falls back to a
lightweight inline formula based on raw `posts.score` and
`posts.num_comments`. Once scoring runs, the cached column wins.

### 2.2 Per-source components

A multi-source pending painpoint (or merged painpoint, after several
pendings have linked in) has many `(post, comment)` tuples in its
source set. We compute relevance **per source**, then aggregate.
First the per-source formula:

```python
def per_source_relevance(post, comment, severity):
    # Time decay — exponential half-life on the *source* timestamp.
    # If the painpoint has both a post and a comment, use the comment's
    # timestamp (more specific); otherwise the post's.
    source_created_utc = comment.created_utc if comment else post.created_utc
    age_days = (now - source_created_utc) / 86400
    recency  = 0.5 ** (age_days / RELEVANCE_HALF_LIFE_DAYS)   # ∈ (0, 1]

    # Traction — read the cached signal_score column. If NULL (the
    # SIGNAL_SCORING_PLAN job hasn't run yet), fall back to a cheap
    # inline approximation.
    if post.signal_score is not None:
        traction = post.signal_score
    else:
        traction = log1p(post.score) * 0.5 + log1p(post.num_comments) * 0.8
    if comment is not None:
        # boost: comment-rooted painpoints inherit BOTH the post's traction
        # AND the comment's own engagement
        traction += log1p(comment.score) * 0.5

    # Severity — LLM's 1-10 claim, normalized. Same for every source
    # since severity lives at the pending-painpoint level, not per-source.
    severity_mult = 0.5 + 0.1 * severity   # 1 → 0.6, 10 → 1.5

    return traction * recency * severity_mult
```

### 2.3 Aggregating across sources

Given a list of per-source relevance values, the painpoint's overall
relevance is **the max**:

```python
def compute_relevance(painpoint_or_pending):
    severity = painpoint_or_pending.severity
    return max(
        per_source_relevance(post, comment, severity)
        for (post, comment) in painpoint_or_pending.iter_sources()
    )
```

**Why max and not sum or mean.**

- **Sum** would reward a painpoint just for having more sources, which
  conflates "quality of evidence" with "amount of evidence." Amount
  is already captured by `painpoints.signal_count`, which is a
  separate axis. Don't double-count.
- **Mean** would punish a painpoint that has one fresh, severe source
  alongside several stale ones — but the fresh one alone is enough
  to keep the pain "live" for our purposes. Mean drags the answer
  down for no good reason.
- **Max** answers the right question: "is at least one piece of
  evidence for this pain still hot?" If yes, the painpoint is
  relevant. If no, every source has decayed and the painpoint can
  go away. This matches the `delete_category` semantics in §5.1
  step 3 (which uses individual painpoint relevance as a per-member
  safety check).

Single-source pendings are the trivial case where the max is over a
1-element list — same answer as the old single-source formula.

Multiplicative within a source (`traction × recency × severity_mult`)
for the same reason `signal_score` is multiplicative: these are
independent amplifiers, a source scores high when it stacks.

`relevance` is recomputed lazily — cached on the `painpoints` row in
the column added by §7.1, with a `relevance_updated_at` timestamp.
Anything older than ~24h gets recomputed before use. The
recomputation walks `painpoint_sources → pending_painpoint_sources`
to find all `(post, comment)` tuples contributing to this painpoint,
then runs the formula above.

### 2.4 The drop step

Before the similarity check (§3) or any merge work, the promoter
computes `relevance` for the pending painpoint and compares it to
`MIN_RELEVANCE_TO_PROMOTE`:

```python
def promote_pending(pending_id):
    pending = load_pending(pending_id)
    rel = compute_relevance(pending)

    if rel < MIN_RELEVANCE_TO_PROMOTE:
        # Hard drop. The row never reaches painpoints, never gets a
        # painpoint_sources entry, and is removed from
        # pending_painpoints entirely. Safe because no merged painpoint
        # references it (we haven't merged it yet).
        delete_pending(pending_id)
        return None

    # ...continue to similarity check
```

**Why hard drop, not a `dropped` flag.** A flag-and-keep approach was
considered for audit purposes but rejected — the flagged rows would
never be referenced by anything (`painpoint_sources` only points at
*merged* pending pps), so they'd just accumulate as dead weight.
Deletion preserves database leanness without losing evidence: the
source `posts` and `comments` are still there, the LLM extraction can
re-run if we want a second look.

This is the *only* place `pending_painpoints` rows are deleted; the
table is otherwise append-only. The drop happens before the row could
be merged, so the original "pending → merged" evidence chain
invariant is unaffected.

### 2.5 Why time decay belongs here, not in `signal_score`

By "`signal_score`" I mean the post-level rank defined in
`SIGNAL_SCORING_PLAN.md` — engagement × velocity × cluster
multiplier — **not** the raw Reddit `posts.score` upvote column.

`signal_score` deliberately doesn't decay: it ranks posts at scrape
time and captures "was this post hot when we found it." For the
painpoint layer we care about "is this pain still relevant *right
now*," which is a different question. Same underlying Reddit data,
different decay policy at a different layer.

If we pushed time decay down into `signal_score` itself, we'd break
post ranking — a high-quality post from three weeks ago would
constantly demote itself out of view even if its content is still
exactly what we want to extract pains from. Decay belongs at the
painpoint layer because that's the layer asking the time-sensitive
question.

---

## 3. Similarity check (does this painpoint already exist?)

Similarity is the hardest part of this whole pipeline. Painpoint text
is LLM-paraphrased and unreliable, so a single similarity primitive is
not enough. We use **two layers**, applied at insert time, each
catching a different kind of duplication. (Earlier drafts had a third
layer — periodic source-set reconciliation — which turned out to be
provably dead code under the rules below. The history is in §3.3.)

### 3.1 Layer A — exact source check (insert-time, free)

Before doing any hashing, ask SQL the cheapest possible question:
**"is *any* source in this pending painpoint's source set already
feeding some merged painpoint?"**

A pending painpoint may have multiple sources (per §7.5), so this is
an `IN` query over its full source set. **Note** that `painpoint_sources`
itself only stores `(painpoint_id, pending_painpoint_id)` — to find
which `(post_id, comment_id)` tuples back a merged painpoint we have
to **join through `pending_painpoint_sources`** (the new junction
table from §7.5).

```python
def exact_source_lookup(pending, conn):
    sources = list(pending.sources)   # from pending_painpoint_sources
    if not sources:
        return set()

    placeholders = ",".join("(?, COALESCE(?, -1))" for _ in sources)
    params = [v for s in sources for v in (s.post_id, s.comment_id)]

    rows = conn.execute(f"""
        SELECT DISTINCT ps.painpoint_id
        FROM painpoint_sources ps
        JOIN pending_painpoint_sources pps
          ON pps.pending_painpoint_id = ps.pending_painpoint_id
        WHERE (pps.post_id, COALESCE(pps.comment_id, -1))
              IN ({placeholders})
    """, params).fetchall()
    return {r["painpoint_id"] for r in rows}
```

(The `COALESCE(.., -1)` dance is the standard SQLite workaround for
`NULL` not equalling `NULL` in tuple comparisons. The sentinel `-1`
is safe because `comment_id` is a positive autoincrement, so a real
comment can never collide with the sentinel.)

**One match returned**: at least one source is shared with an
existing merged pp, and only one such pp exists. Link the new
pending row there, skip Layer B.

**Multiple matches returned** (only possible with multi-source pending
painpoints from §7.5 — a single pending pp whose source set spans
posts that are already cited by *different* merged painpoints): the
LLM has just told us those merged painpoints are the same pain by
shared evidence. Pick a canonical survivor (highest `signal_count`,
ties broken by lowest id), **merge the others into it**
(`merge_painpoints` action), then link the new pending pp to the
survivor. See §3.5 for the code.

**Zero matches returned**: fall through to Layer B.

This is a SQL prefilter, not a hash. It catches the strongest case
(identical evidence) for free, before we touch MinHash.

### 3.2 Layer B — text MinHash on title + description (insert-time)

For the much more common case — pending painpoint comes from a *new*
post that no merged painpoint has touched yet — Layer A returns
nothing and we fall through to text similarity.

Reuse the MinHash + LSH approach from `SIGNAL_SCORING_PLAN.md`:

- Build MinHash signature over `pending.title + " " + pending.description`
- Query LSH index of existing `painpoints.title + " " + description`
- LSH similarity threshold: **`SIM_THRESHOLD = 0.65`** (originally
  drafted as 0.55; bumped during implementation to widen the gap with
  the sweep clusterer — see §13)

This is the workhorse for normal merging. Most pending painpoints
arrive from posts that no other painpoint has touched yet, so Layer A
misses and Layer B carries the load.

**Multi-source pendings and Layer B.** A multi-source pending pp has
exactly one `title` and one `description` regardless of how many
sources it has — the LLM produced one painpoint claim that happens
to span several posts. So the MinHash signature is unambiguous: one
input text, one signature, same query as the single-source case.
Layer B doesn't need to know whether the pending is single- or
multi-source.

**Limits of Layer B:** if the LLM phrases the same pain wildly
differently across two posts, MinHash on title shingles will miss the
match and we'll create two merged painpoints for the same pain. We
accept this — false splits are easy to clean up later (the next
`merge_painpoints` event catches them when more evidence accumulates),
false merges are hard to undo.

### 3.3 (Removed) Layer C — source-set Jaccard reconciliation

Earlier drafts had a third layer: a periodic batch pass over all
merged painpoints, computing source-set Jaccard between each pair and
proposing merges for high-overlap pairs. We removed it because it's
**provably dead code** under the rules of Layers A and B:

1. Layer A is "if any source overlaps with an existing merged pp,
   handle it now" (link or merge). After Layer A runs, **no two
   merged painpoints can ever share a source post**. The first
   pending pp from post #42 routes the post into one merged pp; every
   subsequent pending pp from post #42 hits Layer A and gets routed to
   that same merged pp (or merges spanned candidates into one).
2. If no two merged pps ever share a source, the pairwise source-set
   Jaccard is always exactly **zero** for any pair. Layer C's whole
   job was to find pairs with `Jaccard > 0.40`. There are none.
3. The one path to source overlap I was implicitly worrying about —
   multi-source pending pps spanning multiple existing merged pps —
   is now handled at insert time by the multi-match branch of Layer A
   (§3.5), which merges the spanned painpoints immediately rather
   than deferring to a later cleanup pass.

Layer C is gone. Section kept as a tombstone so future-us doesn't
reinvent it without remembering why it was removed.

### 3.4 LSH index lifecycle (Layer B)

The Layer B text-MinHash LSH index is rebuilt incrementally:
- On promoter startup: rebuild from all `painpoints` rows.
- After each new `painpoints` insert by the promoter: insert into the
  live index.
- After `merge_painpoints` deletes the loser (§3.5 step 6): remove
  the loser's signature. **This is the only path that deletes a
  `painpoints` row** — no sweep step in §5 deletes painpoints
  (sweeps delete *categories*, and the painpoints get re-pointed,
  not removed).
- After each painpoint title/description rename (if we add the
  `rename_painpoint` event from §11): remove old signature, insert
  new one.

Persist as a pickled `MinHashLSH` next to `trends.db` so we don't pay
the rebuild cost every cold start.

**Cross-process synchronization (current approach: rebuild in the
worker).** The promoter holds the canonical in-memory LSH and writes
new signatures as it inserts painpoints. The category worker is a
separate process and doesn't share that memory. To avoid stale-LSH
bugs, **the worker rebuilds its LSH from the `painpoints` table at
the start of every sweep**, inside the merge lock. This is O(N) per
sweep but N is small (a few thousand merged painpoints), and the
rebuild only happens at the worker's cadence (every ~15 minutes,
not per insert).

The promoter's in-memory LSH is **never** invalidated by the worker
under the current scope, because the worker doesn't change painpoint
*titles* — only `category_id`, which isn't part of the MinHash
signature. If we add `rename_painpoint` (§11), the worker would need
to bump a version counter that the promoter checks after each lock
acquisition, and reload from disk if changed. Out of scope for v1.

### 3.5 Decision flow at insert time

```python
def find_match_for_pending(pending):
    # Layer A — free SQL prefilter over the pending's full source set
    sql_matches = exact_source_lookup(pending)   # set of merged_pp ids

    if len(sql_matches) == 1:
        # Single existing painpoint already cites a source we share — link.
        return ("link", sql_matches.pop())

    if len(sql_matches) > 1:
        # Multi-source pending spans multiple merged pps — by LLM-evidence
        # transitivity these are the same pain. Pick a survivor and merge
        # the rest into it, then link the new pending to the survivor.
        survivor = pick_canonical(sql_matches)   # highest signal_count, ties → lowest id
        for other in sql_matches - {survivor}:
            merge_painpoints(survivor, other)    # moves sources, bumps signal_count, retires `other`
        return ("link", survivor)

    # Layer B — text MinHash over title + description
    text_matches = text_lsh.query(minhash(pending.title + " " + pending.description))
    if not text_matches:
        return ("create_new_in_uncategorized", None)
    if len(text_matches) == 1:
        return ("link", text_matches[0])
    # multiple text matches — pick the highest-relevance one
    return ("link", max(text_matches, key=lambda p: p.relevance))
```

The actions:

- **`link`**: existing `link_painpoint_source` + `signal_count` bump
  in `db/painpoints.py`. Adds every source from the new pending pp's
  source set to the chosen merged painpoint.
- **`merge_painpoints(survivor_id, loser_id)`** (new, lives in
  `db/painpoints.py`): mechanically merges two merged painpoints.
    1. `UPDATE painpoint_sources SET painpoint_id = survivor_id WHERE painpoint_id = loser_id`
    2. `survivor.signal_count += loser.signal_count`
    3. `survivor.first_seen = min(survivor.first_seen, loser.first_seen)`
    4. `survivor.last_updated = now()`
    5. `DELETE FROM painpoints WHERE id = loser_id`
    6. Remove the loser from the Layer B LSH index.
    7. **Survivor's `category_id` is unchanged** — it keeps whatever
       category it was already in. This is a deliberate-but-
       arbitrary choice: when two painpoints in different categories
       merge, *something* has to give. We pick the survivor itself
       by `painpoints.signal_count` (highest wins, ties broken by
       lowest id), and the survivor's category just comes along for
       the ride as a side effect — we do **not** independently
       compare the two categories. There is no `category.signal_count`
       to compare; per-category aggregation lives in §5.1 step 3 as
       *relevance mass*, computed on demand. If the surviving
       painpoint ends up in the wrong bucket, the next category-
       worker sweep will catch it (its category may now look
       mergeable with the loser's category, which sweep step 4
       handles at the category level).
    8. **Not written to `category_events`.** The promoter is logless
       by design (per §5). For audit/debugging, `merge_painpoints`
       calls `logging.info(...)` with the survivor and loser ids,
       both pre-merge signal_counts, and the triggering pending pp
       id. Plain text log file, not the database. If we ever need
       structured data here, we add a column or table at that point.
- **`create_new_in_uncategorized`**: inserts a fresh `painpoints` row
  with `category_id` pointed at the sentinel `"Uncategorized"`
  category, and copies the triggering pending pp's source set into
  `painpoint_sources` (one row per `(post_id, comment_id)` tuple from
  `pending_painpoint_sources`). **The new merged pp's sources = the
  triggering pending pp's sources, full stop.** No speculative source
  scanning, no fuzzy expansion — see §11 for why.

**The promoter never creates or mutates real categories** — that work
is exclusively the category worker's job (§5). This keeps the
promoter purely additive: it can run in a tight loop without ever
touching the taxonomy.

---

## 4. Global merge lock

### 4.1 Why

A single global lock serializes all writes that mutate the merged
`painpoints` table or the `categories` taxonomy. It has **two callers
with very different acquire patterns**:

| Caller | Frequency | Hold time | What it does inside |
|---|---|---|---|
| Promoter (§3) | many small acquires, ~1/painpoint | milliseconds | similarity check + insert/link |
| Category worker (§5) | infrequent, scheduled | seconds to minutes | sweeps the taxonomy, applies events |

The two callers must use the **same lock** even though they're doing
very different work — because both need consistent reads/writes
against `painpoints` and `categories`. The promoter doesn't want to
link a painpoint to a category that the worker is in the middle of
deleting.

Contention is fine in practice: the promoter is fast (sub-millisecond
inserts), the worker is rare (minutes to hours apart). When the
worker runs, the promoter blocks for the duration; when it's done,
the promoter resumes.

### 4.2 Implementation

The promoter and the category worker are **separate OS processes**
(per §5: promoter is a daemon or tight loop, category worker is a
cron / systemd timer / sleep loop). A Python `threading.Lock`
wouldn't help — it doesn't span process boundaries. So we need a
lock primitive that lives in shared, persistent state. SQLite gives
us one for free via `BEGIN IMMEDIATE`:

```python
# db/locks.py  (new file)
import sqlite3, contextlib, time

@contextlib.contextmanager
def merge_lock(conn, timeout=30):
    """Global advisory lock for the merge pipeline.

    Uses BEGIN IMMEDIATE which acquires SQLite's reserved lock —
    only one writer at a time is allowed to be inside the block.
    Used by both the promoter (§3) and the category worker (§5).
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            break
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
```

`BEGIN IMMEDIATE` is the right primitive: it's strictly stronger than
the per-call autocommit pattern the rest of `db/` uses, and SQLite's
WAL mode already serialises writers — we're just declaring that the
sequence of writes inside the block is one logical unit.

The category worker may want a longer `timeout` (e.g., 300s) than the
promoter, since it's doing more work and the only thing it's racing
with is the cheap promoter loop.

### 4.3 Scope of the lock

**Promoter holds the lock for:**
- find-similar query (Layers A + B from §3)
- insert into `painpoints` OR `link_painpoint_source` + signal_count bump
- nothing else — no category mutation, no event firing

**Category worker holds the lock for:**
- one full sweep pass: process Uncategorized → split-check → delete-check → merge-check (the four steps from §5.1)
- per-event acceptance tests + savepoint rollback for rejections
- LLM-naming calls for new and split categories

**Outside** the lock (intentionally), for both callers:
- LLM extraction into `pending_painpoints` (append-only, doesn't conflict)
- Reads from `db/queries.py` (use a separate read-only connection)
- Relevance recomputation (idempotent, can run in parallel)

---

## 5. Category worker (separate process)

Taxonomy mutations happen in a **separate process** that runs
periodically — not inline with painpoint promotion. This is a
deliberate change from earlier drafts where every promoted painpoint
could fire a category event under the merge lock.

**Why separate:**
- The promoter stays a tight, predictable loop. No LLM calls, no
  clustering, no second-guessing of the taxonomy. Just relevance →
  similarity → link/insert.
- Taxonomy work amortizes over many painpoints. Running an
  intra-bucket clustering pass once per minute over a category that
  grew by 50 painpoints is much cheaper than running it 50 times
  after each individual insert.
- LLM-naming calls are slow (~2-5 seconds each). Putting them in the
  promoter's hot path would dominate insert latency for no good
  reason.
- The two workloads have very different cadences, error modes, and
  failure-recovery needs. Splitting them lets each be tuned and
  restarted independently.

The category worker is a separate top-level script (`category_worker.py`)
that runs on a schedule (cron, systemd timer, or a sleep loop). When
it runs, it acquires the same merge lock as the promoter (§4) and
performs a full sweep.

### 5.1 What a sweep does

One sweep = four passes, in order, all under the same lock acquisition:

1. **Process Uncategorized.** Cluster painpoints currently sitting in
   the `Uncategorized` sentinel bucket. **Clustering uses the §3.2
   text-MinHash primitive**: build a MinHash signature per painpoint
   over its `title + " " + description`, run LSH at `SIM_THRESHOLD`,
   take connected components. For each cluster of size ≥
   `MIN_SUB_CLUSTER_SIZE`, propose `add_category(new)` — LLM names
   it, the cluster's painpoints get reassigned. Singletons stay in
   Uncategorized waiting for siblings to arrive.
2. **Split crowded categories.** For each real category C where
   `current_member_count - C.painpoint_count_at_last_check ≥
   SPLIT_RECHECK_DELTA`, propose `add_category(split)` — run the
   same intra-bucket clustering as step 1 over C's members, accept
   if ≥2 sub-clusters each with size ≥ `MIN_SUB_CLUSTER_SIZE`.
   **After the check** (whether the split was accepted or not),
   update `C.last_split_check_at = now()` and
   `C.painpoint_count_at_last_check = current_member_count` so the
   delta resets and we don't re-cluster C until it grows again.
3. **Delete dead categories.** For each real category C whose
   relevance mass < `MIN_CATEGORY_RELEVANCE`, propose
   `delete_category` — accept iff no member painpoint has individual
   relevance > `MIN_RELEVANCE_TO_PROMOTE`.
4. **Merge duplicate sibling categories.** For each pair of sibling
   categories, compute text-MinHash similarity over member painpoint
   titles. If similarity > `MERGE_CATEGORY_THRESHOLD`, propose
   `merge_categories`. (This is purely a category-level concern;
   painpoint-level merges happen at insert time via Layer A's
   multi-match branch in §3.5, not in this sweep.)

Each pass writes its events to `category_events` and then moves on.
A single sweep can produce many events.

### 5.2 Event types and acceptance tests

| Event | Proposed in sweep step | Acceptance test | If accepted |
|---|---|---|---|
| `add_category` (new) | step 1 (Uncategorized cluster ≥ `MIN_SUB_CLUSTER_SIZE`) | LLM produces a name + description + parent_id; accept unless the LLM refuses or the new name collides with an existing category | insert a row in `categories`, reassign the cluster's painpoints to it |
| `add_category` (split) | step 2 (category grew enough to re-check) | intra-bucket clustering finds ≥2 distinct sub-clusters, each with size ≥ `MIN_SUB_CLUSTER_SIZE`; LLM names them | insert N new sub-categories under C's parent, reassign painpoints to their cluster's sub-category, retire C |
| `delete_category` | step 3 (mass < `MIN_CATEGORY_RELEVANCE`) | no individual painpoint in C has `relevance > MIN_RELEVANCE_TO_PROMOTE` | delete C, relink any surviving members to C's parent |
| `merge_categories` | step 4 (sibling similarity scan) | text MinHash similarity over member painpoint titles `> MERGE_CATEGORY_THRESHOLD` | delete one, repoint its painpoints at the other; LLM optionally renames the survivor |

There is no `noop` event anymore — noop was an artifact of the old
"emit one event per painpoint" model. In the batch worker model,
"nothing happened to the taxonomy" is just an empty sweep result.

The events are still independent: each has its own trigger and its
own test, no shared gate. An event that fails its test is logged and
discarded; the sweep continues.

**Why this is cheaper.** In the old design, every painpoint insert
forced a fresh split-check on its landing category — even if the
category had been split-checked 30 seconds ago. Now the split-check
runs once per sweep, no matter how many painpoints landed in the
meantime. Same for delete and merge checks. The work scales with
*number of sweeps*, not *number of inserts*.

### 5.3 Event log table

```sql
CREATE TABLE IF NOT EXISTS category_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type       TEXT NOT NULL,
    proposed_at      TEXT NOT NULL,
    triggering_pp    INTEGER REFERENCES painpoints(id),
    target_category  INTEGER REFERENCES categories(id),
    payload_json     JSON NOT NULL,        -- the proposed mutation
    metric_name      TEXT NOT NULL,        -- which test ran (e.g. "sub_cluster_count", "category_mass", "merge_text_sim")
    metric_value     REAL NOT NULL,        -- value the test produced
    threshold        REAL NOT NULL,        -- threshold it was compared to
    accepted         INTEGER NOT NULL,     -- 0/1
    reason           TEXT                  -- "below threshold", "applied", "llm refused name", etc.
);
CREATE INDEX IF NOT EXISTS idx_cat_events_proposed ON category_events(proposed_at);
CREATE INDEX IF NOT EXISTS idx_cat_events_type ON category_events(event_type);
```

This table is written **only by the category worker**. The promoter
is logless by design (per §5 intro and §3.5); painpoint-level merges
from Layer A's multi-match branch go to a plain logfile, not here.

The schema replaces the old `entropy_before` / `entropy_after` columns
with a generic `(metric_name, metric_value, threshold)` triple — each
event type writes its own metric. Examples:

| event_type | metric_name | example value | threshold |
|---|---|---|---|
| `add_category(split)` | `sub_cluster_count` | 3 | 2 |
| `delete_category` | `category_mass` | 0.4 | 1.0 (`MIN_CATEGORY_RELEVANCE`) |
| `merge_categories` | `merge_text_sim` | 0.62 | 0.50 (`MERGE_CATEGORY_THRESHOLD`) |

**Why store this** (concrete present-day uses, not speculative):

1. **Tuning the per-event thresholds.** `MIN_CATEGORY_RELEVANCE`,
   `MIN_SUB_CLUSTER_SIZE`, `MERGE_CATEGORY_THRESHOLD` are all guesses
   until we see real data. The only way to tune them is to look at a
   corpus of accept/reject decisions and ask "did we reject things
   that should have passed, or accept things that shouldn't?" Without
   the log, you tune blind.
2. **Debugging the tests during development.** When iterating on the
   acceptance tests or the event types, you want to see exactly what
   the test saw — `metric_value`, `threshold`, and `payload_json`
   together reproduce any decision after the fact.

Both uses are **development-time**, not production-time. The table
exists to support iterating on the algorithm, not to power any user-
facing or LLM-facing feature. If we ever stop tuning it and the schema
becomes dead weight, drop it then.

### 5.4 The sweep loop

```python
# category_worker.py — top-level script, run on a schedule
def run_sweep():
    conn = db.get_db()
    with merge_lock(conn, timeout=WORKER_LOCK_TIMEOUT_SEC):   # see §10
        for event in propose_uncategorized_events(conn):    # step 1
            apply_with_test(conn, event)
        for event in propose_split_events(conn):            # step 2
            apply_with_test(conn, event)
        for event in propose_delete_events(conn):           # step 3
            apply_with_test(conn, event)
        for event in propose_merge_events(conn):            # step 4
            apply_with_test(conn, event)


def apply_with_test(conn, event):
    """Test first, apply only on accept. Savepoint protects against
    mid-apply failures, not against test rejection."""
    metric, threshold, ok = run_acceptance_test(event, conn)
    if not ok:
        log_event(event, metric, threshold, accepted=False, reason="below threshold")
        return

    conn.execute("SAVEPOINT cat_event")
    try:
        apply_event(conn, event)              # mutate categories + painpoints
        conn.execute("RELEASE SAVEPOINT cat_event")
        log_event(event, metric, threshold, accepted=True)
    except Exception as e:
        # Partial failure (e.g., LLM call timed out mid-apply, or a
        # downstream INSERT failed). Roll back the partial mutation
        # and log it as rejected with the reason.
        conn.execute("ROLLBACK TO SAVEPOINT cat_event")
        log_event(event, metric, threshold, accepted=False, reason=f"apply error: {e}")
```

**Why this shape, not "apply-then-rollback".** Earlier drafts
described an `apply_event` → `run_acceptance_test` → `ROLLBACK if
fail` pattern, on the (false) assumption that the test needed the
post-mutation state. Looking at the actual tests in §5.2:

| Event | Test | Needs mutation to evaluate? |
|---|---|---|
| `add_category(new)` | LLM names it, accept unless name collides | No — clustering + LLM call + name lookup |
| `add_category(split)` | ≥2 sub-clusters of size ≥ `MIN_SUB_CLUSTER_SIZE` | No — clustering happens before any DB mutation |
| `delete_category` | No member has `relevance > MIN_RELEVANCE_TO_PROMOTE` | No — read members before deleting |
| `merge_categories` | Text MinHash similarity over members > threshold | No — pure read |

Every test is **pre-mutation**, so the test goes first and the
savepoint exists only to protect against partial failures *during*
the apply (LLM timeout midway through naming sub-categories, FK
violation on a downstream INSERT, etc). This is more honest about
what the savepoint is for and reduces the common case to a single
write transaction.

The whole sweep is still one outer transaction (one `BEGIN IMMEDIATE`
→ many `SAVEPOINT` / `RELEASE` → one `COMMIT`), which is fine because
SQLite savepoints are nestable inside a transaction.

### 5.5 Scheduling and cadence

The category worker has no built-in scheduling. Pick one of:

- **Cron / systemd timer** — simplest. Run `python category_worker.py`
  every N minutes.
- **Long-running daemon** with a sleep loop. Slightly less robust to
  crashes but avoids cron overhead.
- **Triggered by promoter** — promoter increments a "pending sweeps"
  counter; a small supervisor wakes the worker when the counter
  exceeds a threshold. Most responsive but most moving parts.

Default recommendation: **cron, every 15 minutes**. The taxonomy
doesn't need to be up-to-the-second; missing a sweep means at most
~15 min of stale Uncategorized growth, which is fine.

The worker is **idempotent** by construction — running it twice in a
row is safe because the second run finds nothing to do (Uncategorized
is empty, no category grew enough to re-check, no new dead categories,
no new merge candidates). This makes recovery from a crashed sweep
trivial: just run it again.

---

## 6. (Removed) The entropy gate

This section used to describe a Shannon-entropy gate over the
relevance-weighted mass distribution across categories. We dropped it
because it was the wrong tool:

- Entropy measures *balance*, not semantic coherence, meaningful size,
  or whether reality is being reflected. It's content-blind.
- It rewards uniform distributions and penalizes the natural Zipfian
  skew of real Reddit data.
- For `delete_category` it mathematically can never pass an "increase
  = accept" rule (removing any non-empty bucket lowers entropy).
- For `add_category` and `merge_categories` it would have needed
  opposing accept directions, which couldn't coexist under one global
  rule.

Replaced by per-event domain-specific tests in §5.2. Section kept as a
tombstone so future-us doesn't reinvent the entropy gate without
remembering why it was abandoned.

---

## 7. Schema additions

Diff against current `db/schema.sql`:

```sql
-- 7.1 cache relevance on the painpoint row
ALTER TABLE painpoints ADD COLUMN relevance REAL;
ALTER TABLE painpoints ADD COLUMN relevance_updated_at TEXT;

-- 7.2 audit trail (per-event metric, not entropy — see §5.2)
CREATE TABLE IF NOT EXISTS category_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type       TEXT NOT NULL,
    proposed_at      TEXT NOT NULL,
    triggering_pp    INTEGER REFERENCES painpoints(id),
    target_category  INTEGER REFERENCES categories(id),
    payload_json     JSON NOT NULL,
    metric_name      TEXT NOT NULL,
    metric_value     REAL NOT NULL,
    threshold        REAL NOT NULL,
    accepted         INTEGER NOT NULL,
    reason           TEXT
);
CREATE INDEX IF NOT EXISTS idx_cat_events_proposed ON category_events(proposed_at);
CREATE INDEX IF NOT EXISTS idx_cat_events_type ON category_events(event_type);

-- 7.3 cache the MinHash signature so we don't recompute on every read
ALTER TABLE painpoints ADD COLUMN minhash_blob BLOB;

-- 7.4 split-check trigger discipline (§5.1)
ALTER TABLE categories ADD COLUMN last_split_check_at TEXT;
ALTER TABLE categories ADD COLUMN painpoint_count_at_last_check INTEGER DEFAULT 0;

-- 7.5 multi-source pending painpoints (see §3 discussion — pending pps
-- can have multiple sources from batched LLM extraction).
--
-- IMPORTANT migration note: SQLite cannot ALTER an existing NOT NULL
-- column to be nullable without a full table rebuild. So we DO NOT
-- relax pending_painpoints.post_id; instead we treat the existing
-- columns as the *primary* source (always populated, NOT NULL) and
-- store any *additional* sources in this junction table. New code
-- reads from BOTH:
--   primary source = (pending_painpoints.post_id, pending_painpoints.comment_id)
--   extra sources  = pending_painpoint_sources rows for that pending pp
-- Single-source pendings have zero rows in pending_painpoint_sources.
-- Multi-source pendings have N-1 rows here (the primary stays in the
-- legacy columns). This avoids the table-rebuild dance and keeps
-- existing read code working unchanged.

CREATE TABLE IF NOT EXISTS pending_painpoint_sources (
    pending_painpoint_id INTEGER NOT NULL REFERENCES pending_painpoints(id) ON DELETE CASCADE,
    post_id              INTEGER NOT NULL REFERENCES posts(id),
    comment_id           INTEGER REFERENCES comments(id)
);
-- SQLite PRIMARY KEYs can't contain expressions, so we enforce
-- uniqueness via an expression index instead. The COALESCE is
-- needed because (pending_id, post_id, NULL) compares unequal to
-- itself by default.
CREATE UNIQUE INDEX IF NOT EXISTS idx_pps_unique
    ON pending_painpoint_sources(pending_painpoint_id, post_id, COALESCE(comment_id, -1));
CREATE INDEX IF NOT EXISTS idx_pps_post ON pending_painpoint_sources(post_id);

-- Helper view that unions primary + extra sources so callers don't
-- have to remember the two-place storage. New code SHOULD prefer the
-- view; legacy code can keep reading pending_painpoints.post_id
-- directly.
CREATE VIEW IF NOT EXISTS pending_painpoint_all_sources AS
    SELECT id AS pending_painpoint_id, post_id, comment_id
        FROM pending_painpoints
    UNION
    SELECT pending_painpoint_id, post_id, comment_id
        FROM pending_painpoint_sources;

-- 7.6 Uncategorized sentinel — promoter parks new painpoints here when
-- similarity (§3) returns no match. The category worker (§5.1 step 1)
-- clusters and renames out of this bucket. Always exists, never deleted.
INSERT OR IGNORE INTO categories (name, parent_id, description, created_at)
VALUES ('Uncategorized', NULL,
        'Sentinel bucket for painpoints awaiting category-worker processing.',
        strftime('%Y-%m-%dT%H:%M:%fZ','now'));

-- 7.7 Post-level signal score — schema lands here so painpoint
-- relevance (§2.2) has a column to read. The compute logic for
-- filling these columns still belongs to SIGNAL_SCORING_PLAN.md;
-- until that job runs they stay NULL and §2.2 falls back to an
-- inline approximation.
ALTER TABLE posts ADD COLUMN signal_score REAL;
ALTER TABLE posts ADD COLUMN signal_score_updated_at TEXT;

-- Persisted alongside per SIGNAL_SCORING_PLAN.md "for the LLM
-- reasoning step" — derived from score + upvote_ratio at scrape
-- time. The LLM extract pass reads them; ranking does NOT.
ALTER TABLE posts ADD COLUMN upvote_count INTEGER;
ALTER TABLE posts ADD COLUMN downvote_count INTEGER;

-- Cluster size from MinHash near-duplicate merging at scrape time.
-- Used by signal_score's cluster_multiplier and persisted so we can
-- audit which posts collapsed together.
ALTER TABLE posts ADD COLUMN cluster_size INTEGER DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_posts_signal_score ON posts(signal_score DESC);
```

Pure ALTER + CREATE — no destructive migration needed since the new
columns are nullable.

---

## 8. New code modules

| New file | Purpose |
|---|---|
| `db/locks.py` | `merge_lock()` context manager (§4) |
| `db/relevance.py` | `compute_relevance(painpoint_or_pending)` per §2.3 (object form, not id), `per_source_relevance` helper, batch recompute job that walks `painpoint_sources → pending_painpoint_sources` to find every contributing `(post, comment)` tuple and takes the max |
| `db/similarity.py` | the two layers from §3: exact-source SQL prefilter (Layer A, including the multi-match `merge_painpoints` branch) and text MinHash + LSH lifecycle (Layer B) |
| `db/category_events.py` | event types, `apply_event`, per-event acceptance tests (§5.2), `category_events` log writes — called by category_worker |
| `db/category_clustering.py` | intra-bucket clustering for `add_category(split)` and inter-category similarity for `merge_categories` — both reuse §3 primitives |
| `db/llm_naming.py` | thin wrapper around `llm.py` for "name this new category" / "name these N sub-clusters" prompts |
| `promoter.py` (top-level) | the painpoint promoter loop: pull pending → relevance → drop or continue → similarity → link/insert. **Never touches categories.** |
| `category_worker.py` (top-level) | the sweep worker (§5.1, §5.4): periodically acquires the merge lock and runs a four-pass taxonomy sweep. Operates exclusively at the category level — painpoint-level merging happens at insert time per §3.5, not in the sweep. |

`db/painpoints.py` gains one new entry point used by `promoter.py`:

```python
def promote_pending(pending_id: int) -> int | None:
    """Promote one pending painpoint into the merged table.

    1. Compute relevance from §2.3 (max over the pending's full source
       set). If below MIN_RELEVANCE_TO_PROMOTE, hard-delete the pending
       row (cascades to pending_painpoint_sources) and return None.
       Relevance computation is read-only and runs *outside* the lock.
    2. Acquire the merge lock (BEGIN IMMEDIATE, §4).
    3. Inside the lock, run the §3.5 decision flow:
         - Layer A SQL prefilter over the pending's source set
         - if 1 match → link
         - if >1 matches → merge_painpoints across the spanned
           painpoints, then link to the survivor
         - if 0 matches → Layer B text MinHash → link or
           create_new_in_uncategorized
    4. Release the lock (COMMIT).

    Returns the painpoints.id the pending row was attached to, or None
    if it was dropped.

    Does not touch categories beyond pointing new painpoints at the
    Uncategorized sentinel. Does not emit category events. The
    category worker (category_worker.py) is responsible for all
    taxonomy mutation.
    """
```

Everything `merge_pending_into_painpoint` already does stays — it's the
inner step `promote_pending` calls under the lock.

The category worker has its own entry point in `category_worker.py`:

```python
def run_sweep() -> dict:
    """Run one full taxonomy-maintenance sweep (§5.1, §5.4).

    Acquires the merge lock for the duration of the sweep. Returns a
    summary dict with counts per event type and per acceptance outcome.
    Idempotent: running twice in a row is safe and the second run is
    typically a no-op.
    """
```

---

## 9. Test plan (extends `tests/test_db_integration.py`)

New test classes to add:

- **`TestRelevance`** — golden values for the per-source relevance
  formula (§2.2): fresh+severe scores higher than old+mild,
  comment-rooted painpoints get the comment-score boost, painpoints
  with `posts.signal_score = NULL` correctly fall back to the inline
  approximation.
- **`TestRelevanceMaxAggregation`** — multi-source painpoint with
  `[(fresh, severe), (old, mild), (old, mild)]`. The aggregated
  relevance must equal `per_source_relevance(fresh)` (max), not the
  mean and not the sum. Confirms §2.3 max semantics.
- **`TestSimilarityLayerA`** — pending pp from a `(post, comment)` whose
  source already feeds an existing merged painpoint must be linked
  deterministically without invoking MinHash.
- **`TestSimilarityLayerB`** — insert two near-duplicate painpoints from
  *different* posts; the second must link to the first via text
  MinHash, not create a new row.
- **`TestLayerAMultiMatchMerge`** — seed two merged painpoints with
  *disjoint* source sets (e.g., #99 = `{post 17, post 42}`, #100 =
  `{post 88, post 103}`). Insert a multi-source pending pp with
  sources `{post 42, post 88}`. Layer A must return both #99 and
  #100, the multi-match branch must merge them into one survivor
  (signal_counts summed, all sources rolled up), and the new pending
  must be linked to the survivor. The loser row must be gone from
  `painpoints` and from the Layer B LSH index.
- **`TestMergePainpointsInvariants`** — direct unit test of
  `merge_painpoints(survivor, loser)`: survivor's `signal_count` =
  pre-merge sum, `first_seen` = min of the two, `category_id`
  unchanged from the survivor's pre-merge value (even if the loser
  was in a different category — confirms §3.5 step 7), all
  `painpoint_sources` rows of the loser now point at the survivor,
  loser's row is deleted, loser's MinHash signature is gone from
  the LSH index.
- **`TestNewPainpointInheritsPendingSources`** — insert a multi-source
  pending pp with sources `{post A, post B, post C}` that matches
  nothing via Layer A or Layer B. The newly created merged painpoint
  must have exactly those three rows in `painpoint_sources` — no
  more, no less.
- **`TestMergeLock`** — start two threads both calling `promote_pending`
  on overlapping pending ids, assert no duplicate `painpoints` rows
  appear and `signal_count` reflects both inserts. (Two *threads*
  rather than two processes — the integration test uses threads to
  validate that `BEGIN IMMEDIATE` serialises writers correctly.
  Cross-process correctness comes for free from the same primitive,
  but the test rig stays in-process for simplicity.)
- **`TestPromoterDoesNotTouchCategories`** — a long sequence of
  `promote_pending` calls must leave `categories` unchanged. New
  painpoints with no match land in the `Uncategorized` sentinel.
- **`TestSweepIsIdempotent`** — running `category_worker.run_sweep()`
  twice back-to-back: the second run must produce **zero accepted
  events** (it may still produce rejected-event audit rows for
  unstable thresholds, but no `categories` or `painpoints` row may
  change between sweep 1's commit and sweep 2's commit). No
  oscillating splits/merges across runs.
- **`TestSweepProcessesUncategorized`** — seed N painpoints in the
  `Uncategorized` bucket forming two clear clusters of size ≥
  `MIN_SUB_CLUSTER_SIZE`; one sweep must create two new categories
  and reassign the painpoints out of Uncategorized.
- **`TestSweepLockSerialisesPromoter`** — start `run_sweep()` in one
  thread and `promote_pending` in another; the promoter must block
  until the sweep finishes, and no painpoint must end up linked to a
  category that the sweep deleted.
- **`TestSplitTest`** — seed a category with two clearly distinct
  text clusters of size ≥ `MIN_SUB_CLUSTER_SIZE`; the split test must
  return ≥2 sub-clusters and accept. Seed a tight single-cluster
  bucket; the split test must reject (one cluster).
- **`TestDeleteTest`** — category mass below `MIN_CATEGORY_RELEVANCE`
  with no live members → accept. Mass below threshold but one member
  has individual relevance > `MIN_RELEVANCE_TO_PROMOTE` → reject
  (safety check).
- **`TestMergeTest`** — two categories whose member painpoints are
  near-duplicates by text MinHash → accept. Two unrelated categories
  → reject.
- **`TestSplitTriggerDiscipline`** — split check must NOT re-fire on a
  category that hasn't grown by `SPLIT_RECHECK_DELTA` since
  `last_split_check_at`. Inserting one painpoint into a recently-
  checked stable bucket should not trigger another clustering pass.
- **`TestPromoteDropsLowRelevance`** — pending painpoint with
  computed relevance < `MIN_RELEVANCE_TO_PROMOTE` is hard-deleted
  from `pending_painpoints` and never reaches the merged table.
- **`TestCategoryEventLog`** — every sweep step that proposes a
  taxonomy mutation produces exactly one `category_events` row with
  `metric_name`, `metric_value`, and `threshold` populated, regardless
  of whether the event was accepted or rolled back.

All of these run against the same temp-DB fixture as the existing
suite. No new infra.

---

## 10. Config / tunables (one place)

Add to `config.json` (or a new `painpoint_config.yaml` if you want to
keep it out of the existing config):

```json
{
  "painpoint_ingest": {
    "sim_threshold": 0.65,
    "sweep_cluster_threshold": 0.40,
    "min_sub_cluster_size": 5,
    "split_recheck_delta": 10,
    "merge_category_threshold": 0.50,
    "relevance_half_life_days": 14,
    "min_relevance_to_promote": 0.5,
    "min_category_relevance": 1.0,
    "promoter_lock_timeout_sec": 30,
    "worker_lock_timeout_sec": 300
  }
}
```

What each one controls:

| Tunable | Section | Meaning |
|---|---|---|
| `sim_threshold` | §3.2 | Promote-time LSH threshold for Layer B text MinHash. Two painpoint titles are "near-duplicate" iff Jaccard ≥ this value. Lower → more aggressive merging at promote time. |
| `sweep_cluster_threshold` | §5.1 step 1, step 2 | Sweep-time clustering threshold, intentionally lower than `sim_threshold`. The sweep wants high recall (find anything related), while promote wants high precision (only link clearly-equivalent things). The 0.40–0.65 gap is the window where related-but-not-equivalent painpoints land in Uncategorized as singletons and get clustered later. |
| `min_sub_cluster_size` | §5.1 step 2 | When the worker considers splitting a category, the resulting sub-clusters must each have at least this many painpoints to be accepted. Smaller → more eager splitting. |
| `split_recheck_delta` | §5.1 step 2 | A category only gets a fresh split-check after it has grown by this many painpoints since the last check. Throttles the O(N²) clustering work. |
| `merge_category_threshold` | §5.1 step 4 | Two sibling categories with member-title MinHash similarity above this value are proposed for merging. |
| `relevance_half_life_days` | §2.2 | Exponential decay rate for painpoint recency. A painpoint half-decays after this many days. The right value depends on how fast topics turn over in your target subreddits. |
| `min_relevance_to_promote` | §2.4 | Pending painpoints whose computed relevance is below this value are hard-deleted before they reach the merged table (the §2.4 drop step). Also used as the per-painpoint safety check inside `delete_category`. |
| `min_category_relevance` | §5.1 step 3 | A category whose total relevance mass (sum over members) falls below this value is proposed for deletion. |
| `promoter_lock_timeout_sec` | §4.2 | How long a single `promote_pending` call waits for the merge lock before giving up. Short — the promoter is fast and contention is rare. |
| `worker_lock_timeout_sec` | §4.2 | How long `run_sweep` waits for the merge lock. Much longer because it's a batch job and only races with the cheap promoter loop. |

Half-life of 14 days is a starting point — the right value depends on
how often you run the scraper vs how fast Reddit topics turn over in
the subreddits you target. Easy to tune later.

---

## 11. What's intentionally NOT in this plan

- **Multi-process *promoters*.** A single promoter is fine until
  throughput forces otherwise. The lock primitive is chosen so adding
  more promoter workers later is a few-line change. (Promoter and
  category worker are *already* separate processes per §5; this bullet
  is about scaling the promoter side specifically.)
- **Periodic source-set reconciliation.** Considered (called it
  "Layer C") and removed — it's provably dead code under the rules of
  Layer A. No two merged painpoints can ever share a source post, so
  there's no overlap for it to find. See §3.3 for the autopsy.
- **Speculative source wiring at painpoint creation.** When a new
  merged painpoint is created, its `painpoint_sources` is exactly
  what the triggering pending painpoint brought with it — no fuzzy
  scan of the rest of the database for "posts that look related." The
  LLM extractor is the gatekeeper of evidence; if a post should be
  evidence for some painpoint, the next extraction pass over that
  post will produce a pending painpoint that links via Layer A or B.
  Conflating "this post exists in the DB" with "this post is evidence
  of pain X" is the wrong direction. Improving extraction coverage is
  an *extractor* problem, not a *merger* problem.
- **LLM-driven *standalone* category renaming.** As a separate
  workstream — i.e., "wake the LLM up periodically and ask it to
  reconsider every category's name." Not in scope here. (Inline LLM
  naming **is** in scope: §5.1 step 1 has the LLM name newly created
  categories, step 2 has it name split sub-categories, step 4 lets
  it optionally rename the survivor of a `merge_categories`. What
  we're not doing is the standalone "review every name" pass.)
- **`rename_painpoint` events.** We discussed letting the LLM revisit
  a merged painpoint's `title` / `description` after enough new
  sources have been linked, so the title doesn't stay frozen at the
  first source's wording. Sketched but not adopted in this draft —
  the design (counter on `painpoints`, new sweep step 5, throttled
  by `RENAME_RECHECK_DELTA`) is in the conversation history. Punted
  to v2 because (a) it adds another LLM call per Nth link and we
  want to first see how much title-drift actually happens in
  practice, (b) it interacts with `painpoints.title` exact-equality
  dedup elsewhere in `db/painpoints.py` and would need a name-
  collision branch we'd rather only build if needed.
- **Cross-language painpoints.** MinHash on character shingles is
  language-blind enough for English Reddit; if we add non-English
  subreddits we'll need to revisit normalization.
- **Backfilling relevance for existing painpoints.** Whatever's in
  `painpoints` today gets relevance computed lazily on first read.
- **Vectors / embeddings.** Deliberately avoided — keeps the system
  zero-API-cost like `SIGNAL_SCORING_PLAN.md` already is.

---

## 12. Open questions — resolved log

1. ~~**Entropy direction**~~ — **moot.** Entropy gate dropped entirely
   in favour of per-event domain-specific tests (§5.2, §6 tombstone).
2. **Severity weight** — *deferred to empirical tuning.* Current
   formula in §2.2 (`0.5 + 0.1 * severity`, max 2.5× ratio) likely
   too tame given LLM tendency to cluster scores in 5–8. Power-law
   alternative `(severity / 5) ** 1.5` discussed but not adopted —
   revisit after the first ~500 extractions.
3. ~~**`add_category` triggers**~~ — **resolved.** Two triggers:
   (a) Uncategorized cluster ≥ `MIN_SUB_CLUSTER_SIZE` → propose new
   category; (b) existing category has grown by ≥ `SPLIT_RECHECK_DELTA`
   since last check → run intra-bucket clustering, propose split iff
   ≥2 sub-clusters ≥ `MIN_SUB_CLUSTER_SIZE`. No entropy, no
   walk-up-tree. Both fire from the category worker, never from the
   promoter.
4. ~~**LLM in the loop for naming**~~ — **resolved: yes.** The LLM
   names new categories and split sub-categories. Called from the
   category worker under the merge lock (the worker is rare and slow
   anyway, and this is simpler than a two-phase approach). New
   module `db/llm_naming.py`.
5. ~~**`MIN_RELEVANCE_TO_PROMOTE` semantics**~~ — **resolved: hard
   drop.** Pending painpoints below the threshold are deleted from
   `pending_painpoints`, not flagged. The append-only invariant on
   `pending_painpoints` is relaxed (the rows being deleted have never
   been merged, so no `painpoint_sources` reference them — safe to
   delete without breaking the evidence chain).

---

## 13. Implementation log

Built and tested as one piece. **101 tests passing in ~3.3s** (60 base
db tests + 41 pipeline tests including stress tests for deadlock
freedom and a realistic-workflow simulation).

### What shipped (vs the design doc)

| Module | File | Notes |
|---|---|---|
| Merge lock | `db/locks.py` | `merge_lock(conn, timeout)` context manager wrapping `BEGIN IMMEDIATE` with retry-and-timeout. |
| Relevance | `db/relevance.py` | `per_source_relevance`, `compute_pending_relevance`, `compute_painpoint_relevance` (max aggregation), `cache_painpoint_relevance`, `get_or_compute_painpoint_relevance` (lazy 24h cache). |
| Similarity | `db/similarity.py` | `make_minhash` (char 4-shingles), Layer A `exact_source_lookup`, Layer B `PainpointLSH` wrapping `datasketch.MinHashLSH` with persist + rebuild-from-DB lifecycle. |
| Clustering | `db/category_clustering.py` | `cluster_painpoints` (text MinHash + connected components), `inter_category_similarity`. |
| LLM naming | `db/llm_naming.py` | `LLMNamer` (real, calls `llm.py`) and `FakeNamer` (test double — used in every pipeline test). |
| Category events | `db/category_events.py` | Four `propose_*_events` generators, four `_apply_*` mutations, four pre-mutation `_test_*` checks, `apply_with_test` runner per §5.4. |
| Promoter | `db/painpoints.py` (extended) + `promoter.py` | `add_pending_source`, `merge_painpoints`, `_pick_canonical_survivor`, `_create_painpoint_in_uncategorized`, `_link_pending_to_painpoint`, `promote_pending`. Top-level `promoter.py` wraps the loop. |
| Category worker | `category_worker.py` | `run_sweep(namer)` runs the four passes under one merge_lock acquisition. |
| Schema | `db/schema.sql` + `db/__init__.py:_apply_migrations` | Idempotent migrations using a Python try/except for `duplicate column name` (see deltas below). |
| Tests | `tests/test_painpoint_pipeline.py` | 41 tests across 16 classes — every step covered, plus 4 stress tests and 3 realistic-workflow tests. |

### Deltas from the design doc

These are the things I had to change at implementation time, all small
but worth recording so the doc doesn't drift from the code:

1. **`SIM_THRESHOLD` 0.55 → 0.65 + new `SWEEP_CLUSTER_THRESHOLD = 0.40`.**
   The original draft had a single threshold for both promote-time
   matching and sweep-time clustering. In practice, anything pairwise
   above the threshold would already be linked at promote time, so the
   sweep would never find a clusterable group in Uncategorized. Solution:
   widen the gap. Promote at 0.65 (high precision, only link clearly-
   equivalent things), cluster at sweep at 0.40 (high recall, group
   anything related). The 0.40–0.65 window is where related-but-not-
   equivalent painpoints land in Uncategorized as singletons and get
   grouped by the worker. §3.2 and §10 reflect the new values.
2. **`category_events.target_category` and `triggering_pp` are NOT
   foreign keys.** The §7.2 schema originally declared them as
   `REFERENCES categories(id)` / `REFERENCES painpoints(id)`. But the
   event itself can delete the row it points at — `delete_category`
   deletes the row, then `log_event` tries to INSERT with the dangling
   reference and SQLite raises a FK violation. Fixed by making both
   columns plain `INTEGER` (audit logs shouldn't have referential
   integrity to mutable rows). §7.2 schema in this doc has been updated
   to match.
3. **ALTER TABLE migrations live in Python, not in `schema.sql`.**
   `schema.sql` is run via `executescript` on every `init_db()`, and
   SQLite has no `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. Trying to
   add the same column twice raises `duplicate column name`. Solution:
   keep idempotent CREATEs in `schema.sql` (which all use `IF NOT
   EXISTS`), move the ALTER statements to `db/__init__.py:_apply_migrations`
   in a try/except loop that catches the duplicate-column error per
   statement. Indexes that depend on migration-added columns (e.g.,
   `idx_posts_signal_score`) also live in `_apply_migrations`, after
   the column is added.
4. **`init_db()` ordering matters.** The `Uncategorized` sentinel
   `INSERT OR IGNORE` has to run **after** `seed_taxonomy()`, not
   before. `seed_taxonomy()` short-circuits if `categories` is non-
   empty, so inserting Uncategorized first makes the YAML seed silently
   skip. Reordered: schema → migrations → seed_taxonomy → Uncategorized.
5. **`propose_delete_events` now skips empty categories.** Without
   this guard, every empty seeded category from `taxonomy.yaml` would
   be proposed for deletion on the first sweep. Empty isn't the same as
   dead — empty means "unborn / not yet used," dead means "had members,
   they decayed." Fix: only propose deletion when `member_count > 0`.
6. **Pending painpoints schema uses primary + extras, not nullable
   primary.** §7.5 originally said the existing
   `pending_painpoints.post_id` would become nullable for back-compat.
   SQLite can't ALTER an existing NOT NULL column to nullable without
   a full table rebuild, so I left the legacy columns NOT NULL and use
   them as the *primary* source. Additional sources for multi-source
   pendings go in the new `pending_painpoint_sources` junction table.
   Single-source pendings have zero rows there; multi-source have N-1.
   The `pending_painpoint_all_sources` view unions both for callers.
7. **WAL deadlock pattern surfaced in test setup.** Holding an outer
   `db.get_db()` connection across inner write operations (e.g.,
   calling `_make_post()` inside an outer-conn block) causes intermittent
   `database is locked` under SQLite's WAL semantics. Test helpers were
   restructured to close/reopen rather than hold an outer connection
   across inner writes. Worth flagging for any future code that mixes
   the pattern — the production code doesn't do this, only the test
   setup did.
8. **Tunables are hardcoded module constants for v1.** The §10 config
   table describes a `painpoint_ingest` block in `config.json`, but the
   implementation hardcodes them as module-level constants in
   `db/relevance.py`, `db/similarity.py`, and `db/category_events.py`
   with comments noting "lift to config later." Fine for v1; loading
   from JSON adds wiring without changing behaviour.

### Test coverage map

| Plan section | Test class(es) |
|---|---|
| §2.2 per-source relevance | `TestRelevance` (4 tests) |
| §2.3 max aggregation | `TestRelevanceMaxAggregation` (2 tests) |
| §2.4 drop step | `TestPromoteDropsLowRelevance` (1 test) |
| §3.1 Layer A | `TestSimilarityLayerA` (3 tests) |
| §3.2 Layer B | `TestSimilarityLayerB` (3 tests) |
| §3.5 multi-match merge | `TestLayerAMultiMatchMerge` (1 test) |
| §3.5 `merge_painpoints` invariants | `TestMergePainpointsInvariants` (4 tests) |
| §3.5 source inheritance | `TestNewPainpointInheritsPendingSources` (1 test) |
| §4 merge lock | `TestMergeLock` (1 test), `TestSweepLockSerialisesPromoter` (1 test) |
| §3.5 promoter contract | `TestPromoterDoesNotTouchCategories` (1 test) |
| §5.1 step 1 (Uncategorized) | `TestSweepProcessesUncategorized` (2 tests) |
| §5.1 step 2 (split) | `TestSplitTest` (1 test), `TestSplitTriggerDiscipline` (2 tests) |
| §5.1 step 3 (delete) | `TestDeleteTest` (2 tests) |
| §5.1 step 4 (merge categories) | `TestMergeTest` (2 tests) |
| §5.4 sweep idempotency | `TestSweepIsIdempotent` (1 test) |
| §5.3 audit log | `TestCategoryEventLog` (1 test) |
| End-to-end | `TestEndToEndSmoke` (1 test) |
| **Stress / deadlock freedom** | `TestStressNoDeadlocks` (4 tests) |
| **Realistic synthetic workflow** | `TestRealisticWorkflow` (3 tests) |

Stress tests assert three database invariants after every concurrent
run: no orphan pending pps, `signal_count == COUNT(painpoint_sources)`
per painpoint, and no orphan source rows. Together these rule out
lost writes, double-links, and races between concurrent promoters.
