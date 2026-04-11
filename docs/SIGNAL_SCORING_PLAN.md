# Post rating algorithm

A single `signal_score` per post computed from **only the fields Reddit
already returns in the listing response** plus a pure-Python MinHash
pass. **Zero external API calls, zero LLM tokens.** This replaces the
current LLM filter pass entirely.

The rating job is narrow: **"is this post worth looking at?"** It's not
trying to judge whether controversy is good or bad, whether a topic is
hot, or whether the community is fighting — those are interpretive
questions best left to the LLM reasoning step downstream. The ranker
just surfaces high-traction posts cheaply.

Three components:

1. **Engagement** (raw upvotes + comments)
2. **Velocity** (rising posts > decayed posts)
3. **Cluster size** (near-duplicates amplify each other)

Raw vote stats (`score`, `upvote_ratio`, downvote count) are **stored
alongside** each post so the LLM extract pass can factor them into its
interpretation, but they do not influence the rank order.

The score is a positive real number; higher = better. You rank by it,
you don't compare against a threshold.

---

## Inputs per post

All fields come from the Reddit listing JSON — one request returns
everything needed for 100 posts at a time:

| Field | Role |
|---|---|
| `score` (net upvotes) | ranking input + persisted |
| `num_comments` | ranking input + persisted |
| `upvote_ratio` (0.0–1.0) | **persisted only**, not ranked on |
| `created_utc` (epoch seconds) | ranking input |
| `title` + `selftext` | MinHash clustering + persisted |
| `cluster_size` | computed locally, ranking input |

No per-author lookups. No LLM calls. No embeddings. Pure arithmetic on
data we already have.

---

## Component scores

### 1. Engagement

```python
engagement_score = (
      log1p(score)         * 0.5
    + log1p(num_comments)  * 0.8
)
```

**Why log:** a 10k-upvote post shouldn't be worth 100× a 100-upvote
post. Log compresses the tail.

**Why comments are weighted heavier:** an upvote is a 1-bit opinion, a
comment is deliberate effort. Threads where people bother to write
something are where real signal lives.

### 2. Velocity

```python
hours_alive = max(1, (now - created_utc) / 3600)
velocity_score = score / hours_alive
velocity_norm  = min(1.0, velocity_score / 50)
```

Raw upvotes-per-hour, capped at 50 (considered "saturated"). A 500-vote
post that's 1 hour old beats a 500-vote post that's a week old — it's a
rising topic rather than a decayed one.

Clamping `hours_alive` to at least 1 prevents 5-minute-old posts from
ranking absurdly high.

### 3. Cluster size (near-duplicate merging)

Run MinHash over `title + selftext` for every post in the batch, bucket
them with LSH at similarity threshold 0.60, collapse each cluster to its
highest-engagement member, count how many were merged in.

```python
cluster_multiplier = 1 + 0.2 * log1p(cluster_size - 1)
```

| `cluster_size` | multiplier |
|---|---|
| 1 (unique) | 1.00× |
| 3 | 1.14× |
| 10 | 1.44× |
| 30 | 1.68× |

**Intuition:** if 10 people independently posted the same question in
the same week, that's a much stronger signal than one person posting it
once. Cluster size captures community-wide repetition without double-
counting the raw votes.

Cost: ~100ms of CPU for 200 posts via the `datasketch` library.

---

## Final formula

```python
signal_score = (
      engagement_score
    * (1 + 0.5 * velocity_norm)
    * cluster_multiplier
)
```

Multiplicative because the factors are independent *amplifiers* of the
base engagement score. A post gets a big final score when it stacks:
lots of engagement × rising fast × repeated across threads.

### Worked example

**Post A** — fresh, repeated question, moderate votes:
- `score = 820`, `num_comments = 340`
- posted 3 hours ago
- 2 near-duplicate threads merged in (`cluster_size = 3`)

```
engagement_score  = log1p(820)*0.5 + log1p(340)*0.8
                  = 3.36 + 4.66 = 8.02

velocity_score    = 820 / 3 = 273
velocity_norm     = min(1, 273/50) = 1.0

cluster_mult      = 1 + 0.2 * log1p(2) = 1.22

signal_score      = 8.02 * 1.5 * 1.22 = 14.68
```

**Post B** — week-old high-consensus thread:
- `score = 900`, `num_comments = 120`
- posted 168 hours ago
- unique (`cluster_size = 1`)

```
engagement_score  = log1p(900)*0.5 + log1p(120)*0.8
                  = 3.40 + 3.85 = 7.25

velocity_norm     = min(1, (900/168)/50) = 0.11

cluster_mult      = 1.00

signal_score      = 7.25 * 1.055 * 1.0 = 7.65
```

Post A outranks Post B **~1.9×** — driven entirely by freshness and
repetition, not by the raw vote count (which is actually lower). A
high-consensus post with massive traction (`score = 5000`, ratio 0.97,
1 hour old) would blow both of them out of the water because its
engagement + velocity multipliers dominate. Consensus is not penalized
in any way.

---

## Persisted alongside each post (for the LLM reasoning step)

These fields are saved but **do not affect ranking**. The LLM extract
pass reads them when interpreting the post:

- `score`
- `upvote_ratio`
- `num_comments`
- `upvote_count` (derived: `round(score / (2*upvote_ratio - 1))` if
  `upvote_ratio > 0.5`, else a fuzzed estimate)
- `downvote_count` (derived similarly)

Why persist them instead of scoring on them: the meaning of a 0.6 ratio
depends on the content. An LLM reading *"Llama 3 is better than GPT-5
for coding"* with 800 upvotes / 0.58 ratio can tell that's a real
community fight worth noting. The heuristic ranker has no way to
distinguish that from a random low-effort shitpost that got brigaded.
Let the LLM decide.

---

## Parameters / tunables

| Constant | Default | Meaning |
|---|---|---|
| `ENGAGEMENT_SCORE_WEIGHT` | 0.5 | weight on `log1p(score)` |
| `ENGAGEMENT_COMMENTS_WEIGHT` | 0.8 | weight on `log1p(num_comments)` |
| `VELOCITY_SATURATION` | 50 | upvotes/hr considered maximum |
| `VELOCITY_AMPLIFIER` | 0.5 | how much velocity boosts the base score |
| `CLUSTER_AMPLIFIER` | 0.2 | how much cluster size boosts the base score |
| `MINHASH_THRESHOLD` | 0.60 | LSH similarity for near-duplicate clustering |

---

## Edge cases

- **Stickied / pinned posts:** skip entirely before scoring, they
  distort `velocity_score`.
- **Age < 1 hour:** clamp `hours_alive` to 1 in the velocity denominator.
- **`score < 0` (heavily downvoted):** clamp with `log1p(max(0, score))`
  since `log1p` of a negative number is undefined. A heavily downvoted
  post gets near-zero ranking signal but its stats are still persisted
  so the LLM can surface it if the content warrants attention.
- **`cluster_size = 0`:** impossible — a post is always in a cluster of
  at least itself. Always `>= 1`.
- **Empty selftext:** MinHash uses the title only. Clustering still
  works, just with less text to compare.

---

## Cost summary

| Resource | Cost per 200-post batch |
|---|---|
| Reddit API calls beyond the initial listing | **0** |
| OpenAI / LLM tokens | **0** |
| CPU | ~100ms (MinHash) + negligible arithmetic |
| Memory | ~1MB for MinHash signatures |

Ranking runs in under a second on a batch of 200 posts and costs nothing.
