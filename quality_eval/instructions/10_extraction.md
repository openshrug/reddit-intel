# Dimension 1 -- Pending painpoint extraction (per single subreddit)

> Read `00_protocol.md` first for inputs, snapshot semantics, the
> snapshot-inspection helpers, and the citation rules.

## What's evaluated

Does the extractor turn a subreddit's posts + comments into a useful
set of pending painpoints? Per-subreddit, judging:

- **Validity** -- is each pending an actual user pain, or noise / opinion
  / pricing complaint that the extractor was supposed to skip?
- **Specificity** -- is the title concrete enough to act on?
- **Attribution** -- does `quoted_text` actually appear in the
  `comment_body` (when `comment_id` is set) or in the
  `post_title + selftext` (when not)?
- **Recall** -- skim 5-10 source posts and look for clear pains the
  extractor missed.

## Snapshots to use

| Snapshot | Why |
| --- | --- |
| `01_<sub1>` | One per subreddit. Each is the promote-time state right after `analyze(<sub>)` finished, so the new pendings in this snapshot are exactly what the extractor produced for that subreddit. |
| `02_<sub2>` | (same) |
| `03_<sub3>` | (same) |

Do **not** use `04_post_sweep` for this dimension -- the sweep mutates
painpoints and categories, but extraction quality is fixed at promote
time.

## Source-of-truth code (read before judging)

| Path | What to look for |
| --- | --- |
| `painpoint_extraction/extractor.py` | The full extraction pipeline: prompt assembly, model invocation, parsing, severity / quoted_text validation, anything filtered before insertion. The prompt itself is the spec for what counts as a "valid" pending. |
| `db/painpoints.py: save_pending_painpoints_batch` | How the extractor's output lands in `pending_painpoints` (so you understand what the row fields mean before grading them). The dedup branch is dimension 2's concern, but knowing where it sits in the same function helps. |

## Where the evidence lives

| Source | What's in it |
| --- | --- |
| `dump.md` section 2 ("New pendings since previous snapshot") | The per-subreddit sample with post + comment context. Sorted by `extra_source_count` desc. |
| `metrics.json` -> `per_subreddit.<name>.pendings` | Total pending count attributable to that subreddit. |
| `metrics.json` -> `per_subreddit.<name>.{posts, comments}` | Denominator for "extraction yield per post". |
| `inspect_db.list_pending_painpoints_for_subreddit(sub)` | All pendings (full set, not just the dump's sample) joined with post + comment context. |
| `db.posts.get_posts_by_ids(ids)` + `db.posts.get_comments_for_post(post_id)` | For the recall check -- pull a few source posts the extractor saw and look for unsurfaced pains. |

## Citation rules

For every example you cite:

- **Pending id** (the `pending_id` field).
- **Snapshot label** (`01_openclaw`, etc.).
- **Source quote**, copy-pasted verbatim from `quoted_text`,
  `comment_body`, or `post_title` + `selftext`. Never paraphrase.
- **Post permalink** so the reviewer can open the original Reddit
  thread.
- A **one-sentence reviewer note** explaining why the pending is good
  / bad / mis-attributed.

## Output for `report.md`

A `## Dimension 1 -- Pending painpoint extraction` section with the
shape defined in `00_protocol.md` section 6 (verdict, numbers, 5+
examples, failure modes). Mix at least one **good** and one **bad**
example per failure mode you call out.

If multiple subreddits exhibit the same failure mode, name it once and
list one example from each subreddit underneath. If a single
subreddit's behaviour is markedly different from the others, say so
and tie the difference to something concrete (subreddit content type,
post volume, comment depth).
