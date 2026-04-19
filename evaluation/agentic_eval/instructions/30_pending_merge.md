# Dimension 3 -- Merging of pending painpoints into a single painpoint

> Read `00_protocol.md` first for inputs, snapshot semantics, the
> snapshot-inspection helpers, and the citation rules.

## What's evaluated

Two pipeline steps add rows to `painpoint_sources` (the linkage table
that gives a painpoint its `signal_count > 1`):

1. **Promote-time** -- `db/painpoints.py: promote_pending` links a new
   pending to an existing painpoint when their embedding cosine beats
   `MERGE_COSINE_THRESHOLD`. This fires inside every `analyze()` call.
2. **Sweep-time `painpoint_merge`** -- `category_worker.run_sweep`
   calls `propose_painpoint_merge_events` (in `db/category_events.py`)
   to find near-duplicate **painpoints** within the same category that
   the promoter's threshold missed (wording variation in the
   ~`MERGE_COSINE_THRESHOLD - 0.10` zone), then confirms each pair
   with an LLM boolean call before collapsing them.

Are the resulting `signal_count > 1` clusters genuine?

- **True positive** -- the linked pendings are different observations
  of the same pain.
- **False positive** -- pendings under one painpoint span unrelated
  topics. The cosine threshold is too loose, or the LLM merge confirm
  was wrong, or the embedding is misleading.
- **Under-merging** -- two near-identical pendings live in *separate*
  painpoints. The threshold is too tight, or the sweep's painpoint
  merge step hasn't been run yet (snapshots 2-3) or didn't catch them
  (snapshot 4).

## Snapshots to use

| Snapshot | Why |
| --- | --- |
| `02_<sub2>` | Promote-time merging only (no sweep yet). Useful to see what the promoter did before the sweep got involved. |
| `03_<sub3>` | Same, with maximum cross-subreddit signal -- a r/sub3 pending merging onto a painpoint first seeded by sub1 / sub2 is the strongest evidence the embedding is doing useful work. |
| `04_post_sweep` | The only snapshot where the sweep's `painpoint_merge` step has run. Compare painpoint counts vs snapshot 3 to see what got collapsed. |

## Source-of-truth code (read before judging)

| Path | What to look for |
| --- | --- |
| `db/painpoints.py: promote_pending` | Promote-time linking: how a pending becomes a painpoint *or* gets attached to an existing one, where `MERGE_COSINE_THRESHOLD` is applied, what fields end up in `painpoint_sources`. |
| `db/category_events.py: propose_painpoint_merge_events` | Sweep-time painpoint dedup proposer: candidate selection (which pairs even get considered), the LLM-confirm prompt, and the merge mechanics. |
| `category_worker.run_sweep` (Step 4.5 block) | Where the proposer is called and how its events are applied. Tells you when in the sweep painpoint merging happens relative to the other steps -- this matters because reroute depends on the post-merge state. |

Verify the **active threshold value** with `rg '^MERGE_COSINE_THRESHOLD' --type py`
and quote it (with the symbol name) in the report.

## Where the evidence lives

| Source | What's in it |
| --- | --- |
| `dump.md` section 4 ("Painpoints with multiple linked pendings") | Top painpoints by `signal_count > 1` with each linked pending's title + source quote + post context. |
| `metrics.json` -> `painpoint_merge.{painpoints_with_multi_sources, pendings_linked, distinct_painpoints, max_signal_count, rate}` | Volume + collapse rate. `rate = (pendings_linked - distinct_painpoints) / pendings_linked`. |
| `metrics.json` -> `delta_vs_previous.linked_to_existing_painpoints` | Cross-subreddit merge signal: count of new pendings that linked to a painpoint that already existed in the previous snapshot. |
| `metrics.json` -> `category_events.painpoint_merge.{proposed, accepted}` (snapshot 4 only) | How many sweep-time painpoint merges fired and how many were accepted. |
| `db.queries.get_top_painpoints(limit=...)` + `db.queries.get_painpoint_evidence(pp_id)` | Ad-hoc deeper sampling beyond what the dump caps (`MULTI_SOURCE_PAINPOINTS_LIMIT`). |

## Citation rules

For every multi-source painpoint you cite:

- **Painpoint id**, **`signal_count`**, **category** at the snapshot.
- **Snapshot label**.
- For each linked pending (at least 2): pending id + verbatim source
  quote.
- A **one-sentence reviewer note**: "same pain" / "different pains" +
  reason.

For under-merging, cite **both** painpoint ids, one source quote from
each, and explain why the embedding plausibly missed (e.g. very
different vocabulary, different category placement, etc.).

For sweep-time painpoint_merge events specifically, also cite the
event id from the audit log (snapshot 4's `dump.md` section 6, or
`metrics.json -> category_events.painpoint_merge`).

## Output for `report.md`

A `## Dimension 3 -- Pending -> painpoint merge` section in the shape
defined in `00_protocol.md` section 7. Cite at least one **good** and
one **bad** multi-source painpoint, and explicitly contrast
promote-time merging (snapshots 2-3) with sweep-time merging
(snapshot 4 only) -- they share a verdict but have different failure
modes.

## When invoked as a sub-agent

If a parent evaluator dispatched you (per `00_protocol.md` section 2),
return a single message in exactly this shape:

```
META: score=<int 1-5>; verdict=<pass|mixed|fail>; headline=<one sentence>

## Dimension 3 -- Pending -> painpoint merge

<full section body as specified in "Output for `report.md`" above>
```

The `META:` line must be the first line of your response, contain no
internal `;` in the headline, and be followed by a blank line then the
`##` heading. Don't add wrapper prose, meta-commentary, or notes about
how you arrived at the verdict -- the parent already has the
protocol; your output is pasted into the report verbatim.
