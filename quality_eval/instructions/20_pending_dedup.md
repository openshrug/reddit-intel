# Dimension 2 -- Pending painpoint deduping (extract-time)

> Read `00_protocol.md` first for inputs, snapshot semantics, the
> snapshot-inspection helpers, and the citation rules.

## What's evaluated

At extract time, near-duplicate observations get merged onto an
existing pending via cosine >= `PENDING_MERGE_THRESHOLD` (look up the
active value from the codebase) instead of becoming separate
`pending_painpoints` rows. The collapsed observations land in
`pending_painpoint_sources`. Are the merges correct?

Per group ("group" = one pending plus every extra source attached to
it), judge:

- **True positive** -- the *primary source* and the *extra sources*
  describe the **same** pain in different words; the merge saved a
  near-duplicate row.
- **False positive** -- the texts are about different pains that
  happened to share vocabulary; the merge collapsed signal.
- **Under-merging** is invisible in this section by construction (we
  only see groups that fired). To spot it, cross-reference with
  dimension 3 evidence: pairs of pendings that obviously describe the
  same pain but live in *separate* painpoints.

## Snapshots to use

| Snapshot | Why |
| --- | --- |
| `02_<sub2>` | First snapshot likely to have non-trivial dedup volume (snapshot 1 may be too sparse to exercise the path). |
| `03_<sub3>` | Has the most accumulated pendings, so the extract-time path has had the most opportunity to fire. |

`04_post_sweep` is **not** the right snapshot for this dimension --
the sweep doesn't touch `pending_painpoints` / `pending_painpoint_sources`.

## Source-of-truth code (read before judging)

| Path | What to look for |
| --- | --- |
| `db/painpoints.py: save_pending_painpoints_batch` | The actual dedup branch: how candidate pendings are scored against the embedding nearest-neighbour set, where `PENDING_MERGE_THRESHOLD` is applied, and what gets written to `pending_painpoint_sources` instead of `pending_painpoints`. |
| `db/embeddings.py` | The embedding model + cosine helper used for the nearest-neighbour query. Skim enough to know what "similarity" actually means here (e.g. is it title-only, title+description, etc.). |

Verify the **active threshold value** with `rg '^PENDING_MERGE_THRESHOLD' --type py`
and quote it (with the symbol name) in the report.

## Where the evidence lives

| Source | What's in it |
| --- | --- |
| `dump.md` section 3 ("Pending dedup groups") | Sampled groups with `extra_source_count >= 1`, primary source + every extra rendered with post + comment context side-by-side. |
| `metrics.json` -> `pending_dedup.{groups_with_extras, extra_observations, total_observations, max_extras_in_one_group, rate}` | Volume + how aggressive the path was. `rate = extra_observations / total_observations`. |
| `inspect_db.list_pending_dedup_groups(min_extra_sources=1)` | Same shape as the dump section but the full set, in case the dump's cap (`DEDUP_GROUPS_LIMIT`) hid something. |

## Citation rules

For every group you cite:

- **Pending id** of the surviving row.
- **Snapshot label**.
- **Primary source quote** (post title or comment body) -- verbatim.
- **Extra source quote(s)** that were collapsed onto it -- verbatim,
  one per line.
- A **one-sentence reviewer note** stating "same pain" / "different
  pains" with the reason you reached that verdict.

For under-merging, cite **both** competing pending ids, both source
quotes, and explain why the embedding plausibly missed the merge
(e.g. very different vocabulary, one is in a comment with extra
context).

## Output for `report.md`

A `## Dimension 2 -- Pending dedup` section in the shape defined in
`00_protocol.md` section 6. If `metrics.json -> pending_dedup.groups_with_extras`
is zero across all snapshots, say so explicitly: report Verdict =
"not exercised", note that the extract-time merge path didn't fire in
this run, and skip the examples list (but still flag this as an
observation in the failure modes -- either the data was too sparse or
the threshold is too tight to ever fire).
