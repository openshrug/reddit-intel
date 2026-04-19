# Dimension 4 -- Category assignment

> Read `00_protocol.md` first for inputs, snapshot semantics, the
> snapshot-inspection helpers, and the citation rules.

## What's evaluated

Every painpoint ends up under some `categories.id`. There are two
distinct stages to grade:

1. **Promote-time placement** -- `db/embeddings.py: find_best_category`
   picks the best category by cosine vs each category's blended
   vector. Below `CATEGORY_COSINE_THRESHOLD` -> the sentinel
   `Uncategorized`.
2. **Sweep-time taxonomy maintenance** -- `category_worker.run_sweep`
   runs five steps in order:
   1. Uncategorized clustering -> mint new categories from clusters of
      `>= MIN_SUB_CLUSTER_SIZE`.
   1b. LLM review of remaining Uncategorized singletons.
   2. Split crowded categories.
   3. Delete dead categories.
   4. Merge near-duplicate sibling categories.
   4.5. Painpoint-level dedup inside a category (graded by dimension 3,
        not here).
   5. Per-painpoint reroute (singleton mis-routings the split step
      can't cluster).

For both stages, judge:

- **Coherence of leaves** -- pull 3-5 painpoints from each runtime
  leaf; do they actually share a theme?
- **Sibling distinctness** -- are sibling categories near-duplicates?
  If so, the merge step under-fired.
- **Stranded misroutings** -- scan painpoints under unexpected roots
  (e.g. an `AI/ML` painpoint under `Dating`); the reroute step should
  have caught those.
- **Dropped categories** -- any `delete_category` accept that removed
  something useful?
- **Uncategorized share** -- expected to be high in snapshots 1-3
  (clean DB only has the seed taxonomy and no runtime categories
  yet); should be substantially lower post-sweep.

## Snapshots to use

| Snapshot | Why |
| --- | --- |
| `01_<sub1>` -- `03_<sub3>` | Promote-time placement quality. Expect a high Uncategorized share -- judge whether the *non*-Uncategorized placements that did happen are coherent. |
| `04_post_sweep` | The main snapshot. Final taxonomy after all five sweep steps. Includes the `category_events` audit log. |

## Source-of-truth code (read before judging)

| Path | What to look for |
| --- | --- |
| `db/embeddings.py: find_best_category` | The promote-time placement function: how the per-category blended vector is computed, where `CATEGORY_COSINE_THRESHOLD` is applied, what happens when no category clears it. |
| `category_worker.run_sweep` | The full sweep skeleton: order of steps, what each one prefetches, where audit-log events are written. |
| `db/category_events.py: propose_uncategorized_events` and `propose_uncategorized_singleton_events` | Step 1 / 1b -- Uncategorized clustering and LLM review. Defines `MIN_SUB_CLUSTER_SIZE` use, candidate selection, and the LLM naming prompt. |
| `db/category_events.py: propose_split_events` | Step 2 -- when a category gets split. |
| `db/category_events.py: propose_delete_events` | Step 3 -- when a category gets deleted. |
| `db/category_events.py: propose_merge_events` | Step 4 -- when sibling categories get merged. |
| `db/category_events.py: propose_reroute_events` | Step 5 -- per-painpoint reroute. |

Verify the **active threshold values** with
`rg '^CATEGORY_COSINE_THRESHOLD|^MIN_SUB_CLUSTER_SIZE' --type py` and
quote them (with the symbol names) in the report.

## Where the evidence lives

| Source | What's in it |
| --- | --- |
| `dump.md` section 5 ("Category tree") | Recursive tree with direct + total painpoint counts. Pre-sweep snapshots show promote-time placement; snapshot 4 shows the final taxonomy. |
| `dump.md` section 6 ("Category worker audit log") | **Snapshot 4 only.** Every proposed sweep event with ACCEPT/REJECT, metric value, threshold, and reason. Read every line -- the reasons are the most informative thing in the whole snapshot. |
| `metrics.json` -> `categorization.{painpoints_total, in_uncategorized, in_seed_category, in_runtime_category, pct_uncategorized}` | Headline placement numbers. |
| `metrics.json` -> `category_events.<event_type>.{proposed, accepted}` | Per-step proposed / accepted counts (snapshot 4 only). |
| `db.queries.get_painpoints_by_category(name)` | Pull painpoints under any specific category to judge leaf coherence. |
| `inspect_db.render_category_tree()` | Programmatic version of dump.md section 5 -- use it to walk the tree if the rendered version is too long to scan. |
| `inspect_db.get_category_events(limit=None)` | All audit-log rows (no cap) when the dump's `AUDIT_LOG_LIMIT` truncates. |

## Citation rules

For every example you cite:

- **Category name with parent path** (e.g. `Dev tooling > AI coding agents`).
- **Snapshot label**.
- **Painpoint ids** under the category being discussed (at least 3 if
  judging coherence).
- For a sweep mistake, **event id** + the audit-log `reason` field
  (verbatim).
- A **one-sentence reviewer note** explaining what's right or wrong
  (theme drift, missed merge, premature delete, reroute miss, etc.).

## Output for `report.md`

A `## Dimension 4 -- Category assignment` section in the shape defined
in `00_protocol.md` section 6. Cover **both** promote-time placement
quality (snapshots 1-3) **and** post-sweep tree quality (snapshot 4),
and quote at least 3 audit-log events with reasons (mix accepts and
rejects). If a sweep step never fired (`proposed = 0` in
`category_events.<event_type>`), call it out -- it's a signal worth
flagging.
