# Synthesis -- assembling `report.md`

> Read `00_protocol.md` first. This file describes the parent
> evaluator's job *after* the four per-dimension sections have been
> written. The per-dimension files (`10_*` -- `40_*`) own the
> per-section content; this file owns the report skeleton, the summary
> table, and the cross-dimensional recommendations.

## When to read this file

After every dimension has been evaluated and the four sections are
ready. The default execution model (see `00_protocol.md` section 2)
fans out one sub-agent per dimension in parallel; in that case, by
the time you reach this file you should have:

- four `META:` header lines (one per sub-agent), already parsed into
  `(dimension, score, verdict, headline)` tuples for the summary
  table;
- four markdown bodies, each starting with `## Dimension N -- ...`,
  ready to paste into the report in dimension order.

If you ran the evaluation sequentially as a single agent, you have
the same four sections; you just produced them yourself and there is
no `META:` line to parse (read the `## Dimension N` heading + the
verdict/score/headline you wrote).

This file is the last step before writing `report.md` to the run
directory.

## Cross-dimensional recommendations

Compound failure modes -- the kind that no single dimension's evaluator
can see on its own -- live here. Walk through each pair below and look
for evidence in the per-dimension findings. **Skip any pair you have
no evidence for; do not invent compound failures.**

| Pair | What to look for | Recommendation shape |
| --- | --- | --- |
| Dim 1 (extraction) x Dim 2 (pending dedup) | Is the extractor producing many near-identical pendings the dedup path didn't catch? Or the opposite -- dedup over-collapses real distinct pains the extractor correctly separated? | Adjust the extractor prompt vs. tweak `PENDING_MERGE_THRESHOLD`. |
| Dim 2 (pending dedup) x Dim 3 (pending->painpoint merge) | Pendings that *should* have collapsed at extract time but did at promote time instead -- ok, just slower. Pendings that should have collapsed and did at *neither* stage -- the gap is real. | Bridge the gap (raise `PENDING_MERGE_THRESHOLD`, lower `MERGE_COSINE_THRESHOLD`, or add a sweep-time pending dedup). |
| Dim 3 (pending->painpoint merge) x Dim 4 (categories) | Are sibling painpoints under the *same* category that should have merged? -> sweep `painpoint_merge` is under-firing. Are merged painpoints accumulating under wrong categories? -> reroute is over- or under-firing. | Tune `painpoint_merge` candidate selection vs. tune reroute thresholds. |
| Dim 1 (extraction) x Dim 4 (categories) | Does the extractor produce pendings that consistently embed close to a wrong category? (If so, the embedding model + the category seed prompt are mis-aligned.) | Re-seed categories or include category labels in the extraction prompt. |
| Dim 2/3 (merging) x Dim 4 (categories) | Is `painpoint_merge` blocked because near-duplicates ended up in different categories? (`propose_painpoint_merge_events` only considers same-category pairs.) | Cross-category dedup, or a reroute pre-pass. |

For each compound failure you do find evidence for, write a single
recommendation that names the **two** dimensions involved, the
**evidence ids** from each (painpoint / pending / event), and the
**suggested change** (either a code path or a threshold).

## Summary table

```markdown
| Dimension | Score | Verdict | Headline finding |
| --- | :---: | --- | --- |
| 1. Extraction | x/5 | pass/mixed/fail | ... (one line, references one example id) |
| 2. Pending dedup | x/5 | pass/mixed/fail | ... |
| 3. Pending->painpoint merge | x/5 | pass/mixed/fail | ... |
| 4. Category assignment | x/5 | pass/mixed/fail | ... |
```

The headline finding for each row should be the most decision-relevant
single sentence from that dimension's section. Avoid restating the
verdict ("worked well") -- name the concrete pattern ("pricing-only
pendings leak through at ~10% of extraction volume").

## Final report structure

Write `report.md` **inside the run directory** (alongside `00_clean/`
... `04_post_sweep/`):

```markdown
# Pipeline quality report

**Run dir**: `quality_eval/runs/<run name>/`
**Subreddits**: <list>
**Timestamp**: <from the run dir name>

## Summary table

<the table above>

## Dimension 1 -- Pending painpoint extraction

### Numbers
... pull from metrics.json ...

### Examples
- **Pending #<id> (good)**: <title> -- "<source quote>" -- <one-line note>
- **Pending #<id> (bad: hallucinated)**: ...
- **Pending #<id> (bad: pricing-only)**: ...
- (5 examples minimum, mix of good and bad)

### Failure modes
- ...

## Dimension 2 -- Pending dedup
(same shape)

## Dimension 3 -- Pending -> painpoint merge
(same shape; cite at least one good and one bad multi-source painpoint)

## Dimension 4 -- Category assignment
(cover both promote-time placement quality from snap 1-3 AND post-sweep
tree quality from snap 4; quote at least 3 audit-log events with
reasons)

## Recommendations

### Per-dimension threshold tweaks
- (Optional) Threshold tweaks suggested by single-dimension failure modes.

### Cross-dimensional findings
- One bullet per compound failure mode you found evidence for, naming
  the two dimensions, the evidence ids, and the suggested change.
```

## Final checklist before saving

Run through this list once before writing the file:

- [ ] Every example carries a real id from the snapshot.
- [ ] Every quoted string was copy-pasted from `dump.md` or fetched
      via the helpers (no paraphrasing).
- [ ] Every dimension cites at least one snapshot label so the
      reviewer can re-open the source row.
- [ ] Threshold values cited with the symbol name (e.g.
      ``PENDING_MERGE_THRESHOLD = 0.65``), not naked numbers.
- [ ] Any dimension that couldn't be evaluated is called out
      explicitly with a reason (not silently skipped).
- [ ] The Recommendations section either lists concrete tweaks with
      evidence, or says "no actionable findings -- see failure modes
      per dimension".
