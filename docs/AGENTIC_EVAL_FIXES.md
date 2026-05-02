# Agentic eval highest-value fixes

This doc distills recurring problems found across the April 2026
`evaluation/agentic_eval/runs/*/report.md` files. It focuses on fixes
that should materially improve downstream painpoint quality, category
quality, or evidence trustworthiness.

Reports reviewed:

- `evaluation/agentic_eval/runs/sideproject_indiehackers_entrepreneurridealong_20260420-005859/report.md`
- `evaluation/agentic_eval/runs/parenting_productivity_personalfinance_20260420-010622/report.md`
- `evaluation/agentic_eval/runs/teachers_sales_freelance_20260420-011443/report.md`
- `evaluation/agentic_eval/runs/cursor_localllama_notion_20260420-012037/report.md`

## 1. Enforce quote validity before inserting pendings

### Problem

`quoted_text` is not reliable enough as user-facing evidence. Reports
show quotes that are too long, paraphrased, stitched from multiple
phrases, numerically changed, or not present in the cited source. The
existing attribution repair only helps when the quote is an exact source
substring; otherwise the bad quote still ships.

The current "under 5 words" instruction also appears unnatural for the
model. Several bad examples are long because the model ignores the cap,
but several good examples are short clauses that need more than five
words to remain faithful. The stronger invariant is not word count; it is
that the quote is copied verbatim from one source span.

### Evidence from runs

- `parenting_productivity_personalfinance`: Dimension 1 says the
  under-5-word quote rule is "routinely ignored" and ~5-10% of sampled
  rows contain paraphrased or stitched quotes. Examples include pending
  `#86` with `"Drowning isn't like movies"` when the source says
  `"drowning looks nothing like the movies"`, and pending `#666` with a
  literal ellipsis joining two phrases.
- `teachers_sales_freelance`: Dimension 1 found ~8% paraphrased or
  blended `quoted_text`, including pending `#396` changing `25%` to
  `30%`, and pending `#38` blending title and selftext into a quote that
  exists nowhere as a substring.
- `cursor_localllama_notion`: Dimension 1 reports many anchors exceeding
  the under-5-word rule, including `#366` at roughly 17 words and `#346`
  at roughly 11 words.

### Fix

Add a parser-side validation step after extraction and before
`save_pending_painpoints_batch`:

- Reject or re-prompt any pending whose `quoted_text` is not an exact
  substring of the selected source.
- Change the prompt from "under 5 words" to "prefer a short phrase; a
  short clause is okay when needed, but keep it under one sentence."
- Enforce source faithfulness mechanically: no paraphrases, stitched
  fragments, or changed numbers.
- Sanitize titles before insert by stripping markup tags and collapsing
  whitespace.
- Emit metrics for `quote_not_found`, `quote_too_long`, and
  `title_sanitized` so eval runs can track the cost.

## 2. Filter non-pain content at extraction time

### Problem

The extractor still converts jokes, praise, show-and-tell, vendor
self-promotion, and thought-leadership posts into painpoints. These rows
then pollute dedup and category stages because they look semantically
similar to real product pains.

### Evidence from runs

- `sideproject_indiehackers_entrepreneurridealong`: Dimension 1 flags
  pending `#12` from a joke "Venmo me 1k refund" thread, pending `#314`
  from a "Tenant Management Portal" vendor pitch, and pending `#696`
  from a preachy "Most startups treat sales as an afterthought" post.
- `cursor_localllama_notion`: Dimension 1 flags pending `#182`, a meme
  / philosophical joke, pending `#386`, a praise comment reframed as an
  Obsidian pain, and pending `#183`, a finished-project brag reframed as
  a low-code pain.
- `teachers_sales_freelance`: Dimension 1 flags several r/teachers
  policy or current-events pendings whose product hook is thin, including
  school-shooting grief reframed as a mental-health product need.

### Fix

Tighten `EXTRACT_INSTRUCTIONS` with concrete anti-examples:

- Skip sarcasm, joke, meme, and gag threads even when a product-shaped
  sentence can be invented.
- Skip "drop your product" and "what are you building" comments unless a
  different user explicitly states a pain.
- Skip praise and show-and-tell unless the speaker names a friction.
- Skip thought-leadership or lessons-learned posts when the "pain" is
  only the author's thesis.

Prefer prompt anti-shots plus post-parse rejection reasons over threshold
changes here. This removes bad input before it becomes dedup noise.

## 3. Fix primary-as-extra source duplication

### Problem

When the LLM emits two near-duplicate observations from the same source
in one batch, the second observation can be inserted as an extra source
for the first pending, even when its `(post_id, comment_id)` matches the
primary source. This inflates `signal_count` without adding evidence.

### Evidence from runs

- `cursor_localllama_notion`: Dimension 2 identifies pending `#361` and
  pending `#409` with the primary source duplicated as an extra. The
  report says this affects 2 of 86 extras in `03_notion` and flows into
  promote-time signal count because the promote path reads the combined
  source view.

### Fix

In `db/painpoints.py:save_pending_painpoints_batch`, before inserting an
extra into `pending_painpoint_sources`, skip it when the extra source
matches the surviving pending's primary `(post_id, comment_id)`.

Add a regression test for the same source pair being emitted twice in one
batch.

## 4. Add missing seed taxonomy roots

### Problem

Several strong extraction runs fail categorization because the seed
taxonomy lacks obvious homes. The sweep then compensates with reroute and
split inheritance, producing wrong parents and dumping-ground runtime
categories.

### Evidence from runs

- `parenting_productivity_personalfinance`: Dimension 4 says
  r/personalfinance ends with 266 painpoints, 39.82%, in
  `Uncategorized` before sweep because `Fintech` is B2B-shaped. After
  sweep, 69 mixed finance painpoints land in `Fintech > Account Closures
  & Access`, including co-signing, inherited IRA, emergency fund, and
  HYSA topics.
- `teachers_sales_freelance`: Dimension 4 says teacher painpoints are
  forced through `Health & Lifestyle > Neurodivergence`, creating
  `Educational Accommodations and Support` under the wrong parent. It
  also flags `Freelance and Service Pricing Strategies` under `App
  Business`.
- `teachers_sales_freelance`: The report explicitly recommends adding
  `Education` and `Freelance & Services` roots to reduce
  `Uncategorized` and prevent wrong-root split inheritance.

### Fix

Add seed coverage for:

- `Education`, including classroom management, accommodations,
  assessment, school administration, student behavior, and teacher
  workload.
- `Freelance & Services`, including contracts, scope, client
  communication, pricing, payment terms, retainers, and lead generation.
- `Personal Finance`, including debt and credit, taxes and inheritance,
  savings and emergency funds, scams and fraud recovery, housing and
  mortgages, and family financial support.

Re-run the same eval triples after seeding to verify lower promote-time
`Uncategorized` and fewer wrong-root runtime children.

## 5. Let painpoint dedup cross sibling categories

### Problem

Sweep-time `painpoint_merge` only considers pairs already in the same
category. Duplicates split across sibling categories are structurally
invisible unless reroute first colocates them, and reroute often does not.

### Evidence from runs

- `parenting_productivity_personalfinance`: Dimension 3 flags a
  wake-up/morning-routine cluster split across `Habits & Discipline`,
  `Sleep & Recovery`, and `Wake-Up and Morning Routines`. The report
  says this is outside sweep reach because the proposer iterates
  per-category.
- `cursor_localllama_notion`: Dimension 3 and Dimension 4 flag three
  documentation-themed AI/ML runtime siblings whose painpoints cannot
  merge until the categories merge first.
- `sideproject_indiehackers_entrepreneurridealong`: Dimension 3 says
  duplicates sitting in different categories cannot be collapsed until a
  later sweep because reroute runs after `painpoint_merge`.

### Fix

Extend `propose_painpoint_merge_events` to scan a bounded cross-category
set:

- Same root, sibling categories.
- Runtime children under the same seed parent.
- Optionally category pairs with category cosine above a lower
  candidate floor, then LLM-confirm the painpoint pair.

Keep this narrower than global all-pairs dedup to control cost and avoid
cross-domain false positives.

## 6. Add an LLM-confirmed runtime category merge path

### Problem

`merge_categories` is silent in every reviewed report despite obvious
duplicate or near-duplicate runtime categories. The current thresholds
appear calibrated for seed categories, not short LLM-generated runtime
category names and descriptions.

### Evidence from runs

- `sideproject_indiehackers_entrepreneurridealong`: Dimension 4 reports
  `merge_categories proposed=0` while obvious automation-related runtime
  categories sit around cosine `0.51-0.61`, below
  `MERGE_CATEGORY_THRESHOLD = 0.80`.
- `cursor_localllama_notion`: Dimension 4 reports three
  `Documentation*` siblings under `AI/ML`, each with 1-2 members, while
  `merge_categories proposed=0`.
- `teachers_sales_freelance`: Dimension 4 reports
  `Sales Leadership and Quotas` as a duplicate root of `Business &
  Sales`, but `merge_categories` still proposed `0`.
- `parenting_productivity_personalfinance`: Dimension 4 lists sibling
  near-duplicates such as `Wake-Up and Morning Routines` vs
  `Sleep & Recovery`, with merge still `0/0`.

### Fix

Introduce a separate runtime merge candidate path:

- Use a lower candidate threshold for runtime-vs-runtime and
  runtime-vs-seed pairs, roughly `0.55-0.65`.
- Gate candidates with an LLM confirmation prompt that checks whether the
  categories serve the same routing purpose.
- Compute runtime category vectors from member painpoints as well as the
  generated name and description.
- Run this immediately after `add_category_split`, before
  `painpoint_merge`.

## 7. Tighten painpoint merge confirmation and headline maintenance

### Problem

Every reviewed sweep accepted 100% of `painpoint_merge` proposals. Many
accepted merges are correct, but the zero-reject pattern plus several
false positives suggest the LLM confirmation prompt is too permissive.
Separately, when a painpoint absorbs several others, the survivor keeps
its original title even when the cluster broadens.

### Evidence from runs

- `sideproject_indiehackers_entrepreneurridealong`: Sweep accepted
  `49/49` painpoint merges. Dimension 3 flags weak accepted merges such
  as unrelated monetization problems merged under one
  `App Monetization & Pricing` painpoint.
- `teachers_sales_freelance`: Sweep accepted `14/14`. Dimension 3 flags
  event `#15`, where "Need to stay visible on social media" absorbed
  "Burnout from constant content creation", described as inverse
  direction.
- `cursor_localllama_notion`: Sweep accepted `29/29`. Dimension 3 flags
  user-side model switching merged with builder-side open-source
  multi-agent framework needs.
- `parenting_productivity_personalfinance`: Sweep accepted `35/35`, with
  no rejection signal available to calibrate candidate quality.
- `cursor_localllama_notion`: Dimension 3 says painpoint `#23` grew into
  a broader pricing / quota / billing cluster but kept the stale title
  `"Pricing ambiguity after changes"`.

### Fix

Update `decide_painpoint_merge` to reject when:

- The pair is only in the same broad topic, not the same concrete pain.
- One item is a symptom and the other is a proposed solution.
- User roles differ materially, such as end-user complaint vs builder
  framework request.
- Direction or polarity is inverted, such as "avoid AI" vs "use
  AI-powered tool".

After a painpoint absorbs more than one loser, optionally regenerate or
validate the survivor title and description from the combined sources.

## 8. Reorder sweep passes around reroute

### Problem

Several sweep steps need the post-reroute state, but reroute currently
happens late. This means category delete and painpoint dedup can miss
conditions created by reroute in the same sweep.

### Evidence from runs

- `sideproject_indiehackers_entrepreneurridealong`: Dimension 4 reports
  an empty ghost category, `App Business > Community and Psychological
  Support`, likely because delete ran before late reroute drained it.
- `sideproject_indiehackers_entrepreneurridealong`: Dimension 3 says
  cross-category duplicates will not collapse until the next sweep
  because `painpoint_merge` runs before reroute.
- `cursor_localllama_notion`: Dimension 4 reports reroute accepts 106
  events and drains `Uncategorized`, while delete and merge both proposed
  `0`; this makes the run's cleanup highly dependent on pass order.

### Fix

Change `run_sweep` ordering or add a second pass:

- Run reroute before delete.
- Run `painpoint_merge` after reroute, or run it both before and after
  reroute.
- Run runtime category merge after split and before painpoint merge.
- Run delete after all reroutes and merges that can drain categories.

If full reordering is risky, add a second lightweight post-reroute cleanup
phase for category merge, painpoint merge, and delete.

## 9. Make low-confidence reroute auditable

### Problem

Reroute is doing most of the category rescue work. Many events are good,
but low-margin reroutes appear near the threshold and contribute to mixed
runtime categories. Because `Uncategorized` has no meaningful current
centroid, almost any existing category can win by the margin gate.

### Evidence from runs

- `parenting_productivity_personalfinance`: Dimension 4 reports
  `reroute_painpoint` accepted `285/285`, while Uncategorized clustering,
  LLM review, delete, and merge all proposed `0`. The report says reroute
  stuffed consumer-finance painpoints into the closest existing runtime
  leaf.
- `teachers_sales_freelance`: Dimension 4 flags reroute events with
  margins near `0.10-0.17`, contributing to cross-vertical leaks into
  generic `Productivity & Workflow` categories.
- `sideproject_indiehackers_entrepreneurridealong`: Dimension 4 compares
  a strong reroute at margin `0.503` with a weak one at `0.124`, and
  links the weak tail to misroutes.
- `cursor_localllama_notion`: Dimension 4 flags event `#140` at margin
  `0.101`, one tick above the floor.

### Fix

Keep reroute, but make weak moves safer:

- Raise `REROUTE_MARGIN` for non-Uncategorized moves, or require LLM
  confirmation below a margin such as `0.15`.
- Track low-margin reroutes in metrics.
- For `Uncategorized`, require either a strong absolute target cosine or
  a category-specific LLM placement check when the target is a runtime
  category minted in the same sweep.
- Consider leaving low-confidence painpoints in `Uncategorized` rather
  than forcing `pct_uncategorized` to zero.

## Suggested implementation order

1. Fix primary-as-extra duplication.
2. Enforce quote substring validation and replace "under 5 words" with
   "under one sentence" in the extractor prompt.
3. Tighten extraction skip rules for jokes, praise, self-promo, and
   thought-leadership.
4. Add missing seed taxonomy roots.
5. Add cross-sibling painpoint dedup.
6. Add runtime category merge with LLM confirmation.
7. Reorder or repeat sweep cleanup after reroute.
8. Tighten painpoint merge confirmation and add optional headline refresh.
9. Add low-margin reroute metrics and stricter handling.
