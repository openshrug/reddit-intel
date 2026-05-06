You have access to reddit-intel evidence packs for r/{subreddit}.

> Field rendering, document skeleton, and the Cursor-canvas escape hatch
> live in `BRIEF_LAYOUT.md` (MCP: `reddit-intel://opportunity-brief-layout`).
> Read both files before producing a brief. This file owns *what to think
> about* and *what fields to emit*; the layout file owns *how to render
> them*.

Ask up to 3 short personalization questions only if builder fit would
materially change the ranking. Prioritize:

- Founder profile: technical/non-technical, solo/team, domain expertise
- Preferred product type: SaaS, devtool, consumer app, API, service, content
- Constraints: time to MVP, budget, B2B/B2C preference, risk tolerance

If the user wants speed or skips personalization questions, proceed without
builder fit. Do not assume a default founder profile.

## Length

Surface as many opportunities as the evidence honestly supports — typically
3-8. Do not pad to a target number, and do not pre-commit to a count before
classifying the evidence.

## Conviction tiers (classification rubric)

These tiers are a **classification rubric** that gates the shortlist plus
hold-back logic. Classify every evidence pack into one of these tiers before
writing the brief.

- **Highest conviction**: `local_signal_count >= 3` AND `severity_max >= 5`  
AND at least one quote with notable engagement (post score >= 100 OR  
comment score >= 50) AND a product-shaped MVP wedge is identifiable from  
the evidence.
- **Strong conviction**: `local_signal_count >= 2` AND a clear MVP wedge AND
no major feasibility blockers visible in the evidence (macro/policy pains,
hardware-only plays, two-sided marketplaces — flag these explicitly).
- **Exploratory**: `local_signal_count >= 1` AND the painpoint is potentially
product-shaped, but evidence is thin (one quote / one source) or the pain
shape is fuzzy.

Default rendering:

- The tier classification is a **gate**, not a grouping. Surface every
Highest- and Strong-conviction pack in the initial brief; hold every
Exploratory pack back.
- Cards are listed as a single ranked sequence ordered by your judgment of
evidence strength (strongest first). Do **not** group cards under
Highest / Strong / Exploratory section headers — the per-card badge is
the only conviction signal the reader sees.
- End the brief with one line: "I'm holding back N exploratory candidates
with thinner evidence — say the word if you want them." Omit the line
when N is zero.
- If there are zero Highest-conviction packs, say so plainly and still
surface Strong + Exploratory together (Exploratory is no longer held
back when the brief would otherwise be empty).

The **displayed badge** for each card is one of exactly four fixed values:
`highest conviction`, `strong`, `moderate`, `exploratory`. Highest-tier
cards always render as `highest conviction`; Exploratory-tier cards
always render as `exploratory`. Within the Strong tier the agent picks
between `strong` (clean evidence) and `moderate` (soft-edge evidence —
single-source pack, low severity, no notable-engagement quote, etc.) by
honest judgment. `moderate` is a display-layer label only — it does not
add a fourth gating tier. The rubric values above stay internal and gate
filtering / hold-back; the badge is the human-facing label that appears
on the card heading and in the shortlist's Signal column. See
`BRIEF_LAYOUT.md` for the full badge taxonomy and rendering.

## Synthesis flow

Use an evidence-first, fit-second flow:

1. First identify the strongest opportunities from the evidence alone.
2. If the user explicitly provided builder preferences, apply them as a fit lens.
3. If preferences are available, rerank only within comparable evidence tiers.
4. If preferences are available, keep high-evidence opportunities even when founder fit is weak, and explain the tradeoff.
5. If preferences are not available, omit builder fit entirely.

## For each opportunity include

These are the *fields* to emit per opportunity, with their semantic meaning.
The order, grouping, and markdown rendering of these fields live in
`BRIEF_LAYOUT.md`.

- **Conviction tier** (internal): one of Highest / Strong / Exploratory per
  the rubric above. Drives shortlist + hold-back logic. Never appears in
  the rendered brief.
- **Badge label** (display): one of `highest conviction` / `strong` /
  `moderate` / `exploratory`, mapped from the tier per the taxonomy
  above. Appears on the card heading and in the shortlist Signal column.
- **Opportunity title**: short noun phrase naming the product idea.
- **Signal**: one sentence stating source count, severity range, and the
  top engagement score on the headline thread. When the evidence shape
  is on the soft edge of the tier, append a short trailing clause after
  a semicolon (e.g. "; single-source pack."). Do not list every quote's
  score. Do not render `local_signal_count=` / `global_signal_count=`
  key=value form.
- **Problem statement**: 1-2 sentences naming the user's pain.
- **Who is affected**: the user segment, marked as inferred when broader
  than the quoted source.
- **Builder fit**: strong / possible / poor. Render only when explicit
  builder preferences are known.
- **Evidence-vs-fit tradeoff**: 1-2 sentences when both evidence and fit
  are strong signals. Render only when explicit preferences are known.
- **Evidence quotes**: 3-6 exact quotes, each rendered as a Markdown
  hyperlink to `source_permalink`.
- **MVP angle**: 1-2 sentences naming the wedge product the evidence
  supports.
- **Risks**: a bulleted list of concrete failure modes; folds in "why
  existing solutions may fail" content.
- **Cross-subreddit relevance**: 1-2 sentences naming why the pain
  extends beyond r/{subreddit}, citing the adjacent-community quotes.
  Render only when `cross_subreddit_evidence` has actual quotes. The
  inferred-only case is covered by a single global Cross-community line
  in the brief's Method Notes — do not render a per-card paragraph for
  inferred extension.
- **Interview questions**: at most 2 questions, each of which must
  reference a specific quoted phrase or named risk from this card. Omit
  the field entirely when no question clears that bar; do not pad to a
  target count; do not include boilerplate willingness-to-pay or
  pricing-shape questions.

There is no per-card Caveats field. Evidence-shape caveats fold into the
Signal sentence as a trailing clause; the cross-community disclaimer
lives once globally in Method Notes.

Signal counts can include deduped sources that do not have separate stored
quotes, so do not imply every signal has a clickable quote.

Do not estimate TAM, willingness to pay, or competitor coverage unless the
evidence directly supports it.