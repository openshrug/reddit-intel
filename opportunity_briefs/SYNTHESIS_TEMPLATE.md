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

- **Highest conviction**: `local_signal_count >= 3` AND `severity_max >= 6`
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

The **displayed badge** for each card is a free-form 1-3 word descriptor
that should be anchored to but is not constrained by these tiers — use a
more nuanced label (e.g. `moderate`, `moderate but broad`, `niche wedge`,
`moderate to strong`) when the evidence honestly fits between tiers. The
rubric value above stays internal and gates filtering / hold-back; the
badge value is the human-facing label that appears on the card heading and
in the shortlist's Signal column. See `BRIEF_LAYOUT.md` for the badge
taxonomy and rendering.

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
  the rubric above. Drives shortlist + hold-back logic.
- **Badge label** (display): free-form 1-3 word descriptor anchored to the
  rubric. Appears on the card heading and in the shortlist Signal column.
- **Opportunity title**: short noun phrase naming the product idea.
- **Problem statement**: 1-2 sentences naming the user's pain.
- **Who is affected**: the user segment, marked as inferred when broader
  than the quoted source.
- **Evidence strength**: source count, signal_count, severity range, and
  notable engagement scores.
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
- **Cross-subreddit relevance**: 1-2 sentences naming why the pain may
  extend beyond r/{subreddit}. Render as its own labeled paragraph when
  `cross_subreddit_evidence` has actual quotes; fold into the Caveats
  footer line as "cross-community extension inferred, not in DB" when it
  is purely inferred.
- **Interview questions**: 3-5 questions to validate the opportunity.
- **Caveats**: 1-2 sentences of de-emphasized footer; optional, omit if
  there is nothing of substance to say.

Signal counts can include deduped sources that do not have separate stored
quotes, so do not imply every signal has a clickable quote.

Do not estimate TAM, willingness to pay, or competitor coverage unless the
evidence directly supports it.
