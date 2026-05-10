# Opportunity Brief Spec

This file is the canonical synthesis + rendering spec for opportunity
briefs. Pair it with `AGENTS.md` (workflow / tool-use flow); together
they fully specify how to turn `get_opportunity_evidence` output into a
brief. Briefs are produced as Markdown — the persisted
`runs/.../brief.md` file is the canonical artifact, diffable and
reproducible.

## Personalization questions

Ask up to 3 short personalization questions only if builder fit would
materially change the ranking. Prioritize:

- Founder profile: technical/non-technical, solo/team, domain expertise
- Preferred product type: SaaS, devtool, consumer app, API, service, content
- Constraints: time to MVP, budget, B2B/B2C preference, risk tolerance

If the user wants speed or skips personalization questions, proceed
without builder fit. Do not assume a default founder profile.

## Length & hold-back logic

Surface as many opportunities as the evidence honestly supports —
typically 3-8. Do not pad to a target number, and do not pre-commit to
a count before classifying the evidence.

Conviction-tier classification (next section) is a **gate**, not a
grouping. Surface every Highest- and Strong-conviction pack in the
initial brief; hold every Exploratory pack back.

Cards are listed as a single ranked sequence ordered by the agent's
judgment of evidence strength (strongest first). Do **not** group
cards under Highest / Strong / Exploratory section headers — the
per-card badge is the only conviction signal the reader sees.

End the brief with one line: "I'm holding back N exploratory
candidates with thinner evidence — say the word if you want them."
Omit the line when N is zero.

If there are zero Highest-conviction packs, say so plainly and still
surface Strong + Exploratory together (Exploratory is no longer held
back when the brief would otherwise be empty).

## Conviction-tier rubric (internal classification)

Classify every evidence pack into one of these tiers before writing
the brief. The tiers are internal — they gate the shortlist plus
hold-back logic and never appear as headings in the rendered brief.

- **Highest conviction**: `local_signal_count >= 3` AND
`severity_max >= 5` AND at least one quote with notable engagement
(post score >= 100 OR comment score >= 50) AND a product-shaped MVP
wedge is identifiable from the evidence.
- **Strong conviction**: `local_signal_count >= 2` AND a clear MVP
wedge AND no major feasibility blockers visible in the evidence
(macro/policy pains, hardware-only plays, two-sided marketplaces —
flag these explicitly).
- **Exploratory**: `local_signal_count >= 1` AND the painpoint is
potentially product-shaped, but evidence is thin (one quote / one
source) or the pain shape is fuzzy.

## Conviction badge taxonomy (display)

The badge displayed on each card heading and in the shortlist Signal
column is one of exactly four fixed values:

- `highest conviction` — used for every card classified Highest tier.
- `strong` — used for a Strong-tier card with clean evidence.
- `moderate` — used for a Strong-tier card on the soft edge of the
tier (single-source pack, low severity, no quote with notable
engagement, or similar). The agent picks `strong` vs. `moderate` by
judgment within the Strong tier; the constraint is on vocabulary,
not on the call.
- `exploratory` — used for every card classified Exploratory tier.
Only appears in the brief when Exploratory items are surfaced
(zero-Highest fallback, or the user explicitly asked for breadth).

`moderate` is a display-layer label only — it does not add a fourth
gating tier. The rubric stays at three tiers (Highest / Strong /
Exploratory); `moderate` does not gate hold-back.

## Synthesis flow

Use an evidence-first, fit-second flow:

1. First identify the strongest opportunities from the evidence alone.
2. If the user explicitly provided builder preferences, apply them as
  a fit lens.
3. If preferences are available, rerank only within comparable
  evidence tiers.
4. If preferences are available, keep high-evidence opportunities even
  when founder fit is weak, and explain the tradeoff.
5. If preferences are not available, omit builder fit entirely.

## Document skeleton

Render the brief as Markdown with the following parts in order.

1. **Title** — `# r/{subreddit} — Opportunity Brief`.
2. **Subtitle** — a 1-2 sentence paragraph immediately under the title
  stating how the data was sourced (which subreddit, fresh scrape
   vs. existing DB, evidence pack tool).
3. **Stats strip** — a 4-column markdown table sourced from the MCP
  stats and summary tools:
   `| Posts scraped | Comments persisted | Promoted signals | Painpoints |`.
   Render zero or em-dash for any metric not computable in the
   current run rather than dropping it.
4. **Headline insight** — a single `> blockquote`, 2-3 sentences max,
  synthesizing the macro pattern across all surfaced opportunities.
   No internal paragraph break. Do not include synthesizer-internal
   commentary (e.g. "the loudest threads are deliberately held back")
   — that belongs in Method Notes.
5. **Opportunity shortlist** — a 4-column markdown table covering
  every opportunity that will be detailed below:
   `| Rank | Opportunity | Primary Segment | Signal |`. The Signal
   column uses one of the four fixed badge values from the taxonomy
   above, matching the badge on the corresponding card heading.
6. **Detailed opportunities** — `## Detailed Opportunities` followed
  by one card per opportunity, separated by a `---` rule. Cards
   follow the per-card spec below.
7. **Method Notes** — `## Method Notes` followed by up to three
  stacked `**Label.`**  paragraphs (pure markdown, no two-column
   HTML):
  - **MCP DB.** db path plus before/after stats.
  - **Signal Interpretation.** the standard "signal strength is
  directional founder research, not TAM sizing" disclaimer.
  - **Cross-community.** rendered only when the run is
  single-subreddit and at least one card relies on inferred
  cross-community extension: "Single-subreddit run;
  cross-community extension is inferred per card." Cross-community
  wording never repeats per card.

## Per-card spec

Each card uses the following ordered sections. The card has **no
Caveats footer** — evidence-shape caveats fold into the Signal
sentence as a trailing clause, and the cross-community disclaimer
lives once globally in Method Notes.

### Heading

**Render:** `### N. {Title} — *{badge}`* where `{badge}` is one of
`highest conviction` / `strong` / `moderate` / `exploratory` per the
badge taxonomy above. The italic asterisks are required.

**Example:** `### 1. Quoting workflow consolidator — *strong`*

### Signal

**Render:** `**Signal.** <one sentence>`

**Content:** Source count, severity range, and the top engagement
score (post score or comment score) on the headline thread. When the
evidence shape is on the soft edge of the tier — single-source pack,
two-of-N signals from the same thread, low severity, or no quote with
notable engagement — append a short trailing clause after a semicolon.
Do not render `local_signal_count=` / `global_signal_count=` key=value
form; do not list every quote's score.

**Examples:**

- Clean: `**Signal.** 4 sources, severity 5-6, top thread score 2,751.`
- Soft-edge: `**Signal.** 2 sources, severity 4-5, top thread score 312; single-source pack.`

### Problem

**Render:** `**Problem.** <1-2 sentence paragraph>`

**Content:** Describe the user's concrete pain.

### User Segment

**Render:** `**User Segment.** <1-2 sentence paragraph>`

**Content:** Name who is affected. Mark the segment as inferred when
it is broader than the quoted source.

### Builder Fit (conditional)

**Render:** `**Builder Fit ({profile}): {strong|possible|poor}.** <1-2 sentence paragraph>`

**Content:** Explain why this is or isn't a good fit for the supplied
builder profile.

**Conditional:** Render only when explicit builder preferences were
supplied; omit entirely otherwise.

### Evidence-vs-Fit Tradeoff (conditional)

**Render:** `**Evidence-vs-Fit Tradeoff.** <1-2 sentence paragraph>`

**Content:** Call out the tradeoff between evidence strength and
builder fit when both signals are strong.

**Conditional:** Same gating as Builder Fit — render only when
explicit preferences were supplied.

### Evidence Quotes

**Render:** `#### Evidence Quotes` sub-heading, then 3-6 quotes (more
is better, up to 6). Each quote on its own line, separated from the
previous by a blank line. Each quote follows this exact format, where
the link target is the quote's `source_permalink`:

```markdown
> "quote text" — [Source type (score N): short label](https://reddit.com/...)
```

**Example:**

```markdown
> "I spend three hours every Friday reconciling invoices by hand" — [Post (score 844): weekly reconciliation pain](https://reddit.com/r/example/comments/abc123/example_post/)
```

### MVP Angle

**Render:** `**MVP Angle.** <1-2 sentence paragraph>`

**Content:** Name the wedge product the evidence supports — the
smallest thing that would test willingness to pay for this pain.

### Risks

**Render:** `#### Risks` sub-heading plus a bulleted list of concrete
failure modes. Folds in the prior "why existing solutions may fail"
content as risk bullets.

**Content:** Each bullet is a concrete, specific failure mode named
from the evidence — not generic startup risk boilerplate.

**Examples:**

- Good: `Buyer is the operations lead, not the accountant — sales motion is two-step instead of one-step.`
- Good: `Existing tools (QuickBooks, Bill.com) already cover the end-of-month flow; wedge has to be the weekly cadence.`
- Banned (too generic): `Market may not pay for this.` — every
opportunity has this risk; surfacing it teaches nothing.

### Cross-subreddit relevance (conditional)

**Render:** `**Cross-subreddit relevance.** <1-2 sentence paragraph>`

**Content:** Name why the pain extends beyond the requested  
subreddit, citing the adjacent-community quotes with links in the same format as Evidence Quotes from  
`cross_subreddit_evidence`.

**Conditional:** Render only when `cross_subreddit_evidence` has
actual quotes from adjacent communities. The purely inferred case is
covered once globally by the Cross-community line in Method Notes —
never render a per-card paragraph that just says "extension inferred,
not in DB."

### Interview Questions (conditional)

**Render:** `#### Interview Questions` sub-heading plus a numbered
list of at most 2 questions.

**Content:** Each question must reference a specific quoted phrase or
a named risk from this card. Do not pad to a target count; do not
include boilerplate willingness-to-pay or pricing-shape questions.

**Conditional:** Omit the entire section when no question clears that
bar.

**Examples:**

- Good (cites a phrase): `You said reconciliation takes "three hours every Friday" — walk me through the version of that workflow that takes 30 minutes.`
- Good (probes a named risk): `If the buyer ends up being the operations lead instead of the accountant, what changes in how this would land in your team?`
- Banned (boilerplate): `How much would you pay for this?` — does not
cite a quoted phrase or a named risk; surfaces nothing the brief
couldn't already infer.

## Caveats

- Signal counts can include deduped sources that do not have separate
stored quotes, so quote counts may be lower than signal counts; do
not imply every signal has a clickable quote.
- Do not estimate TAM, willingness to pay, or competitor coverage
unless the evidence directly supports it.

