# Opportunity Brief Layout

This file owns the **rendering** of an opportunity brief. The synthesis
rules (conviction-tier rubric, evidence rules, per-opportunity field
semantics) live in `SYNTHESIS_TEMPLATE.md`. Read both before producing a
brief.

Briefs are produced as Markdown. The persisted `runs/.../brief.md` file
is the canonical artifact — diffable and reproducible.

## Document parts (in order)

1. **Title** — `# r/{subreddit} — Opportunity Brief`.
2. **Subtitle** — a 1-2 sentence paragraph immediately under the title
   stating how the data was sourced (which subreddit, fresh scrape vs.
   existing DB, evidence pack tool).
3. **Stats strip** — a 4-column markdown table sourced from the MCP
   stats and summary tools:
   `| Posts scraped | Comments persisted | Promoted signals | Painpoints |`.
   Render zero or em-dash for any metric not computable in the current
   run rather than dropping it.
4. **Headline insight** — a single `> blockquote`, 2-3 sentences max,
   synthesizing the macro pattern across all surfaced opportunities. No
   internal paragraph break. Do not include synthesizer-internal
   commentary (e.g. "the loudest threads are deliberately held back") —
   that belongs in Method Notes.
5. **Opportunity shortlist** — a 4-column markdown table covering every
   opportunity that will be detailed below:
   `| Rank | Opportunity | Primary Segment | Signal |`. The Signal
   column uses one of the four fixed badge values from the taxonomy
   below, matching the badge on the corresponding card heading.
6. **Detailed opportunities** — `## Detailed Opportunities` followed by
   one card per opportunity, separated by `---`. Cards are listed as a
   single ranked sequence ordered by the agent's judgment of evidence
   strength (strongest first). Do not group cards under conviction-tier
   sub-headers; the per-card badge is the only conviction signal the
   reader sees. The tier classification in `SYNTHESIS_TEMPLATE.md` still
   gates surface vs. hold-back, but it does not dictate a visible
   grouping.
7. **Method Notes** — `## Method Notes` followed by up to three stacked
   `**Label.**` paragraphs (pure markdown, no two-column HTML):
   - **MCP DB.** db path plus before/after stats.
   - **Signal Interpretation.** the standard "signal strength is
     directional founder research, not TAM sizing" disclaimer.
   - **Cross-community.** rendered only when the run is single-subreddit
     and at least one card relies on inferred cross-community extension:
     "Single-subreddit run; cross-community extension is inferred per
     card." Cross-community wording never repeats per card.

## Per-opportunity card

Each card uses the following ordered sections.

1. **Heading** — `### N. {Title} — *{badge}*`. The italic badge suffix
   is one of `highest conviction` / `strong` / `moderate` /
   `exploratory` (see Conviction badge taxonomy below).
2. **Signal** — `**Signal.** ...` — a single sentence stating source
   count, severity range, and the top engagement score (post score or
   comment score) on the headline thread. When the evidence shape is on
   the soft edge (single-source pack, two-of-N signals from the same
   thread, etc.), append a short trailing clause after a semicolon, e.g.
   `; single-source pack.`. Do not render `local_signal_count=` /
   `global_signal_count=` key=value form; do not list every quote's
   score.
3. **Problem** — `**Problem.** ...` — a 1-2 sentence paragraph describing
   the user's pain.
4. **User Segment** — `**User Segment.** ...` — a 1-2 sentence paragraph
   naming who is affected.
5. **Builder Fit** — `**Builder Fit ({profile}): {strong|possible|poor}.** ...`
   — a 1-2 sentence paragraph. *Conditional:* render only when explicit
   builder preferences were supplied.
6. **Evidence-vs-Fit Tradeoff** — `**Evidence-vs-Fit Tradeoff.** ...` —
   a 1-2 sentence paragraph. *Conditional:* same gating as Builder Fit.
7. **Evidence Quotes** — `#### Evidence Quotes` sub-heading, then 3-6
   quotes. Each quote on its own line as
   `> "quote text" — [Post (score N): short label](https://reddit.com/...)`.
8. **MVP Angle** — `**MVP Angle.** ...` — a 1-2 sentence paragraph naming
   the wedge product the evidence supports.
9. **Risks** — `#### Risks` sub-heading plus a bulleted list of concrete
   failure modes. Folds in the prior "why existing solutions may fail"
   content as risk bullets.
10. **Cross-subreddit relevance** — `**Cross-subreddit relevance.** ...`
    — a 1-2 sentence paragraph naming why the pain extends beyond the
    requested subreddit, citing the adjacent-community quotes.
    *Conditional:* render only when `cross_subreddit_evidence` has
    actual quotes from adjacent communities. The purely inferred case is
    covered by the global Cross-community line in Method Notes — never
    render a per-card paragraph that just says "extension inferred, not
    in DB."
11. **Interview Questions** — `#### Interview Questions` sub-heading
    plus a numbered list of at most 2 questions, each of which must
    reference a specific quoted phrase or named risk from this card.
    *Conditional:* omit the entire section when no question clears that
    bar. Do not pad to a target count; do not include boilerplate
    willingness-to-pay or pricing-shape questions.

The card has no Caveats footer. Evidence-shape caveats fold into the
Signal sentence as a trailing clause; the cross-community disclaimer
lives once globally in Method Notes.

Cards are separated by a `---` rule.

## Conviction badge taxonomy

The badge displayed on each card heading (and in the shortlist Signal
column) is one of exactly four fixed values:

- `highest conviction` — used for every card classified Highest tier.
- `strong` — used for a Strong-tier card with clean evidence.
- `moderate` — used for a Strong-tier card on the soft edge of the tier
  (single-source pack, low severity, no quote with notable engagement,
  or similar). The agent picks `strong` vs. `moderate` by judgment
  within the Strong tier; the constraint is on vocabulary, not on the
  call.
- `exploratory` — used for every card classified Exploratory tier. Only
  appears in the brief when Exploratory items are surfaced (zero-Highest
  fallback, or the user explicitly asked for breadth).

`moderate` is a display-layer label only — the rubric in
`SYNTHESIS_TEMPLATE.md` stays at three tiers (Highest / Strong /
Exploratory). `moderate` does not gate hold-back.

## Cardinalities and conditional rules

- Stats strip: 4 metrics in the order listed above; placeholder (zero or
  em-dash) when a metric is not computable in this run.
- Shortlist table: exactly 4 columns, in the order listed above.
- Shortlist Signal column value matches the badge on the corresponding
  card heading and is one of the four fixed values above.
- Builder Fit and Evidence-vs-Fit Tradeoff sections: omitted entirely
  when no explicit builder preferences were supplied.
- Cross-subreddit relevance: rendered as a per-card paragraph only when
  `cross_subreddit_evidence` has actual quotes. The inferred-only case
  is covered by the single global Cross-community line in Method Notes;
  do not render a per-card paragraph for inferred extension.
- Interview Questions: at most 2 per card; each must cite a specific
  quoted phrase or named risk from this card. Section is omitted
  entirely when no question clears that bar.
- No per-card Caveats footer. Evidence-shape caveats fold into the
  Signal line as a trailing clause.
- Card ordering: agent-ranked from strongest to weakest evidence
  judgment. No tier section headers; the per-card badge carries the
  conviction signal.

## Worked example card

The block below is a fabricated example that demonstrates the card
skeleton including the conditional Builder Fit, Evidence-vs-Fit
Tradeoff, Cross-subreddit relevance, and Interview Questions sections.
Placeholder values only; copy and adapt.

```markdown
### 1. Example opportunity title — *strong*

**Signal.** 4 sources, severity 5-6, top thread score 2,751; single-source pack.

**Problem.** One or two sentences naming the user's concrete pain.

**User Segment.** One or two sentences naming who is affected, marked as
inferred when the segment is broader than the quoted source.

**Builder Fit (solo tech founder): strong.** One or two sentences explaining
why this is or isn't a good fit for the supplied builder profile. Render
this paragraph only when explicit preferences were supplied.

**Evidence-vs-Fit Tradeoff.** One or two sentences calling out the tradeoff
between the evidence strength and the builder fit, when both are known.

#### Evidence Quotes

> "exact quote text from the post or comment" — [Post (score 844): short label](https://reddit.com/r/example/comments/abc123/example_post/)
> "second quote, different source if possible" — [Comment (score 112): short label](https://reddit.com/r/example/comments/abc123/example_post/def456/)

**MVP Angle.** One or two sentences describing the smallest product that
would test willingness to pay for this pain.

#### Risks

- Risk one: a concrete failure mode the agent can name from the evidence.
- Risk two: why existing solutions may fail (folded in as a risk bullet).
- Risk three: any feasibility or category risk visible in the evidence.

**Cross-subreddit relevance.** One or two sentences naming why the pain
extends beyond the requested subreddit, citing the adjacent-community
quotes from `cross_subreddit_evidence`. Render only when actual cross-
community quotes exist; the inferred-only case is covered by the global
Cross-community line in Method Notes.

#### Interview Questions

1. Question that quotes a specific phrase from the evidence (e.g. "you
   said the workflow takes 5 clicks — walk me through the version that
   takes 1").
2. Question that probes a concrete named risk from this card (e.g. the
   feasibility blocker, the competitive moat, or the buyer-segment
   mismatch). Omit the second question, or the entire section, if no
   question clears that bar.
```
