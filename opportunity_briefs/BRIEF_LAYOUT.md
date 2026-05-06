# Opportunity Brief Layout

This file owns the **rendering** of an opportunity brief. The synthesis rules
(conviction-tier rubric, evidence rules, per-opportunity field semantics) live
in `SYNTHESIS_TEMPLATE.md`. Read both before producing a brief.

The file has three layers:

- **Layer 1** tells the agent *which renderer to pick* (Cursor canvas vs.
  markdown).
- **Layer 2** is the **renderer-agnostic structure spine**. It names every
  document part and every per-opportunity card section in order, with their
  semantics and conditional rules. It contains no markdown-syntax characters
  on purpose, so a canvas agent can use it as the structural source of truth
  without filtering noise.
- **Layer 3** is the **markdown rendering guide**. It translates each Layer 2
  element into a concrete markdown construct and ends with a worked example
  card.

## Layer 1: Cursor canvas escape hatch

If the rendering client supports Cursor canvases (Cursor Desktop / IDE) and
the brief will be presented interactively, prefer producing the brief as a
`.canvas.tsx` artifact using the Canvas skill at
`/Users/viktar/.cursor/skills-cursor/canvas/SKILL.md`. Use Layer 2 below as
the structural source of truth for what to put on the canvas; the canvas
skill owns how to build canvases.

Markdown (Layer 3) is the fallback for terminal and file-based clients, and
is **always** used for the persisted `runs/.../brief.md` file regardless of
which path the agent took live. This keeps the persisted artifact
reproducible and diffable.

## Layer 2: Brief structure spine (renderer-agnostic)

The spine — describes parts, order, semantics, cardinalities, and conditional
rules. No markdown syntax appears in this section.

### Document parts (in order)

1. Title — subreddit name plus the words "Opportunity Brief".
2. Subtitle — a 1-2 sentence paragraph stating how the data was sourced
   (which subreddit, fresh scrape vs. existing DB, evidence pack tool).
3. Stats strip — 4 metrics rendered as a horizontal row, sourced from the
   MCP stats and summary tools: Posts scraped, Comments persisted, Promoted
   signals, Painpoints. Render zero or em-dash for any metric not
   computable in the current run rather than dropping it.
4. Headline insight — a single short callout, 2-3 sentences max,
   synthesizing the macro pattern across all surfaced opportunities.
   Visually distinct from body paragraphs. Do not include synthesizer-
   internal commentary (e.g. "the loudest threads are deliberately held
   back") — that belongs in Method notes.
5. Opportunity shortlist — a 4-column table covering every opportunity that
   will be detailed below. Columns: Rank, Opportunity, Primary Segment,
   Signal. The Signal column uses the same fixed badge value the card
   heading uses (see Conviction badge taxonomy below) so the two stay in
   sync.
6. Detailed opportunities — a section header followed by one card per
   opportunity. Cards are listed as a single ranked sequence ordered by
   the agent's judgment of evidence strength (strongest first). Do not
   group cards under conviction-tier sub-headers; the per-card badge is
   the only conviction signal the reader sees. The tier classification in
   `SYNTHESIS_TEMPLATE.md` still gates surface vs. hold-back, but it does
   not dictate a visible grouping.
7. Method notes — a short footer with up to three single-line labeled
   blocks: MCP DB (db path plus before/after stats), Signal Interpretation
   (the standard "signal strength is directional founder research, not
   TAM sizing" disclaimer), and — only when the run is single-subreddit
   and at least one card relies on inferred cross-community extension —
   one global Cross-community line ("Single-subreddit run; cross-
   community extension is inferred per card."). Cross-community wording
   never repeats per card.

### Per-opportunity card spine (sections in order)

1. Heading — rank, opportunity title, and a small conviction badge. The
   badge is visually subordinate to the title.
2. Signal — a single labeled sentence stating source count, severity
   range, and the top engagement score (post score or comment score) on
   the headline thread. When the evidence shape is on the soft edge
   (single-source pack, two-of-N signals from the same thread, etc.),
   append a short trailing clause after a semicolon, e.g. "; single-
   source pack." Do not render `local_signal_count=` / `global_signal_
   count=` key=value form; do not list every quote's score.
3. Problem — 1-2 sentence paragraph describing the user's pain.
4. User Segment — 1-2 sentence paragraph naming who is affected.
5. Builder Fit — 1-2 sentence paragraph. Conditional: render only when
   explicit builder preferences were supplied.
6. Evidence-vs-Fit Tradeoff — 1-2 sentence paragraph. Conditional: same
   gating as Builder Fit.
7. Evidence Quotes — 3-6 quotes, each rendered as a pulled-out callout
   (visually distinct from body text) with a hyperlink to the source
   permalink.
8. MVP Angle — 1-2 sentence paragraph naming the wedge product the evidence
   supports.
9. Risks — a bulleted list of risks. Folds in the prior "why existing
   solutions may fail" content as risk bullets.
10. Cross-subreddit relevance — 1-2 sentence labeled paragraph naming why
    the pain may extend beyond the requested subreddit. Conditional:
    render only when `cross_subreddit_evidence` has actual quotes from
    adjacent communities. The purely inferred case is covered by the
    single global Cross-community line in Method notes — never render a
    per-card paragraph that just says "extension inferred, not in DB."
11. Interview Questions — a numbered list of at most 2 questions, each of
    which must reference a specific quoted phrase or named risk from this
    card. Conditional: omit the entire section if no question clears that
    bar. Do not pad to a target count; do not include boilerplate
    willingness-to-pay or pricing-shape questions.

The card has no Caveats footer. Evidence-shape caveats fold into the
Signal line; the cross-community disclaimer lives once in Method notes.

### Conviction badge taxonomy

The badge displayed on each card heading is one of exactly four fixed
values:

- `highest conviction` — used for every card classified Highest tier.
- `strong` — used for a Strong-tier card with clean evidence.
- `moderate` — used for a Strong-tier card on the soft edge of the tier
  (single-source pack, low severity, no quote with notable engagement,
  or similar). The agent picks `strong` vs. `moderate` by judgment within
  the Strong tier; the constraint is on vocabulary, not on the call.
- `exploratory` — used for every card classified Exploratory tier. Only
  appears in the brief when Exploratory items are surfaced (zero-Highest
  fallback, or the user explicitly asked for breadth).

`moderate` is a display-layer label only — the rubric in
`SYNTHESIS_TEMPLATE.md` stays at three tiers (Highest / Strong /
Exploratory). `moderate` does not gate hold-back.

The Signal column in the shortlist table uses the same fixed badge value
as the card heading.

### Cardinalities and conditional rules

- Stats strip: 4 metrics in the order listed above; placeholder (zero or
  em-dash) when a metric is not computable in this run.
- Shortlist table: exactly 4 columns, in the order listed above.
- Shortlist Signal column value matches the badge on the corresponding
  card heading and is one of the four fixed values above.
- Builder Fit and Evidence-vs-Fit Tradeoff sections: omitted entirely when
  no explicit builder preferences were supplied.
- Cross-subreddit relevance: rendered as a per-card paragraph only when
  `cross_subreddit_evidence` has actual quotes. The inferred-only case is
  covered by the single global Cross-community line in Method notes; do
  not render a per-card paragraph for inferred extension.
- Interview Questions: at most 2 per card; each must cite a specific
  quoted phrase or named risk from this card. Section is omitted entirely
  when no question clears that bar.
- No per-card Caveats footer. Evidence-shape caveats fold into the Signal
  line as a trailing clause.
- Card ordering: agent-ranked from strongest to weakest evidence judgment.
  No tier section headers; the per-card badge carries the conviction
  signal.

## Layer 3: Markdown rendering notes

For each Layer 2 element, the markdown construct to use:

- Title → `# r/{subreddit} — Opportunity Brief`.
- Subtitle → plain paragraph immediately under the title.
- Stats strip → 4-column markdown table:
  `| Posts scraped | Comments persisted | Promoted signals | Painpoints |`.
  Markdown tables are the only pure-markdown construct giving a clean
  horizontal metrics row.
- Headline insight → single `> blockquote`, 2-3 sentences, no internal
  paragraph break.
- Opportunity shortlist → markdown table:
  `| Rank | Opportunity | Primary Segment | Signal |`. Signal column uses
  one of the four fixed badge values from the taxonomy in Layer 2.
- Detailed opportunities header → `## Detailed Opportunities`.
- Card heading → `### N. {Title} — *{badge}*` (italic suffix is the
  markdown stand-in for the floating canvas badge; the badge value is one
  of `highest conviction` / `strong` / `moderate` / `exploratory`). Cards
  live directly under `## Detailed Opportunities` — no intermediate tier
  section header.
- Signal / Problem / User Segment / Builder Fit / Evidence-vs-Fit Tradeoff
  / MVP Angle / Cross-subreddit relevance → bold-labeled paragraph, e.g.
  `**Signal.** ...`. The Signal paragraph is a single sentence; append a
  trailing evidence-shape caveat after a semicolon when warranted.
- Evidence Quotes → `#### Evidence Quotes` sub-heading, then each quote on
  its own line as
  `> "quote text" — [Post (score N): short label](https://reddit.com/...)`.
- Risks → `#### Risks` sub-heading plus bullet list.
- Interview Questions → `#### Interview Questions` sub-heading plus
  numbered list of at most 2 items. Omit the sub-heading entirely when
  no question clears the evidence-grounded bar.
- Card separator → `---` between cards.
- Method notes → `## Method Notes` then up to three stacked `**Label.**`
  paragraphs (pure markdown, no two-column HTML): MCP DB, Signal
  Interpretation, and (only when applicable) Cross-community.

### Worked markdown example card

The block below is a fabricated example that demonstrates the card skeleton
including the conditional Builder Fit, Evidence-vs-Fit Tradeoff, Cross-
subreddit relevance, and Interview Questions sections. Placeholder values
only; copy and adapt.

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
