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
3. Stats strip — 5 metrics rendered as a horizontal row, sourced from the
   MCP stats and summary tools: Posts scraped, Comments persisted, Promoted
   signals, Painpoints, Initial posts (the count in the DB before this run,
   for delta context on fresh DBs). Render zero or em-dash for any metric
   not computable in the current run rather than dropping it.
4. Headline insight — a callout of 1-2 short paragraphs (2-4 sentences
   total) synthesizing the macro pattern across all opportunities. Visually
   distinct from body paragraphs.
5. Opportunity shortlist — a 4-column table covering every opportunity that
   will be detailed below. Columns: Rank, Opportunity, Primary Segment,
   Signal. The Signal column uses the same free-form badge value the card
   heading uses (see Conviction badge taxonomy below) so the two stay in
   sync.
6. Detailed opportunities — a section header followed by one card per
   opportunity. Cards are listed as a single ranked sequence ordered by
   the agent's judgment of evidence strength (strongest first). Do not
   group cards under conviction-tier sub-headers; the per-card badge is
   the only conviction signal the reader sees. The tier classification in
   `SYNTHESIS_TEMPLATE.md` still gates surface vs. hold-back, but it does
   not dictate a visible grouping.
7. Method notes — a footer with two short labeled blocks: MCP DB (db path
   plus before/after stats) and Signal Interpretation (the standard
   "signal strength is directional founder research, not TAM sizing"
   disclaimer).

### Per-opportunity card spine (sections in order)

1. Heading — rank, opportunity title, and a small conviction badge. The
   badge is visually subordinate to the title.
2. Problem — 1-2 sentence paragraph describing the user's pain.
3. User Segment — 1-2 sentence paragraph naming who is affected.
4. Signal Strength — 1-2 sentence paragraph naming source count, signal
   counts, severity range, and any notable engagement scores.
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
    render as its own paragraph when `cross_subreddit_evidence` has actual
    quotes; fold into the Caveats footer line as "cross-community
    extension inferred, not in DB" when it is purely inferred.
11. Interview Questions — a numbered list of 3-5 questions to validate the
    opportunity with real users.
12. Caveats — a short de-emphasized footer (1-2 sentences). Optional;
    omit if there is nothing of substance to say.

### Conviction badge taxonomy

The badge displayed on each card heading is a free-form 1-3 word descriptor
that should be anchored to but not constrained by the three-tier
classification rubric in `SYNTHESIS_TEMPLATE.md`. The model picks the badge
label that most honestly describes the evidence shape.

Examples: `highest conviction`, `strong`, `moderate`, `moderate but broad`,
`moderate to strong`, `exploratory`, `niche wedge`.

The classification rubric (Highest / Strong / Exploratory) still gates the
shortlist plus hold-back logic underneath; the badge is the human-facing
label. The Signal column in the shortlist table uses the same badge value as
the card heading.

### Cardinalities and conditional rules

- Stats strip: 5 metrics in the order listed above; placeholder (zero or
  em-dash) when a metric is not computable in this run.
- Shortlist table: exactly 4 columns, in the order listed above.
- Shortlist Signal column value matches the badge on the corresponding
  card heading.
- Builder Fit and Evidence-vs-Fit Tradeoff sections: omitted entirely when
  no explicit builder preferences were supplied.
- Cross-subreddit relevance: standalone paragraph when
  `cross_subreddit_evidence` has quotes; folded one-liner into Caveats when
  purely inferred.
- Caveats footer: optional per card; omit if there is nothing of substance
  to say.
- Card ordering: agent-ranked from strongest to weakest evidence judgment.
  No tier section headers; the per-card badge carries the conviction
  signal.

## Layer 3: Markdown rendering notes

For each Layer 2 element, the markdown construct to use:

- Title → `# r/{subreddit} — Opportunity Brief`.
- Subtitle → plain paragraph immediately under the title.
- Stats strip → 5-column markdown table:
  `| Posts scraped | Comments persisted | Promoted signals | Painpoints | Initial posts |`.
  Markdown tables are the only pure-markdown construct giving a clean
  horizontal metrics row.
- Headline insight → single `> blockquote` (multi-line if needed; markdown
  blockquotes preserve internal paragraph breaks with `>` on a blank line).
- Opportunity shortlist → markdown table:
  `| Rank | Opportunity | Primary Segment | Signal |`.
- Detailed opportunities header → `## Detailed Opportunities`.
- Card heading → `### N. {Title} — *{badge}*` (italic suffix is the
  markdown stand-in for the floating canvas badge; the badge value follows
  the loosened taxonomy in Layer 2). Cards live directly under
  `## Detailed Opportunities` — no intermediate tier section header.
- Problem / User Segment / Signal Strength / Builder Fit / Evidence-vs-Fit
  Tradeoff / MVP Angle / Cross-subreddit relevance → bold-labeled paragraph,
  e.g. `**Problem.** ...`.
- Evidence Quotes → `#### Evidence Quotes` sub-heading, then each quote on
  its own line as
  `> "quote text" — [Post (score N): short label](https://reddit.com/...)`.
- Risks → `#### Risks` sub-heading plus bullet list.
- Interview Questions → `#### Interview Questions` sub-heading plus
  numbered list.
- Caveats → `_Caveats: ..._` italic footer (1-2 sentences, single block).
- Card separator → `---` between cards.
- Method notes → `## Method Notes` then two stacked `**Label.**` paragraphs
  (pure markdown, no two-column HTML).

### Worked markdown example card

The block below is a fabricated example that demonstrates the card skeleton
including the conditional Builder Fit, Evidence-vs-Fit Tradeoff, and
Cross-subreddit relevance sections. Placeholder values only; copy and adapt.

```markdown
### 1. Example opportunity title — *strong*

**Problem.** One or two sentences naming the user's concrete pain.

**User Segment.** One or two sentences naming who is affected, marked as
inferred when the segment is broader than the quoted source.

**Signal Strength.** `local_signal_count=N`, `global_signal_count=N`,
severity X-Y, K stored local quotes, with one or two notable engagement
scores called out (post score 844, comment score 927).

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

**Cross-subreddit relevance.** Render as a paragraph when
`cross_subreddit_evidence` has actual quotes from adjacent communities;
otherwise fold into the Caveats line as "cross-community extension
inferred, not in DB."

#### Interview Questions

1. Question that probes whether the named pain matches the user's actual
   workflow.
2. Question that probes the willingness-to-pay shape.
3. Question that probes the integration or distribution wedge.
4. Question that probes the bottleneck — tool vs. behavior vs. policy.
5. Question that probes the vertical-vs-horizontal cut.

_Caveats: short de-emphasized footer line; can be 1-2 sentences if the
caveat is two-pronged (e.g., evidence is concentrated in one thread; cross-
community extension inferred, not in DB)._
```
