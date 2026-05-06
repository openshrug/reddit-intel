# Opportunity Brief Agent Instructions

Use these instructions when turning `get_opportunity_evidence` output into
product opportunity briefs.

`SYNTHESIS_TEMPLATE.md` is the canonical synthesis prompt (fields and
conviction-tier rubric). `BRIEF_LAYOUT.md` is the canonical layout /
rendering guide (document skeleton and per-opportunity card markdown).
For run-specific behavior, edit or copy those files. All three files are
exposed via MCP so any MCP-aware agent can fetch the current versions
without going through the repo filesystem:

- `reddit-intel://opportunity-brief-instructions` serves this file.
- `reddit-intel://opportunity-brief-template` serves `SYNTHESIS_TEMPLATE.md`.
- `reddit-intel://opportunity-brief-layout` serves `BRIEF_LAYOUT.md`.

## Flow

1. Parse the requested subreddit from prompts like `brief me on r/smallbusiness`.
2. Ask small personalization questions only if builder fit would materially
  change the final ranking.
3. Read `reddit-intel://stats` and call `get_subreddit_summary` for the
  requested subreddit.
4. If data is missing, ask before calling `scrape_subreddit`; scraping is slow
  and uses Reddit/OpenAI quota.
5. Call `get_opportunity_evidence(subreddit, limit=30)` (the value comes from `opportunities.BRIEF_EVIDENCE_LIMIT` — update there to re-tune).
6. Classify each pack into a conviction tier per `SYNTHESIS_TEMPLATE.md`
  (fields and rubric) and render the brief per `BRIEF_LAYOUT.md` (document
  skeleton and card markdown); build the shortlist from highest + strong
  conviction tiers and hold exploratory candidates back unless the user
  asks for more breadth.
7. After presenting the brief, ask whether the user wants to persist it under
  `opportunity_briefs/runs/`.
8. If the user says yes, persist the exact `SYNTHESIS_TEMPLATE.md` and
  `BRIEF_LAYOUT.md` used alongside `evidence.json`, `assumptions.md`, and
  `brief.md` so both the synthesis and the rendering are reproducible later.
9. If the user asks for more breadth, surface the held-back exploratory
  candidates using the same per-opportunity structure.

## Evidence Rules

- Start from the evidence packs; do not invent opportunities from general
market knowledge.
- Use local evidence from the requested subreddit first.
- Use cross-subreddit evidence as clearly labeled adjacent support.
- Render every quote as a Markdown hyperlink using `source_permalink`.
- Treat signal counts as repetition signals, not market-size estimates.
- Treat category labels as navigation aids, not ground truth.
- Treat Reddit evidence as qualitative signal, not proof of demand or
willingness to pay.
- Do not estimate TAM, willingness to pay, or competitor coverage unless the
evidence directly supports it.

Signal counts can include deduped sources that do not have separate stored
quotes, so quote counts may be lower than signal counts.