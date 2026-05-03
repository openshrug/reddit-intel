# Opportunity Brief Agent Instructions

Use these instructions when turning `get_opportunity_evidence` output into
product opportunity briefs.

`SYNTHESIS_TEMPLATE.md` is the canonical synthesis prompt. For run-specific
behavior, edit or copy that template; the MCP evidence response returns it as
the agent-facing synthesis prompt.

## Flow

1. Parse the requested subreddit from prompts like `brief me on r/smallbusiness`.
2. If the user did not specify a count, ask how many opportunities they want.
3. Ask small personalization questions only if builder fit would materially
  change the final ranking.
4. Read `reddit-intel://stats` and call `get_subreddit_summary` for the
  requested subreddit.
5. If data is missing, ask before calling `scrape_subreddit`; scraping is slow
  and uses Reddit/OpenAI quota.
6. Call `get_opportunity_evidence(subreddit, limit=10)`.
7. Build an evidence-first opportunity shortlist from the returned packs.
8. After presenting the brief, ask whether the user wants to persist it under
  `opportunity_briefs/runs/`.
9. If the user says yes, persist the exact `SYNTHESIS_TEMPLATE.md` used
  alongside `evidence.json`, `assumptions.md`, and `brief.md` so the synthesis
  is reproducible later.

## Evidence Rules

- Start from the evidence packs; do not invent opportunities from general
market knowledge.
- Use local evidence from the requested subreddit first.
- Use cross-subreddit evidence as clearly labeled adjacent support.
- Render every quote as a Markdown hyperlink using `source_permalink`.
- Treat signal counts as repetition signals, not market-size estimates.
- Treat category labels as navigation aids, not ground truth.
- Do not estimate TAM, willingness to pay, or competitor coverage unless the
evidence directly supports it.

Signal counts can include deduped sources that do not have separate stored
quotes, so quote counts may be lower than signal counts.
