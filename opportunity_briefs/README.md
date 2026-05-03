# Opportunity Briefs

This folder holds agent instructions and optional saved research artifacts for
the MCP opportunity-discovery flow.

`reddit-intel` supplies evidence packs. Your agent turns those packs into
product opportunity briefs.

## First Demo Flow

Ask your agent:

```text
Brief me on r/smallbusiness.
```

The agent should:

1. Ask how many opportunities the user wants if no count was provided.
2. Ask small personalization questions only if builder fit would materially
   change the ranking.
3. Read `reddit-intel://stats`.
4. Call `get_subreddit_summary("smallbusiness")`.
5. Ask before scraping if there is no data.
6. Call `get_opportunity_evidence("smallbusiness", limit=10)`.
7. Produce opportunity briefs using `opportunity_briefs/AGENTS.md`.

## Folder Layout

```text
opportunity_briefs/
  AGENTS.md
  README.md
  SYNTHESIS_TEMPLATE.md
  runs/
    <local generated runs, gitignored>
```

`SYNTHESIS_TEMPLATE.md` is the customizable prompt template returned by
`get_opportunity_evidence`. Edit or copy it when you want a different brief
style for a run. `runs/` is for local outputs and may contain private
assumptions, bulky evidence snapshots, or messy drafts. Example runs will be
generated later from real captured evidence.

## Evidence Scope

In a shared DB, a painpoint can have both local and cross-subreddit evidence.
Opportunity briefs should label both:

- `local_evidence`: quotes from the requested subreddit.
- `cross_subreddit_evidence`: quotes from adjacent communities.
- `local_signal_count`: repetition signal within the requested subreddit.
- `global_signal_count`: repetition signal across the canonical painpoint.

Signal counts can include deduped sources that do not have separate stored
quotes, so quote counts may be lower than signal counts.

If the user wants strict single-subreddit evidence, use a clean DB manually for
that run. The MVP does not automate isolated DB runs.
