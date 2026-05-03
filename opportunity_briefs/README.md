# Opportunity Briefs

This folder holds the agent workflow + synthesis template for the MCP
opportunity-discovery flow, plus optional saved research artifacts.

`reddit-intel` supplies evidence packs (via `get_opportunity_evidence`); your
agent turns those packs into product opportunity briefs by following the
workflow in `AGENTS.md`, the synthesis structure in `SYNTHESIS_TEMPLATE.md`,
and the rendering / canvas-escape-hatch rules in `BRIEF_LAYOUT.md`. All
three files are exposed via MCP so any MCP-aware client can fetch the
current versions:

- `reddit-intel://opportunity-brief-instructions` -> `AGENTS.md`
- `reddit-intel://opportunity-brief-template` -> `SYNTHESIS_TEMPLATE.md`
- `reddit-intel://opportunity-brief-layout` -> `BRIEF_LAYOUT.md`

## First Demo Flow

Ask your agent:

```text
Brief me on r/smallbusiness.
```

In MCP clients that surface prompts, this routes to the `opportunity_brief`
prompt registered by `mcp_server.py`. The prompt takes only the subreddit
name — there is no count parameter and the agent does not ask the user for a
target number. The prompt is a thin launcher that:

1. Tells the agent to fetch `reddit-intel://opportunity-brief-instructions`
   and follow it for tool-use flow (stats check, scrape permission,
   persistence prompt).
2. Tells the agent to fetch `reddit-intel://opportunity-brief-template` and
   follow it for conviction-tier classification and the per-opportunity
   field list.
3. Tells the agent to fetch `reddit-intel://opportunity-brief-layout` and
   follow it for the document skeleton, per-opportunity card rendering,
   and the Cursor-canvas escape hatch.
4. Tells the agent to call `get_opportunity_evidence(subreddit, limit=30)` (the value comes from `opportunities.BRIEF_EVIDENCE_LIMIT`)
   for the evidence.

The agent classifies each evidence pack into highest / strong / exploratory
conviction (criteria in `SYNTHESIS_TEMPLATE.md`) and surfaces highest +
strong in the initial brief; exploratory items are held back unless you ask
for more breadth. All workflow, synthesis, and layout rules live in the
three Markdown files; edit them to change agent behavior without touching
Python.

## Folder Layout

```text
opportunity_briefs/
  AGENTS.md
  BRIEF_LAYOUT.md
  README.md
  SYNTHESIS_TEMPLATE.md
  runs/
    <local generated runs, gitignored>
```

`runs/` is for local outputs and may contain private assumptions, bulky
evidence snapshots, or messy drafts.

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
