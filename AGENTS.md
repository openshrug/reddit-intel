# AGENTS.md

Orientation for AI coding agents (Cursor, Claude Code, OpenClaw, Codex,
OpenCode, etc.) working in this repo. Treat this file as a starting
point — depth lives in [`PIPELINE.md`](PIPELINE.md).

## What this project is

`reddit-intel` is a local-first pipeline that scrapes Reddit, extracts
user painpoints with an LLM, and maintains a self-organising category
taxonomy. Output lands in a single SQLite file (`trends.db`). A bundled
MCP server exposes the database to other AI agents.

## Key entrypoints

- [`main.py`](main.py) — CLI entrypoint (`reddit-intel <subreddit>`).
  Loads `.env`, parses args, calls `subreddit_pipeline.analyze`.
- [`subreddit_pipeline.py`](subreddit_pipeline.py) — orchestrates the
  4-stage pipeline: scrape → persist → extract → promote. Always start
  reading here if you want to understand control flow.
- [`mcp_server.py`](mcp_server.py) — MCP entrypoint
  (`reddit-intel-mcp`). Exposes 11 tools and 5 resources backed by
  the same DB.
- [`reddit_scraper.py`](reddit_scraper.py) — Reddit OAuth + listing
  fetch + comment tree expansion.
- [`painpoint_extraction/`](painpoint_extraction/) — LLM extraction
  package (prompts, schemas, OpenAI calls).
- [`promoter.py`](promoter.py) — embedding-based dedup that promotes
  `pending_painpoints` into the canonical `painpoints` table.
- [`category_worker.py`](category_worker.py) — taxonomy mutation
  worker (split / merge / rename / reparent passes).
- [`db/`](db/) — schema + every typed query. Schema lives in
  [`db/schema.sql`](db/schema.sql); read it before composing ad-hoc
  SQL. Migrations are in `db/__init__.py:_apply_migrations()` because
  SQLite has no `ADD COLUMN IF NOT EXISTS`.
- [`taxonomy.yaml`](taxonomy.yaml) — seed taxonomy loaded on
  `db.init_db()`.

## Tests

- `pytest`           → offline tests (default). Hits the DB layer + pure
  Python only, no network.
- `pytest -m live`   → runs [`tests/live/`](tests/live/) only. These
  hit real Reddit + OpenAI APIs, take 15+ minutes total, and cost
  money. Need `REDDIT_CLIENT_ID/SECRET` and `OPENAI_API_KEY`.
- The `live` marker is auto-applied to everything under `tests/live/`
  via [`tests/live/conftest.py`](tests/live/conftest.py); no per-file
  decorator needed.
- `ruff check .`     → lint.

## Environment

`.env` at repo root holds `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`,
`REDDIT_USER_AGENT`, `OPENAI_API_KEY`. See [`README.md`](README.md) for
how to obtain them.

## Conventions

- Flat-layout package: most modules live at repo root, not under a `src/`
  dir. The `[tool.setuptools]` block in
  [`pyproject.toml`](pyproject.toml) lists exactly which modules ship.
  When adding a new top-level module, add it to `py-modules` there.
- Data files (`taxonomy.yaml`, `db/schema.sql`) are loaded via
  `Path(__file__).parent`, which works in editable installs only. If
  you ever build a wheel, add `MANIFEST.in` first.
- The DB write path goes through typed helpers in `db/`, never raw
  `cursor.execute(...)` from feature code.
- The MCP `run_sql` tool is `SELECT`-only and is the documented escape
  hatch — don't add new typed read tools for one-off queries unless
  they're reusable.
