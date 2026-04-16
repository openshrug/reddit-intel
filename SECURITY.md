# Security Policy

## Supported versions

Only the `main` branch receives security fixes. Pin to a tagged release
if you need stable behaviour.

## Reporting a vulnerability

Please report vulnerabilities **privately** via
[GitHub Security Advisories](https://github.com/openshrug/reddit-intel/security/advisories/new).

Avoid opening public issues for anything you suspect could be exploited.

## Tool-safety notes for MCP integrators

The bundled MCP server ([`mcp_server.py`](mcp_server.py)) exposes both
read and write tools to the connected agent. Worth knowing before you
hand it to an agent with broad permissions:

- **`run_sql` is `SELECT`-only.** Non-`SELECT` statements are rejected
  by `db.queries.run_sql`. The tool reads the schema resource as a
  guardrail, but treat agent-generated SQL as untrusted and review
  before approving in production.
- **Write tools spend external API quota.** `scrape_subreddit`,
  `search_reddit`, and `find_trending_subreddits` make real Reddit /
  OpenAI / Subriff calls. Long agent loops can run up bills and trip
  Reddit rate limits — scope those tools deliberately.
- **No authentication on the stdio transport.** The server inherits the
  privileges of whatever process spawned it. Don't expose it on a
  network interface; keep it stdio-local.
