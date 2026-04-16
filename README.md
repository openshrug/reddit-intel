# reddit-intel

Scrape Reddit, extract user painpoints via LLM, and build a self-maintaining
category taxonomy -- all stored in a local SQLite database.

See [PIPELINE.md](PIPELINE.md) for architecture, data flow, and stage details.

## Quick start

```bash
# Clone and install (Python 3.11+)
pip install -e .

# Set credentials in .env
cp .env.example .env   # then fill in values (see below)

# Run the pipeline for one subreddit
python main.py ExperiencedDevs
```

## Credentials

Create a `.env` file at the project root:

```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
OPENAI_API_KEY=...
```

**Reddit:** Create a "script" app at <https://www.reddit.com/prefs/apps>.
The client ID is shown under the app name; the secret is labeled "secret".
Set any redirect URI (e.g. `http://localhost:8080`).

**OpenAI:** Get an API key at <https://platform.openai.com/api-keys>.
Required for painpoint extraction and embedding-based merging.

## MCP Server (for AI agents)

reddit-intel ships an MCP (Model Context Protocol) server that lets any
MCP-compatible agent -- OpenClaw, Claude Code, Cursor, and others -- query
the painpoint database and drive the scraping pipeline.

### Install

```bash
pip install -e ".[mcp]"
```

### Run standalone

```bash
python mcp_server.py
```

### Connect from Claude Code

The repo includes a `.mcp.json` that configures the server automatically.
Make sure the env vars are set in your shell or `.env` file, then open the
project in Claude Code -- it will pick up the server config.

To add it manually:

```bash
claude mcp add --transport stdio reddit-intel -- python -m mcp_server
```

### Connect from OpenClaw

Add to `~/.config/openclaw/openclaw.json5`:

```json5
{
  mcp: {
    servers: {
      "reddit-intel": {
        transport: "stdio",
        command: "python",
        args: ["-m", "mcp_server"],
        env: {
          REDDIT_CLIENT_ID: "${REDDIT_CLIENT_ID}",
          REDDIT_CLIENT_SECRET: "${REDDIT_CLIENT_SECRET}",
          OPENAI_API_KEY: "${OPENAI_API_KEY}"
        }
      }
    }
  }
}
```

### Connect from Cursor

Add to your Cursor MCP settings (`.cursor/mcp.json` or global config):

```json
{
  "mcpServers": {
    "reddit-intel": {
      "command": "python",
      "args": ["-m", "mcp_server"]
    }
  }
}
```

### Available tools

| Tool | Type | Description |
|------|------|-------------|
| `get_stats` | read | Global DB counts |
| `list_categories` | read | Full taxonomy |
| `get_top_painpoints` | read | Painpoints ranked by signal count, filterable by category/subreddit |
| `get_painpoint` | read | Single painpoint by ID |
| `get_painpoint_evidence` | read | Reddit posts/comments backing a painpoint |
| `get_subreddit_summary` | read | Aggregate stats for a subreddit |
| `get_post` | read | Full post with comments |
| `run_sql` | read | Arbitrary SELECT queries (escape hatch) |
| `scrape_subreddit` | write | Full scrape+extract+promote pipeline |
| `search_reddit` | write | Search Reddit (no DB persistence) |
| `find_trending_subreddits` | write | Discover growing subreddits via Subriff |

### Available resources

| URI | Description |
|-----|-------------|
| `reddit-intel://schema` | Database schema (for composing `run_sql` queries) |
| `reddit-intel://stats` | DB stats snapshot |
| `reddit-intel://taxonomy` | Category taxonomy |
