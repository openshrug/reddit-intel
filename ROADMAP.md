# Roadmap

What's coming next for `reddit-intel`. Non-binding — priorities shift
as the project finds its shape.

## Recently shipped

- **Self-maintaining taxonomy.** `category_worker.py` proposes and runs
  split / merge / rename / reparent passes against the live category
  tree.
- **Embedding-based dedup.** `sqlite-vec` powers cosine-similarity
  promotion of `pending_painpoints` into the canonical `painpoints`
  table.
- **MCP server.** 11 tools + 3 resources ([`mcp_server.py`](mcp_server.py)),
  installable as a `reddit-intel-mcp` console command.

## Planned

### Caching

- **Scrape caching.** `scrape_requests` + `request_posts` tables with a
  versioned cache key (`subreddit + scrape_version`) and per-config
  TTL. Eliminates redundant Reddit API calls — biggest cost/latency
  win.
- **DB-level fetch dedup.** Check `posts.fetched_at` before scraping;
  skip comment fetches for posts stored within a configurable recency
  window (e.g. 24h). Cheap fix that saves the bulk of the request
  budget on repeat runs.
- **Extraction caching.** `extraction_runs` table keyed by
  `post_ids_hash + prompt_version`; reuse prior LLM output when posts
  + prompt are unchanged. Medium complexity around partial-overlap
  semantics.

### Retrieval

- **FTS5 full-text search.** `posts_fts(title, selftext)` and
  `comments_fts(body)` virtual tables, exposed as a new MCP tool. Lets
  agents search raw Reddit text by keyword, complementing the existing
  semantic search over painpoints.
- **Cross-subreddit search caching.** Extend `scrape_requests` to
  support a `query` field with nullable `subreddit`, dedup search
  results against the existing `posts` table. Today's cross-sub keyword
  searches (`"frustrated with" OR "I hate"`) don't fit the
  per-subreddit cache model.
