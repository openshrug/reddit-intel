"""
Ingest pipeline — analyze() scrapes Reddit + Subriff, extracts structured
facts, categorizes them against the taxonomy, and cleans up the DB.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

import database as db
from llm import get_client, llm_call, execute_sql_queries, web_search, DB_SCHEMA
from subriff_scraper import scrape_fastest_growing
from reddit_scraper import deep_scrape_all


SCRAPE_CONFIG = Path(__file__).parent / "scrape_config.yaml"


# ============================================================
# CONFIG
# ============================================================

def _load_config():
    if not SCRAPE_CONFIG.exists():
        return _default_config()
    with open(SCRAPE_CONFIG) as f:
        return yaml.safe_load(f) or _default_config()


def _default_config():
    return {
        "subriff": {"enabled": True, "min_subscribers": 5000, "limit": 15},
    }


# ============================================================
# PROMPTS
# ============================================================

FILTER_PROMPT = """You are a tech signal filter. MOST items are noise — drop them aggressively.

Only keep items that reveal:
- A specific, actionable painpoint (not "tech is hard")
- A product with clear traction or failure signal
- Funding with thesis/amount details
- A genuine market shift (not generic AI hype)

Drop: generic news, opinion pieces, memes, career advice, vague announcements.

Return JSON:
{
  "kept": [{"title": "...", "why": "why this matters", "importance": 1-10}],
  "dropped_count": N,
  "source_summary": "one sentence meta-signal"
}

importance 8+: clear market signal. 5-7: useful context. Below 5: don't keep it."""


EXTRACT_PROMPT = f"""You are building a market intelligence database. Your goal is to map
the REAL competitive landscape — who has what problem, which products exist, how well
they actually work, and where the genuine gaps are.

You have SQL access to the DB and web search. USE BOTH.

Be rigorous:
- Before recording a painpoint, verify it's real and specific (not generic whining)
- Before marking a gap, web search for existing solutions — many "gaps" are just ignorance
- product_painpoint_links are the most valuable output: be honest about effectiveness.
  If a product fully solves a pain, say effectiveness:9. Don't invent gaps that don't exist.
- Severity should reflect real impact, not how loudly people complain.
  Severity 8+ means "blocks entire workflows". Severity 3 means "mild annoyance".

Each response must be JSON:
- To research: {{"done": false, "sql_queries": [...], "web_searches": [...]}}
  sql_queries: [{{"query": "SELECT ...", "reason": "..."}}]
  web_searches: [{{"query": "search terms", "reason": "what to verify"}}]
- To commit: {{"done": true, "products": [...], "painpoints": [...], "funding": [...],
  "product_painpoint_links": [...], "proposed_categories": [...]}}

Output schema:
- products: [{{name, description, builder, tech_complexity (LOW/MEDIUM/HIGH), viral_trigger, why_viral, category_slugs}}]
- painpoints: [{{title, description, severity (1-10), category_slugs, quotes (real user quotes only)}}]
- funding: [{{company, amount, valuation, round_type, investors, what_they_build, painpoint_solved, why_funded, category_slugs}}]
- product_painpoint_links: [{{product, painpoint, relationship (addresses/fails_at/partial), effectiveness (1-10), gap_description, gap_type (pricing/performance/ux/scope/reliability/integration)}}]
- proposed_categories: [{{name, parent_slug, reason}}]

category_slugs must come from the provided taxonomy.

DB schema: {DB_SCHEMA}"""


TAXONOMY_PROMPT = """You manage a tech category taxonomy. Review pending categories and
decide: promote, reject, or merge. Return JSON:
{{
  "promote": [{{"pending_id": 1, "parent_slug": "ai-ml", "description": "..."}}],
  "reject": [{{"pending_id": 2, "reason": "duplicate"}}],
  "merge_pending_into_existing": [{{"pending_id": 3, "target_slug": "..."}}],
  "merge_existing": [{{"keep_id": 60, "merge_ids": [61, 62]}}],
  "rename": [{{"category_id": 70, "new_name": "...", "new_description": "..."}}]
}}
Never touch seed categories. Be conservative."""


CLEANUP_PROMPT = """Clean a trend intelligence database. Return JSON:
{
  "delete": {"painpoints": [IDs], "products": [IDs]},
  "merge_painpoints": [{"keep_id": 5, "merge_ids": [12, 17], "reason": "same issue"}]
}
Be aggressive with vague junk, conservative with specific entries."""


# ============================================================
# SOURCE FORMATTING
# ============================================================

_FMT = {
    "reddit": lambda p: f"- [score:{p.get('score', 0)}] r/{p.get('subreddit', '')}: {p.get('title', '')}" + (f"\n  > {p['selftext'][:150]}" if p.get("selftext") else ""),
    "subriff": lambda s: f"- r/{s.get('name', '')} ({s.get('subscribers', 0):,} subs, +{s.get('weekly_growth_pct', 0)}% growth)",
}


def _format_items(source_name, items, limit=40):
    key = source_name.replace("deep_", "")
    fmt = _FMT.get(key, lambda x: f"- {x.get('title', str(x)[:100])}")
    return [fmt(i) for i in items[:limit]]


def _curated_text(filtered):
    parts = []
    for source, data in filtered.items():
        kept = data.get("kept", [])
        if not kept:
            continue
        parts.append(f"## {source}")
        if data.get("summary"):
            parts.append(f"Meta: {data['summary']}")
        for item in kept[:15]:
            parts.append(f"- [{item.get('importance', '?')}/10] {item.get('title', '')}")
            if item.get("why"):
                parts.append(f"  → {item['why']}")
        parts.append("")
    return "\n".join(parts)


# ============================================================
# SCRAPING
# ============================================================

def _scrape_all(config):
    """Scrape Reddit + Subriff. Everything in one pass."""
    sources = {}

    # Subriff
    sr = config.get("subriff", {})
    if sr.get("enabled", True):
        print("📡 Subriff...")
        try:
            sources["subriff"] = scrape_fastest_growing(
                min_size=sr.get("min_subscribers", 5000), limit=sr.get("limit", 15))
        except Exception as e:
            print(f"  Subriff failed: {e}")

    # Deep scrape: Reddit (subs + keyword search + discovery)
    print("\n📡 Deep scrape (Reddit)...")
    deep_data = deep_scrape_all(openai_api_key=os.getenv("OPENAI_API_KEY"))
    for name, items in deep_data.items():
        sources[name] = items

    return sources


# ============================================================
# LLM PASSES
# ============================================================

def _pass_filter(client, sources):
    """Filter all sources in parallel."""
    print("\n🔍 Filtering signals...")

    def _one(name, items):
        text = "\n".join(_format_items(name, items))
        if not text:
            return name, {"kept": [], "summary": ""}
        try:
            msgs = [{"role": "system", "content": FILTER_PROMPT},
                    {"role": "user", "content": f"Source: {name}\n\n{text}"}]
            data = json.loads(llm_call(client, msgs, max_tokens=2048))
            kept = sorted(data.get("kept", []), key=lambda x: x.get("importance", 0), reverse=True)
            return name, {"kept": kept, "summary": data.get("source_summary", ""), "dropped": data.get("dropped_count", 0)}
        except Exception as e:
            print(f"    {name}: filter failed ({e})")
            return name, {"kept": [], "summary": ""}

    filtered = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_one, n, items): n for n, items in sources.items() if items}
        for f in as_completed(futures):
            name, result = f.result()
            if result["kept"]:
                filtered[name] = result
                print(f"    {name}: kept {len(result['kept'])}, dropped {result.get('dropped', 0)}")
    return filtered


def _pass_populate(client, filtered, max_iterations=5):
    """Extract facts from signals. LLM can research via SQL before committing."""
    print("\n💾 Extracting facts...")

    taxonomy = "\n".join(f"- `{c['slug']}` — {c['path']}: {c['description']}" for c in db.get_category_list_flat())
    stats = db.get_stats()
    hot = db.get_hot_categories(limit=15)
    gaps = db.get_market_gaps(limit=10)

    ctx = [f"## Taxonomy:\n{taxonomy}",
           f"\n## DB: {stats['products']} products, {stats['painpoints']} painpoints, "
           f"{stats['funding_rounds']} funding rounds, {stats['quotes']} quotes"]
    if hot:
        ctx.append("\n## Active categories:")
        for h in hot:
            ctx.append(f"- {h['category']} ({h['slug']}): {h['painpoints']}pp, {h['products']}prod, {h['total_signals']}sig")
    if gaps:
        ctx.append("\n## Market gaps:")
        for g in gaps:
            r = f"{g['pain_to_product_ratio']}:1" if g["pain_to_product_ratio"] < 100 else "∞"
            ctx.append(f"- {g['category']}: {g['painpoint_count']}pp, {g['product_count']}prod ({r})")
    ctx.append("\n## You have SQL access (SELECT only) and web search. Use both.")
    ctx.append(f"\n## New signals:\n{_curated_text(filtered)}")

    messages = [
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": "\n".join(ctx)},
    ]

    for iteration in range(max_iterations):
        try:
            data = json.loads(llm_call(client, messages))
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return

        sql_queries = data.get("sql_queries", [])
        web_searches = data.get("web_searches", [])

        if data.get("done", True) or (not sql_queries and not web_searches and any(
                data.get(k) for k in ("products", "painpoints", "funding"))):
            _save_extracted(data)
            if iteration > 0:
                print(f"  (after {iteration + 1} iterations)")
            return

        if sql_queries or web_searches:
            research_parts = []
            if sql_queries:
                print(f"  🔎 Iteration {iteration + 1}: {len(sql_queries)} SQL queries...")
                research_parts.append("SQL results:\n\n" + "\n\n".join(execute_sql_queries(sql_queries)))
            if web_searches:
                print(f"  🌐 Iteration {iteration + 1}: {len(web_searches)} web searches...")
                for ws in web_searches[:3]:
                    q = ws.get("query", "")
                    reason = ws.get("reason", "")
                    print(f"    🔍 {reason}: \"{q}\"")
                    result = web_search(client, q)
                    research_parts.append(f"Web search ({reason}): {q}\nResult:\n{result[:1500]}")

            messages.append({"role": "assistant", "content": json.dumps(data)})
            messages.append({"role": "user", "content": "\n\n".join(research_parts)})


def _save_extracted(data):
    """Save LLM-extracted facts to DB."""
    for p in data.get("products", []):
        if not p.get("name"):
            continue
        pid = db.upsert_product(p["name"], description=p.get("description"),
            builder=p.get("builder"), tech_complexity=p.get("tech_complexity"),
            viral_trigger=p.get("viral_trigger"), why_viral=p.get("why_viral"))
        for s in p.get("category_slugs", []):
            db.link_product_category(pid, s)

    for pp in data.get("painpoints", []):
        if not pp.get("title"):
            continue
        pid = db.upsert_painpoint(pp["title"], description=pp.get("description", ""),
            severity=pp.get("severity", 5))
        for s in pp.get("category_slugs", []):
            db.link_painpoint_category(pid, s)
        for q in pp.get("quotes", []):
            if q and len(q.strip()) > 10:
                db.add_quote(pid, q, "signals")

    for f in data.get("funding", []):
        if not f.get("company"):
            continue
        rid = db.save_funding_round(f["company"], amount=f.get("amount", ""),
            valuation=f.get("valuation", ""), round_type=f.get("round_type", ""),
            investor_names=f.get("investors"), what_they_build=f.get("what_they_build", ""),
            painpoint_solved=f.get("painpoint_solved", ""), why_funded=f.get("why_funded", ""))
        for s in f.get("category_slugs", []):
            db.link_funding_category(rid, s)

    for link in data.get("product_painpoint_links", []):
        if link.get("product") and link.get("painpoint"):
            db.link_product_painpoint(link["product"], link["painpoint"],
                relationship=link.get("relationship", "addresses"),
                effectiveness=link.get("effectiveness"),
                gap_description=link.get("gap_description", ""),
                gap_type=link.get("gap_type", ""), notes=link.get("notes", ""))

    for cat in data.get("proposed_categories", []):
        if cat.get("name"):
            db.propose_category(cat["name"], cat.get("parent_slug"))

    n = lambda k: len(data.get(k, []))
    print(f"  ✓ {n('products')}prod, {n('painpoints')}pp, {n('funding')}fund, "
          f"{n('product_painpoint_links')}links, {n('proposed_categories')}cat")


def _pass_taxonomy(client):
    """Promote/reject/merge pending categories."""
    pending = db.get_ready_pending_categories(min_signals=2)
    if not pending:
        return

    print("\n🌲 Managing taxonomy...")
    tax = "\n".join(f"- `{c['slug']}` — {c['path']}" for c in db.get_category_list_flat())
    conn = db.get_db()
    llm_cats = conn.execute("SELECT id, name, slug FROM categories WHERE created_by != 'seed'").fetchall()
    conn.close()
    llm_text = "\n".join(f"- id={c['id']} `{c['slug']}` {c['name']}" for c in llm_cats) or "(none)"

    try:
        msgs = [{"role": "system", "content": TAXONOMY_PROMPT},
                {"role": "user", "content": f"## Taxonomy:\n{tax}\n\n## LLM-created:\n{llm_text}\n\n## Pending:\n{json.dumps(pending, indent=2)}"}]
        actions = json.loads(llm_call(client, msgs, max_tokens=1500))
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return

    for p in actions.get("promote", []):
        if db.promote_pending_category(p["pending_id"], p["parent_slug"], p.get("description", "")):
            print(f"  ✅ Promoted #{p['pending_id']}")
    for r in actions.get("reject", []):
        db.reject_pending_category(r["pending_id"])
    for m in actions.get("merge_pending_into_existing", []):
        db.reject_pending_category(m["pending_id"])
    for m in actions.get("merge_existing", []):
        db.merge_categories(m["keep_id"], m["merge_ids"])
    for r in actions.get("rename", []):
        db.rename_category(r["category_id"], r["new_name"], r.get("new_description"))


def _pass_cleanup(client):
    """Delete junk, merge duplicates."""
    painpoints = db.get_all_for_cleanup("painpoints", limit=50)
    products = db.get_all_for_cleanup("products", limit=50)
    if not painpoints and not products:
        return

    print("\n🧹 Cleaning...")
    parts = []
    if painpoints:
        parts.append("## Painpoints\n" + json.dumps(painpoints, indent=1))
    if products:
        parts.append("## Products\n" + json.dumps(products, indent=1))

    try:
        msgs = [{"role": "system", "content": CLEANUP_PROMPT},
                {"role": "user", "content": "\n\n".join(parts)}]
        actions = json.loads(llm_call(client, msgs, max_tokens=1500))
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return

    for table in ("painpoints", "products"):
        ids = actions.get("delete", {}).get(table, [])
        if ids:
            getattr(db, f"delete_{table}")(ids)
            print(f"  🗑️ {len(ids)} {table}")
    for m in actions.get("merge_painpoints", []):
        db.merge_painpoints(m["keep_id"], m["merge_ids"])


# ============================================================
# PUBLIC API
# ============================================================

def analyze(rounds=1):
    """Scrape Reddit + Subriff, extract facts, categorize, clean up."""
    client = get_client()
    config = _load_config()

    for round_num in range(1, rounds + 1):
        print(f"\n{'='*60}\n  ROUND {round_num}/{rounds}\n{'='*60}\n")

        sources = _scrape_all(config)
        run_id = db.start_run(list(sources.keys()))

        for source, items in sources.items():
            if items:
                db.save_signals(run_id, source, items)

        filtered = _pass_filter(client, sources)
        db.compute_percentiles(run_id)
        _pass_populate(client, filtered)
        _pass_taxonomy(client)
        _pass_cleanup(client)

        total = sum(len(v) for v in sources.values())
        kept = sum(len(f.get("kept", [])) for f in filtered.values())
        db.finish_run(run_id, total, kept)

        print(f"\n  📊 Round {round_num} stats:")
        print(f"     Scraped: {total} items from {len(sources)} sources")
        print(f"     Filtered: {kept} signals kept")

    stats = db.get_stats()
    gaps = db.get_market_gaps(limit=5)
    product_gaps = db.get_product_gaps(limit=5)

    print(f"\n{'='*60}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\n  📊 Database totals:")
    print(f"     {stats['products']} products, {stats['painpoints']} painpoints")
    print(f"     {stats['funding_rounds']} funding rounds, {stats['investors']} investors")
    print(f"     {stats['quotes']} user quotes, {stats['categories']} categories")
    print(f"     {stats['signals']} total signals across {stats['runs']} runs")

    if gaps:
        print(f"\n  🎯 Top market gaps:")
        for g in gaps:
            r = f"{g['pain_to_product_ratio']}:1" if g["pain_to_product_ratio"] < 100 else "∞"
            print(f"     {g['category']}: {g['painpoint_count']}pp vs {g['product_count']}prod (ratio {r})")

    if product_gaps:
        print(f"\n  ⚠️  Products failing:")
        for pg in product_gaps:
            eff = f"{pg['effectiveness']}/10" if pg.get('effectiveness') else "?"
            print(f"     {pg['product']} → {pg['painpoint']} (eff: {eff})")

    print(f"\n{'='*60}\n")
    return stats
