"""
Idea generation — propose_idea() reads the facts database and generates
buildable product ideas. The LLM drives its own research loop via SQL.
Pure read — doesn't modify the DB.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import database as db
from llm import get_client, llm_call, execute_sql_queries, web_search, DB_SCHEMA


IDEA_PROMPT = f"""You are looking for the best product to build right now — something a solo
dev can ship in a weekend that addresses a real, validated pain.

You have a market intelligence database + SQL access + web search.

BEFORE proposing any idea, you MUST:
1. Web search to check if it already exists (search by what it does, not just the name)
2. If 3+ competitors exist, don't propose it — find something else
3. Verify the painpoint is real and specific, not generic
4. Confirm the MVP is actually buildable in a weekend (not "just use PyTorch")

Each response must be JSON:
- To research: {{"done": false, "sql_queries": [...], "web_searches": [...]}}
  sql_queries: [{{"query": "SELECT ...", "reason": "..."}}]
  web_searches: [{{"query": "search terms", "reason": "..."}}]
- To propose:  {{"done": true, "ideas": [{{"name": "...", "what_it_does": "...", "target_user": "...",
  "viral_hook": "...", "mvp_scope": "...", "tech_stack": "...", "painpoint_ids": [1,2],
  "market_reasoning": "..."}}]}}

Return FEWER, BETTER ideas. One great idea beats three mediocre ones.
If nothing genuinely underserved emerges, return empty ideas — don't force it.

DB schema: {DB_SCHEMA}"""


COMPETITION_PROMPT = """Assess if an idea already exists. Return JSON:
{"status": "open_field"|"partially_exists"|"crowded", "competitors": [...], "differentiation": "..."}"""


CRITIC_PROMPT = f"""You are a harsh critic reviewing product ideas. Your job is to REJECT bad ideas.

You MUST web search for EVERY idea to check:
1. Does this product already exist? Search for it by name AND by what it does.
2. Are there well-known competitors the proposer missed? (e.g., llama.cpp, Ollama, vLLM for inference)
3. Is the market actually underserved or is the proposer just unaware of existing solutions?

For EACH idea, include at least 2 web searches. Do not skip this.

Return JSON:
{{
  "reviews": [
    {{
      "idea_name": "...",
      "verdict": "accept" | "revise" | "reject",
      "issues": ["specific issue 1", "..."],
      "web_searches": [{{"query": "...", "reason": "..."}}],
      "suggestion": "how to fix if revise"
    }}
  ]
}}

Rejection criteria (reject if ANY apply):
- 3+ established competitors already exist
- The "gap" the idea claims to fill doesn't actually exist
- The viral hook is generic ("saves time", "easy to use")

Be brutal. Most ideas should be rejected. Only accept ideas where you searched
and confirmed the gap is real."""


# ============================================================
# HELPERS
# ============================================================

def _search_hn(query, limit=5):
    try:
        r = requests.get("https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story", "hitsPerPage": limit}, timeout=10)
        r.raise_for_status()
        return [f"[{h.get('points', 0)}↑] {h.get('title', '')}"
                for h in r.json().get("hits", []) if h.get("points", 0) > 3]
    except Exception:
        return []


def _search_github(query, limit=5):
    try:
        r = requests.get("https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "per_page": limit},
            headers={"User-Agent": "AIPulse/1.0"}, timeout=10)
        r.raise_for_status()
        return [f"{r['full_name']} ({r['stargazers_count']}★) — {(r.get('description') or '')[:80]}"
                for r in r.json().get("items", []) if r.get("stargazers_count", 0) > 50]
    except Exception:
        return []


def _check_competition(client, idea):
    query = f"{idea.get('name', '')} {idea.get('what_it_does', '')[:50]}"
    with ThreadPoolExecutor(max_workers=2) as pool:
        hn = pool.submit(_search_hn, query).result()
        gh = pool.submit(_search_github, query).result()

    if not hn and not gh:
        return {"status": "open_field", "competitors": [], "differentiation": ""}

    search = ("HN:\n" + "\n".join(hn) + "\n" if hn else "") + ("GitHub:\n" + "\n".join(gh) if gh else "")
    try:
        msgs = [{"role": "system", "content": COMPETITION_PROMPT},
                {"role": "user", "content": f"Idea: {idea['name']} — {idea['what_it_does']}\n\n{search}"}]
        return json.loads(llm_call(client, msgs, max_tokens=500))
    except Exception:
        return {"status": "unknown", "competitors": [], "differentiation": ""}


def _load_context(focus=None, limit=20):
    if focus:
        facts = db.get_facts_by_category(focus, include_descendants=True)
        painpoints, products, funding = facts["painpoints"][:limit], facts["products"][:limit], facts["funding"][:limit]
    else:
        painpoints = db.get_top_painpoints(limit=limit)
        conn = db.get_db()
        products = [dict(r) for r in conn.execute(
            "SELECT id, name, description, status FROM products ORDER BY last_updated DESC LIMIT ?", (limit,)).fetchall()]
        conn.close()
        funding = db.get_recent_funding(limit=limit)

    pp_ids = [pp["id"] for pp in painpoints]
    return {
        "painpoints": painpoints,
        "products": products,
        "funding": funding,
        "quotes": db.get_quotes_for_painpoints(pp_ids, per_painpoint=2),
        "gaps": db.get_market_gaps(limit=10),
        "product_gaps": db.get_product_gaps(limit=10),
        "hot": db.get_hot_categories(limit=10),
    }


def _load_context_for_ids(painpoint_ids, limit=20):
    """Load context scoped to a specific set of painpoint IDs — used by the
    demo so idea generation keys off exactly the pains extracted from one run.
    Market gaps, recent products, and funding are still pulled globally so the
    model has cross-community context when reasoning about competition."""
    if not painpoint_ids:
        return None

    conn = db.get_db()
    placeholders = ",".join("?" * len(painpoint_ids))
    rows = conn.execute(
        f"""
        SELECT pp.id, pp.title, pp.description, pp.severity, pp.signal_count,
               GROUP_CONCAT(DISTINCT c.name) AS categories
        FROM painpoints pp
        LEFT JOIN painpoint_categories pc ON pc.painpoint_id = pp.id
        LEFT JOIN categories c ON c.id = pc.category_id
        WHERE pp.id IN ({placeholders})
        GROUP BY pp.id
        ORDER BY pp.severity DESC, pp.signal_count DESC
        """,
        painpoint_ids,
    ).fetchall()
    painpoints = [dict(r) for r in rows][:limit]

    products = [dict(r) for r in conn.execute(
        "SELECT id, name, description, status FROM products ORDER BY last_updated DESC LIMIT ?",
        (limit,),
    ).fetchall()]
    conn.close()

    pp_ids = [pp["id"] for pp in painpoints]
    return {
        "painpoints": painpoints,
        "products": products,
        "funding": db.get_recent_funding(limit=limit),
        "quotes": db.get_quotes_for_painpoints(pp_ids, per_painpoint=2),
        "gaps": db.get_market_gaps(limit=10),
        "product_gaps": db.get_product_gaps(limit=10),
        "hot": db.get_hot_categories(limit=10),
    }


def _format_context(ctx):
    parts = []

    if ctx.get("gaps"):
        parts.append("## MARKET GAPS")
        for g in ctx["gaps"]:
            funding = f"${g['funding_usd'] // 1_000_000}M" if g["funding_usd"] > 0 else "none"
            r = f"{g['pain_to_product_ratio']}:1" if g["pain_to_product_ratio"] < 100 else "∞"
            parts.append(f"- {g['category']}: {g['painpoint_count']}pp, {g['product_count']}prod, ratio {r}, funding {funding}")

    if ctx.get("product_gaps"):
        parts.append("\n## PRODUCT GAPS")
        for fp in ctx["product_gaps"]:
            eff = f"{fp['effectiveness']}/10" if fp.get("effectiveness") else "?"
            parts.append(f"- {fp['product']} → {fp['painpoint']} [{fp['relationship']}, eff:{eff}, type:{fp.get('gap_type', '?')}]")
            if fp.get("gap_description"):
                parts.append(f"  GAP: {fp['gap_description']}")

    if ctx["painpoints"]:
        parts.append("\n## TOP PAINPOINTS")
        for pp in ctx["painpoints"]:
            parts.append(f"- ID:{pp['id']} [{pp.get('categories') or ''}] sev {pp.get('severity', '?')}/10, "
                         f"{pp.get('signal_count', 0)} signals: {pp['title']}")
            if pp.get("description"):
                parts.append(f"  {pp['description'][:200]}")
            for q in ctx["quotes"].get(pp["id"], []):
                parts.append(f"  > \"{q[:150]}\"")

    if ctx["products"]:
        parts.append("\n## PRODUCTS")
        for p in ctx["products"]:
            parts.append(f"- ID:{p['id']} {p['name']}: {(p.get('description') or '')[:150]}")

    if ctx["funding"]:
        parts.append("\n## FUNDING")
        for f in ctx["funding"]:
            parts.append(f"- {f.get('product_name') or '?'} raised {f.get('amount', '?')}: {(f.get('what_they_build') or '')[:100]}")

    if ctx.get("hot"):
        parts.append("\n## HOT CATEGORIES")
        for h in ctx["hot"]:
            parts.append(f"- {h['category']}: {h['total_signals']}sig, {h['funding_rounds']}fund")

    return "\n".join(parts)


# ============================================================
# PUBLIC API
# ============================================================

def _emit(on_event, stage, status, detail=""):
    """Fire an event if a listener is attached. Never raises."""
    if on_event:
        try:
            on_event({"stage": stage, "status": status, "detail": detail})
        except Exception:
            pass


def propose_idea(focus=None, count=3, check_competition=True, max_iterations=5,
                 painpoint_ids=None, on_event=None):
    """Generate ideas. LLM researches via SQL until confident, then proposes.

    If `painpoint_ids` is passed, scope the context to exactly those
    painpoints (used by the demo to generate ideas for a single subreddit's
    extracted pains). Otherwise fall back to the global/focus loader.

    When `on_event` is provided, each research step, proposed idea, and
    critic verdict is streamed as a dict so the UI can show progress.
    """
    client = get_client()
    if painpoint_ids:
        ctx = _load_context_for_ids(painpoint_ids)
    else:
        ctx = _load_context(focus=focus)

    if not ctx or not ctx["painpoints"]:
        print("⚠️  No painpoints in DB. Run `python main.py analyze` first.")
        _emit(on_event, "error", "error", "No painpoints available")
        return []

    scope_label = (
        f" for {len(painpoint_ids)} painpoint(s)"
        if painpoint_ids
        else (f" (focus: {focus})" if focus else "")
    )
    print(f"🧠 Generating {count} ideas{scope_label}...")
    _emit(on_event, "start", "running", f"brainstorming {count} ideas from {len(ctx['painpoints'])} painpoints")

    messages = [
        {"role": "system", "content": IDEA_PROMPT},
        {"role": "user", "content": f"{_format_context(ctx)}\n\nGenerate {count} ideas. Research first if needed."},
    ]

    ideas = []
    for iteration in range(max_iterations):
        try:
            data = json.loads(llm_call(client, messages))
        except Exception as e:
            print(f"✗ Iteration {iteration + 1} failed: {e}")
            _emit(on_event, "research", "error", f"iter {iteration + 1} failed: {e}")
            break

        done = data.get("done", False)
        sql = data.get("sql_queries", []) or []
        web = data.get("web_searches", []) or []
        ideas = data.get("ideas", []) or []

        if done or (not sql and not web and ideas):
            print(f"  ✓ Done after {iteration + 1} iteration(s)")
            ideas = ideas[:count]
            _emit(on_event, "propose", "done", f"model proposed {len(ideas)} candidates")
            for idea in ideas:
                name = (idea.get("name") if isinstance(idea, dict) else "") or "Untitled"
                what = (idea.get("what_it_does") if isinstance(idea, dict) else "") or ""
                _emit(on_event, "idea", "running", f"💡 {name} — {what[:140]}")
            break

        if sql or web:
            research_parts = []
            if sql:
                print(f"  🔎 Iteration {iteration + 1}: {len(sql)} SQL queries...")
                _emit(on_event, "research", "running",
                      f"iter {iteration + 1}: {len(sql)} SQL queries")
                for sq in sql[:3]:
                    reason = sq.get("reason", "") if isinstance(sq, dict) else ""
                    if reason:
                        _emit(on_event, "research", "running", f"sql · {reason}")
                research_parts.append("SQL results:\n\n" + "\n\n".join(execute_sql_queries(sql)))
            if web:
                print(f"  🌐 Iteration {iteration + 1}: {len(web)} web searches...")
                _emit(on_event, "research", "running",
                      f"iter {iteration + 1}: {len(web)} web searches")
                for ws in web[:3]:
                    q = ws.get("query", "") if isinstance(ws, dict) else ""
                    reason = ws.get("reason", "") if isinstance(ws, dict) else ""
                    print(f"    🔍 {reason}: \"{q}\"")
                    _emit(on_event, "research", "running", f"web · {reason or q}")
                    result = web_search(client, q)
                    research_parts.append(f"Web search ({reason}): {q}\nResult:\n{result[:1500]}")

            messages.append({"role": "assistant", "content": json.dumps(data)})
            messages.append({"role": "user", "content": "\n\n".join(research_parts)
                             + f"\n\nContinue (done:false) or generate {count} ideas (done:true)."})
        else:
            messages.append({"role": "assistant", "content": json.dumps(data)})
            messages.append({"role": "user", "content": f"Generate {count} ideas now. done:true."})

    if check_competition and ideas:
        print(f"\n🔍 Checking competition ({len(ideas)} ideas)...")
        _emit(on_event, "competition", "running", f"checking competition for {len(ideas)} ideas")
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_check_competition, client, idea): idea for idea in ideas}
            for f in as_completed(futures):
                futures[f]["competition"] = f.result()

    # Critic loop — another LLM reviews and challenges the ideas
    if ideas:
        ideas = _critic_loop(client, ideas, max_rounds=2, on_event=on_event)

    _emit(on_event, "done", "done", f"final: {len(ideas)} idea(s) survived")
    return ideas


def _critic_loop(client, ideas, max_rounds=2, on_event=None):
    """Run ideas through a critic LLM that fact-checks and challenges them.
    Ideas can be accepted, revised, or rejected. Runs until all accepted or max rounds."""
    for critic_round in range(max_rounds):
        print(f"\n🧐 Critic round {critic_round + 1}...")
        _emit(on_event, "critic", "running",
              f"critic round {critic_round + 1} — reviewing {len(ideas)} idea(s)")

        ideas_text = json.dumps(ideas, indent=2, default=str)
        messages = [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": f"Review these ideas:\n{ideas_text}"},
        ]

        try:
            data = json.loads(llm_call(client, messages))
        except Exception as e:
            print(f"  ✗ Critic failed: {e}")
            _emit(on_event, "critic", "error", f"critic failed: {e}")
            break

        reviews = data.get("reviews", []) or []

        # Execute any web searches the critic wants
        web_searches = []
        for r in reviews:
            if isinstance(r, dict):
                web_searches.extend(r.get("web_searches", []) or [])

        if web_searches:
            print(f"  🌐 Critic verifying {len(web_searches)} claims...")
            _emit(on_event, "critic", "running",
                  f"critic verifying {len(web_searches)} claim(s)")
            search_results = []
            for ws in web_searches[:5]:
                q = ws.get("query", "") if isinstance(ws, dict) else ""
                reason = ws.get("reason", "") if isinstance(ws, dict) else ""
                print(f"    🔍 {reason}")
                _emit(on_event, "critic", "running", f"web · {reason or q}")
                result = web_search(client, q)
                search_results.append(f"Search ({reason}): {q}\nResult: {result[:1000]}")

            # Feed search results back and get updated reviews
            messages.append({"role": "assistant", "content": json.dumps(data)})
            messages.append({"role": "user", "content":
                "Web search results:\n\n" + "\n\n".join(search_results) +
                "\n\nUpdate your reviews based on these findings."})
            try:
                data = json.loads(llm_call(client, messages))
                reviews = data.get("reviews", []) or []
            except Exception:
                pass

        # Process reviews
        review_by_name = {
            (r.get("idea_name", "") or "").lower(): r
            for r in reviews if isinstance(r, dict)
        }
        surviving = []
        for idea in ideas:
            name = (idea.get("name", "") or "").lower()
            review = review_by_name.get(name, {})
            verdict = review.get("verdict", "accept")
            issues = review.get("issues", []) or []
            display_name = idea.get("name") or "Untitled"

            if verdict == "accept":
                print(f"  ✅ {display_name}: accepted")
                _emit(on_event, "verdict", "done", f"✓ accepted · {display_name}")
                surviving.append(idea)
            elif verdict == "revise":
                suggestion = review.get("suggestion", "") or ""
                print(f"  ✏️  {display_name}: revised — {suggestion[:80]}")
                _emit(on_event, "verdict", "running",
                      f"✏ revised · {display_name} — {suggestion[:120]}")
                idea["critic_notes"] = suggestion
                idea["critic_issues"] = issues
                surviving.append(idea)
            else:
                reasons = ", ".join(issues)[:160] or "no reason given"
                print(f"  ❌ {display_name}: rejected — {reasons[:80]}")
                _emit(on_event, "verdict", "error",
                      f"✗ rejected · {display_name} — {reasons}")

        ideas = surviving

        # If all accepted, stop
        if all(review_by_name.get((i.get("name", "") or "").lower(), {}).get("verdict") == "accept"
               for i in ideas):
            break

    return ideas


def print_ideas(ideas):
    for i, idea in enumerate(ideas, 1):
        print(f"\n{'='*60}\nIDEA {i}: {idea.get('name', '?')}\n{'='*60}")
        for key in ("what_it_does", "target_user", "viral_hook", "mvp_scope", "tech_stack", "market_reasoning"):
            val = idea.get(key)
            if val:
                label = key.replace("_", " ").title()
                print(f"{label}: {val}")
        comp = idea.get("competition")
        if comp:
            print(f"\nCompetition: {comp.get('status', '?').upper()}")
            if comp.get("competitors"):
                print(f"Existing: {', '.join(comp['competitors'])}")
            if comp.get("differentiation"):
                print(f"Edge: {comp['differentiation']}")
        if idea.get("critic_notes"):
            print(f"\nCritic: {idea['critic_notes']}")
        if idea.get("critic_issues"):
            print(f"Issues: {', '.join(idea['critic_issues'])}")
