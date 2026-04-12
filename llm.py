"""
Shared LLM utilities — debug logging, API calls, SQL execution.
Used by both ingest.py and ideas.py.
"""

import json
import os
import textwrap

from openai import OpenAI

from db.queries import run_sql


def is_debug():
    return os.environ.get("AIPULSE_DEBUG") == "1"


def debug_msg(role, content):
    """Print a message in the LLM conversation when debug mode is on."""
    if not is_debug():
        return
    color = {"system": "🟣", "user": "🔵", "assistant": "🟢"}.get(role, "⚪")
    print(f"\n{color} [{role.upper()}]")
    if len(content) > 2000:
        print(textwrap.indent(content[:1000], "  "))
        print(f"  ... ({len(content) - 1000} chars truncated) ...")
        print(textwrap.indent(content[-500:], "  "))
    else:
        print(textwrap.indent(content, "  "))
    print()


_logged = {}  # id(input) -> count already printed (for multi-turn lists)


def llm_call(client, instructions, input, *, max_tokens=4000,
             json_mode=True, response_model=None, model="gpt-4.1-mini"):
    """Call OpenAI Responses API with debug logging.

    Args:
        client: OpenAI client instance.
        instructions: System-level instructions (maps to ``instructions``).
        input: User message string, or a list of message dicts for
            multi-turn conversations (``[{"role": "user", "content": ...}, ...]``).
        max_tokens: Maximum output tokens.
        json_mode: When True and *response_model* is None, enforce JSON
            object output.  Ignored when *response_model* is set.
        response_model: Optional Pydantic ``BaseModel`` class.  When
            provided, the SDK enforces the schema server-side and returns
            a validated model instance instead of a string.
        model: Model name.

    Returns:
        ``str`` when *response_model* is None, otherwise a Pydantic model
        instance.
    """
    if is_debug():
        if isinstance(input, list):
            key = id(input)
            already = _logged.get(key, 0)
            if already == 0:
                debug_msg("system", instructions)
            for m in input[already:]:
                debug_msg(m["role"], m["content"])
            _logged[key] = len(input)
        else:
            debug_msg("system", instructions)
            debug_msg("user", input)

    if response_model is not None:
        resp = client.responses.parse(
            model=model,
            instructions=instructions,
            input=input,
            text_format=response_model,
            max_output_tokens=max_tokens,
        )
        result = resp.output_parsed
        if is_debug():
            debug_msg("assistant", str(result))
        return result

    kwargs = {
        "model": model,
        "instructions": instructions,
        "input": input,
        "max_output_tokens": max_tokens,
    }
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}

    resp = client.responses.create(**kwargs)
    content = resp.output_text

    if is_debug():
        debug_msg("assistant", content)

    return content


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=api_key)


def execute_sql_queries(sql_queries, max_queries=5, max_rows=20):
    """Execute a batch of SQL queries and return formatted results for LLM consumption."""
    results = []
    for sq in sql_queries[:max_queries]:
        query = sq.get("query", "")
        reason = sq.get("reason", "")
        result = run_sql(query)
        if isinstance(result, dict) and "error" in result:
            results.append(f"Query: {query}\nError: {result['error']}")
            print(f"    ✗ {reason}: {result['error']}")
        else:
            rows = result[:max_rows]
            results.append(
                f"Query ({reason}): {query}\n"
                f"Results ({len(result)} rows, showing {len(rows)}):\n"
                f"{json.dumps(rows, indent=1, default=str)}"
            )
            print(f"    ✓ {reason}: {len(result)} rows")
    return results


def web_search(client, query):
    """Run a web search via OpenAI Responses API. Returns text with citations."""
    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",
            tools=[{"type": "web_search"}],
            input=query,
        )
        return resp.output_text
    except Exception as e:
        return f"[web search failed: {e}]"


DB_SCHEMA = """Tables:
- painpoints(id, title, description, severity 1-10, signal_count)
- products(id, name, description, builder, tech_complexity LOW/MED/HIGH, viral_trigger, why_viral, status)
- quotes(id, painpoint_id→painpoints, product_id→products, text, source, sentiment neg/pos/neutral)
- funding_rounds(id, product_id→products, amount, amount_usd, valuation, round_type, what_they_build, why_funded)
- categories(id, name, slug, parent_id→categories — hierarchical 2 levels)
- investors(id, name, type vc/angel/corporate)

Junctions:
- product_painpoints(product_id, painpoint_id, relationship addresses/fails_at/partial, effectiveness 1-10, gap_description, gap_type pricing/performance/ux/scope/reliability/integration)
- painpoint_categories(painpoint_id, category_id)
- product_categories(product_id, category_id)
- funding_categories(round_id, category_id)
- round_painpoints(round_id, painpoint_id)
- round_investors(round_id, investor_id, lead 0/1)

Useful joins:
- Pain with categories: painpoints p JOIN painpoint_categories pc ON pc.painpoint_id=p.id JOIN categories c ON c.id=pc.category_id
- Product effectiveness: products p JOIN product_painpoints pp ON pp.product_id=p.id JOIN painpoints pa ON pa.id=pp.painpoint_id
- Funding by category: funding_rounds f JOIN funding_categories fc ON fc.round_id=f.id JOIN categories c ON c.id=fc.category_id"""
