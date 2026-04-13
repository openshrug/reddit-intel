"""
Shared LLM utilities — debug logging, API calls, SQL execution.

Used across the codebase: db/llm_naming.py (category naming),
painpoint_extraction/ (extractor + judge), demo/analyzer.py.
"""

import json
import logging
import os
import textwrap
import threading
import time

from openai import OpenAI

log = logging.getLogger(__name__)

from db.queries import run_sql

# Global cap on concurrent OpenAI API calls across the whole process.
# OpenAI tier-1 limits are 3000 RPM for embeddings (~50 RPS) and ~500 RPM
# for nano-class models. With ~200ms per embedding call and ~3s per
# completion, ~10 in flight stays comfortably below the per-second cap
# (≈ rpm/2 / typical_rps ≈ 25/2 ≈ 12, picked 10 as a safer bound).
# Both `llm_call` (here) and `OpenAIEmbedder._embed_with_retry` (in
# db/embeddings.py) acquire this semaphore so all paths share one budget.
OPENAI_CONCURRENCY = 10
OPENAI_API_SEMAPHORE = threading.BoundedSemaphore(OPENAI_CONCURRENCY)


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


class TokenCounter:
    """Thread-safe accumulator for API token usage across multiple calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0

    def add(self, usage):
        """Add usage from a single API response."""
        if usage is None:
            return
        with self._lock:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
                self.reasoning_tokens += getattr(
                    usage.output_tokens_details, "reasoning_tokens", 0
                )

    @property
    def text_tokens(self):
        return self.output_tokens - self.reasoning_tokens

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    def as_dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "text_tokens": self.text_tokens,
            "total_tokens": self.total_tokens,
        }


def llm_call(client, instructions, input, *, max_tokens=4000,
             json_mode=True, response_model=None, model="gpt-5-nano",
             reasoning_effort=None, token_counter=None, retries=2,
             backoff_base=4):
    """Call OpenAI Responses API with debug logging and exponential backoff.

    Args:
        client: OpenAI client instance.
        instructions: System-level instructions (maps to ``instructions``).
        input: User message string, or a list of message dicts for
            multi-turn conversations (``[{"role": "user", "content": ...}, ...]``).
        max_tokens: Maximum output tokens.  ``None`` omits the cap (model
            default).
        json_mode: When True and *response_model* is None, enforce JSON
            object output.  Ignored when *response_model* is set.
        response_model: Optional Pydantic ``BaseModel`` class.  When
            provided, the SDK enforces the schema server-side and returns
            a validated model instance instead of a string.
        model: Model name.
        reasoning_effort: Optional reasoning effort level for reasoning
            models (``"low"``, ``"medium"``, ``"high"``, etc.).
        token_counter: Optional ``TokenCounter`` to accumulate usage.
        retries: Number of retry attempts on transient errors (default 2).
        backoff_base: Base delay in seconds for exponential backoff
            (default 4 → delays of 4s, 8s).

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

    def _do_call():
        if response_model is not None:
            parse_kwargs = dict(
                model=model,
                instructions=instructions,
                input=input,
                text_format=response_model,
            )
            if max_tokens is not None:
                parse_kwargs["max_output_tokens"] = max_tokens
            if reasoning_effort is not None:
                parse_kwargs["reasoning"] = {"effort": reasoning_effort}
            resp = client.responses.parse(**parse_kwargs)
            if token_counter is not None:
                token_counter.add(resp.usage)
            if resp.output_parsed is None:
                refusal = getattr(resp, "refusal", None)
                raw = resp.output_text if hasattr(resp, "output_text") else str(resp)
                raise RuntimeError(
                    f"Structured output returned None"
                    f"{f' (refusal: {refusal})' if refusal else ''}"
                    f" — raw output: {raw[:500]}"
                )
            if is_debug():
                debug_msg("assistant", str(resp.output_parsed))
            return resp.output_parsed

        kwargs = {
            "model": model,
            "instructions": instructions,
            "input": input,
        }
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}

        resp = client.responses.create(**kwargs)
        if token_counter is not None:
            token_counter.add(resp.usage)
        content = resp.output_text

        if is_debug():
            debug_msg("assistant", content)

        return content

    last_exc = None
    for attempt in range(1 + retries):
        try:
            # Acquire the global API concurrency semaphore around the
            # network call. Sleeps between retries happen WITHOUT holding
            # the slot so retrying threads don't starve fresh callers.
            with OPENAI_API_SEMAPHORE:
                return _do_call()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            delay = backoff_base * (2 ** attempt)
            log.warning("llm_call attempt %d/%d failed (%s), retrying in %ds",
                        attempt + 1, 1 + retries, exc, delay)
            time.sleep(delay)
    raise last_exc  # unreachable, but satisfies type checkers


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
