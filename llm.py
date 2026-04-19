"""
Shared LLM utilities — debug logging, API calls, SQL execution.

Used across the codebase: db/llm_naming.py (category naming),
painpoint_extraction/ (extractor + judge), demo/analyzer.py.
"""

import json
import logging
import os
import random
import re
import textwrap
import threading
import time

from openai import APIError, APIStatusError, OpenAI, RateLimitError

log = logging.getLogger(__name__)

from db.queries import run_sql

# Per-model concurrency caps. Phase 1 split (was a single 30-slot pool
# shared across completions + embeddings).
#
# Embeddings get a much larger budget because text-embedding-3-small at
# Tier 1 has ~5x the RPM and ~5x the TPM of gpt-5-nano, AND embedding
# calls are typically <2s round-trip vs 5-15s for a reasoning completion
# — so a single embedding batch holds its slot 5-10x less time than a
# completion. Sharing the old 30-slot pool meant a fresh embedding call
# could sit behind 30 in-flight completions for 10s+ even though its
# own TPM bucket was 80% empty.
#
# The completion bucket stays at 30 (sized for the sweep's uncat-review
# fan-out at ~2s per gpt-4.1-mini call ≈ 15 RPS, well under the
# tier-1 RPM limit). Sleeps between retries happen OUTSIDE the semaphore
# so backing-off threads don't starve fresh callers.
#
# Phase 2 (token-velocity tracker) is documented in
# `docs/IMPROVEMENTS.md` and would subsume both knobs.
OPENAI_COMPLETION_CONCURRENCY = 30
OPENAI_EMBEDDING_CONCURRENCY = 80
OPENAI_COMPLETION_SEMAPHORE = threading.BoundedSemaphore(OPENAI_COMPLETION_CONCURRENCY)
OPENAI_EMBEDDING_SEMAPHORE = threading.BoundedSemaphore(OPENAI_EMBEDDING_CONCURRENCY)

# Deprecated aliases kept for downstream consumers (`reddit-intel-closed`
# imports `OPENAI_API_SEMAPHORE` to swap in a distributed implementation
# at process boot — see PIPELINE.md §12). Both alias the COMPLETION
# bucket because that's the throttled path historical callers were
# trying to control. Will be removed in a future major bump once
# downstream migrates to the per-model names.
OPENAI_CONCURRENCY = OPENAI_COMPLETION_CONCURRENCY
OPENAI_API_SEMAPHORE = OPENAI_COMPLETION_SEMAPHORE

# Rate-limit-aware retry policy. RateLimitError gets a much larger
# attempt budget than transient network/5xx errors because TPM windows
# can stay saturated for 30-60s during a burst (we observed ~15s windows
# in live runs against gpt-5-nano @ 200K TPM). Other transient errors
# (5xx, connection blips) usually resolve in seconds, so the smaller
# budget there avoids hiding real outages behind long retry stacks.
# `_MAX_TOTAL_WAIT_S` caps cumulative sleep time so a permanent quota
# problem still surfaces quickly instead of pinning a worker thread.
_RATE_LIMIT_MAX_ATTEMPTS = 6
_TRANSIENT_MAX_ATTEMPTS = 3
_BACKOFF_BASE_DEFAULT = 4.0
_MAX_TOTAL_WAIT_S = 60.0
# OpenAI's 429 body says "Please try again in <N>s" or "<N>ms". When
# the Retry-After header is missing or unparseable we fall back to this
# regex against the exception body / message so we still respect the
# server's own hint.
_RETRY_AFTER_BODY_RE = re.compile(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)\b", re.I)


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
        # Cached input tokens reported by OpenAI prompt caching. Surfaces
        # how much of `input_tokens` was served from a cached prefix —
        # the ratio is a direct signal of cache effectiveness.
        self.cached_input_tokens = 0

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
            if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
                self.cached_input_tokens += getattr(
                    usage.input_tokens_details, "cached_tokens", 0
                ) or 0

    @property
    def text_tokens(self):
        return self.output_tokens - self.reasoning_tokens

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_pct(self):
        """Share of input tokens served from the prompt cache (0–100)."""
        if self.input_tokens == 0:
            return 0.0
        return 100.0 * self.cached_input_tokens / self.input_tokens

    def as_dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "text_tokens": self.text_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_hit_pct": self.cache_hit_pct,
        }


def llm_call(client, instructions, input, *, max_tokens=4000,
             json_mode=True, response_model=None, model="gpt-5-nano",
             reasoning_effort=None, token_counter=None):
    """Call OpenAI Responses API with debug logging and rate-limit-aware retries.

    Retries are driven by `call_with_openai_retry`: 429s honour the
    server's `Retry-After` (or body hint) with jitter, transient 5xx /
    network errors get full-jitter exponential backoff, and 4xx errors
    raise immediately. Total wall-time per call is capped at
    `_MAX_TOTAL_WAIT_S`.

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

    return call_with_openai_retry(_do_call, label="llm_call")


def call_with_openai_retry(do_call, *, label="openai_call",
                           backoff_base=_BACKOFF_BASE_DEFAULT,
                           semaphore=None):
    """Run ``do_call()`` under a per-model semaphore with rate-limit-aware retry.

    Used by both `llm_call` (Responses API → completions) and
    `OpenAIEmbedder._embed_with_retry` (Embeddings API). Each path
    passes its own bucket so completion fan-out can't starve embedding
    callers (and vice-versa) — see the module-level
    `OPENAI_COMPLETION_SEMAPHORE` / `OPENAI_EMBEDDING_SEMAPHORE` docs
    for the sizing rationale.

    Retry behaviour:
        - `RateLimitError` (HTTP 429): up to ``_RATE_LIMIT_MAX_ATTEMPTS``
          attempts. Sleep length is OpenAI's ``Retry-After`` /
          ``retry-after-ms`` header when present, otherwise the
          ``"try again in <N>(ms|s)"`` hint embedded in the response
          body, otherwise full-jitter exponential backoff. A small extra
          jitter is added on top of server hints so concurrent retriers
          don't all wake on the same TPM tick.
        - `APIStatusError` with 5xx, `APIConnectionError`,
          `APITimeoutError`: up to ``_TRANSIENT_MAX_ATTEMPTS`` attempts
          with full-jitter exponential backoff.
        - Anything else (4xx caller errors, programming bugs, etc.):
          re-raised on the first attempt — retrying won't help.

    The cumulative wall-time spent sleeping is capped at
    ``_MAX_TOTAL_WAIT_S`` so a permanent quota issue can't pin a worker
    thread for minutes.

    Sleeps happen OUTSIDE the semaphore so backing-off threads don't
    hold a slot away from fresh callers.

    Args:
        do_call: Zero-arg callable that performs one API request and
            returns whatever the caller wants (response object,
            parsed model, embedding list, ...).
        label: Short string used in log messages — distinguishes
            ``llm_call`` from ``embed`` failures in the logs.
        backoff_base: Base delay (seconds) for full-jitter exponential
            backoff when no server hint is available.
        semaphore: `threading.BoundedSemaphore` to acquire around each
            attempt. Defaults to `OPENAI_COMPLETION_SEMAPHORE` because
            completions are the historical caller; the embeddings path
            passes `OPENAI_EMBEDDING_SEMAPHORE` explicitly.

    Returns:
        Whatever ``do_call`` returns on success.

    Raises:
        The last exception raised by ``do_call`` after the retry budget
        or wall-time cap is exhausted, or immediately if the exception
        class is not retryable.
    """
    if semaphore is None:
        semaphore = OPENAI_COMPLETION_SEMAPHORE
    last_exc = None
    total_waited = 0.0
    attempt = 0
    while True:
        try:
            with semaphore:
                return do_call()
        except Exception as exc:
            last_exc = exc
            max_attempts = _classify_for_retry(exc)
            if max_attempts == 0 or attempt + 1 >= max_attempts:
                raise
            delay = _compute_retry_delay(exc, attempt, backoff_base=backoff_base)
            remaining = _MAX_TOTAL_WAIT_S - total_waited
            if remaining <= 0:
                raise
            delay = min(delay, remaining)
            log.warning(
                "%s attempt %d/%d failed (%s: %s), sleeping %.2fs "
                "(waited %.1fs/%ds)",
                label, attempt + 1, max_attempts,
                type(exc).__name__, exc, delay,
                total_waited, int(_MAX_TOTAL_WAIT_S),
            )
            time.sleep(delay)
            total_waited += delay
            attempt += 1
    raise last_exc  # unreachable, but satisfies type checkers


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    # 60s per-request timeout: the SDK default (10min) let a single
    # stuck session hang `parallel_namer_calls` for up to 10min × retries,
    # because that helper walks futures FIFO with no deadline. 60s is
    # well above the p99 observed latency (~5-15s) and combined with the
    # SDK's automatic retry, transient slowness still recovers.
    return OpenAI(api_key=api_key, timeout=60.0)


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


# ---------------------------------------------------------------------------
# Retry helpers (used by call_with_openai_retry)
# ---------------------------------------------------------------------------


def _classify_for_retry(exc) -> int:
    """Return the max attempt count for this exception class.

    0 means "do not retry" — re-raise on first failure. Higher numbers
    are full-attempt budgets (1 + retries), not retry counts on top of
    the first try, so the call site reads as
    ``if attempt + 1 >= max_attempts: raise``.

    `RateLimitError` gets the largest budget because TPM-pressure
    typically clears within 1-3s per attempt and we want enough cycles
    to ride out a saturated minute. Generic `APIError` covers
    `APIConnectionError` and `APITimeoutError` (both transient network
    issues) without us having to enumerate them. 4xx caller errors
    (`BadRequestError`, `AuthenticationError`, etc.) are also
    `APIStatusError` subclasses but we exclude them via the explicit
    status_code >= 500 check — retrying a 401 just wastes the budget.
    """
    if isinstance(exc, RateLimitError):
        return _RATE_LIMIT_MAX_ATTEMPTS
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return _TRANSIENT_MAX_ATTEMPTS
    if isinstance(exc, APIError) and not isinstance(exc, APIStatusError):
        # Connection / timeout / decoding errors — APIStatusError covers
        # HTTP responses, plain APIError covers transport-layer issues.
        return _TRANSIENT_MAX_ATTEMPTS
    return 0


def _compute_retry_delay(exc, attempt: int, *, backoff_base: float) -> float:
    """Compute the next sleep length in seconds.

    For 429s we prefer the server's own hint (header > body regex) and
    add 0.1-1.0s of jitter on top so multiple in-flight retriers don't
    all wake on the exact same TPM tick and re-saturate the bucket. For
    everything else we use full-jitter exponential backoff
    (``random.uniform(0, base * 2**attempt)``), which spreads concurrent
    retriers across the entire backoff window — sharper anti-stampede
    behaviour than a fixed exponential delay.
    """
    hinted = _parse_retry_after(exc)
    if hinted is not None:
        return hinted + random.uniform(0.1, 1.0)
    return random.uniform(0, backoff_base * (2 ** attempt))


def _parse_retry_after(exc) -> float | None:
    """Extract a Retry-After hint (in seconds) from an OpenAI exception.

    Three sources, in priority order:

    1. ``retry-after-ms`` header — OpenAI sometimes returns sub-second
       waits this way; multiplying by 0.001 keeps us honest instead of
       rounding up to a whole second.
    2. ``retry-after`` header — standard HTTP, integer seconds (or
       date-string, but OpenAI uses seconds in practice; we ignore
       date-form values rather than depend on `email.utils`).
    3. Body / message regex — OpenAI embeds ``"Please try again in
       <N>s"`` (or ``"<N>ms"``) in the response body's
       ``error.message``. The SDK stringifies the body into the
       exception message, so a single regex over ``str(exc)`` covers
       both ``exc.body`` (dict) and the message form.

    Returns None when no hint is present so the caller can fall back to
    exponential backoff.
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if headers:
            for header, scale in (("retry-after-ms", 0.001),
                                  ("retry-after", 1.0)):
                v = headers.get(header)
                if v:
                    try:
                        return float(v) * scale
                    except (TypeError, ValueError):
                        # Non-numeric (e.g. HTTP-date form) — ignore and
                        # fall through to the body regex / exponential
                        # backoff. OpenAI uses numeric seconds today.
                        pass
    m = _RETRY_AFTER_BODY_RE.search(str(exc))
    if m:
        return float(m.group(1)) * (0.001 if m.group(2).lower() == "ms" else 1.0)
    return None


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
