# Improvements & deferred work

Living doc for design ideas that are **not** worth shipping today but
should be on the radar once the cheap interventions stop carrying us.
Each entry should answer:

- **Why deferred?** What's the simpler fix we tried first, and what
  signal would tell us that fix is no longer enough?
- **Trigger to revisit.** Concrete metric / failure mode that flips
  this from "nice-to-have" to "now."
- **Sketch.** Enough of a design that whoever picks it up doesn't have
  to re-derive the trade-offs.

---

## OpenAI rate limiting — Phase 2: hybrid RPM + TPM token-velocity tracker

### Status

**Deferred.** Phase 1 (per-model concurrency split — see
`OPENAI_COMPLETION_SEMAPHORE` / `OPENAI_EMBEDDING_SEMAPHORE` in
`llm.py`) plus the rate-limit-aware retry layer in
`call_with_openai_retry` should clear the 429 issue we observed in the
`ClaudeAI` quality-eval run (~7 / 66 batches dropped). If a re-run
still drops batches against `gpt-5-nano`, the next intervention is the
token-velocity tracker described below.

### Why deferred

The two interventions we already have address two of the three causes
of 429 loss:

1. **Concurrency contention across models** — fixed by Phase 1.
   Embeddings no longer queue behind 30 in-flight completions for 10s+
   waiting on a slot.
2. **Lack of retry on transient 429s** — fixed by the retry layer.
   `Retry-After` / body-hint-driven sleeps with jitter and per-error
   retry budgets ride out short TPM saturation windows.

The third cause — **issuing a request that would obviously blow the
per-minute TPM budget** — is the one Phase 1 doesn't help with. A
naive concurrency cap approximates TPM only loosely: 30 completions ×
~6.5K avg tokens × 60s/avg-latency-15s ≈ 780K tokens/minute, which is
3.9× over `gpt-5-nano`'s 200K TPM limit. We get away with it today
because actual concurrency rarely sustains 30 in-flight, and because
the retry layer absorbs the overflow. But there's a bias-variance
trade: the more reliably the upstream client respects the bucket, the
fewer 429s OpenAI ever has to send.

A token-velocity tracker eliminates this category of waste entirely
— we'd stop issuing requests *we already know will fail* — at the
cost of a meaningful amount of moving parts (per-model state, header
parsing, blocking semantics, observability).

### Trigger to revisit

Build it when **any one** of the following holds after a representative
quality-eval run with the Phase 1 patch in place:

- ≥ 1 batch fails permanently (post-retries) with 429.
- p99 of `_compute_retry_delay`-driven sleeps exceeds 10s (= we're
  riding out long TPM windows that a tracker would have prevented).
- Cumulative wall-time spent inside `call_with_openai_retry`-induced
  sleep exceeds ~5% of pipeline runtime.
- We add a model whose published RPM/TPM is wildly different from
  `gpt-5-nano` and we don't trust the semaphore size to be in the
  right ballpark for it.

The first three are observable from existing log warnings; the last
one is a code-review signal.

### Sketch

A header-driven, per-model token-velocity tracker. Three responsibilities:

1. **Cheap-side budgeting.** Maintain a per-model rolling window of
   `(timestamp, tokens_used)` covering the last 60s. Before issuing a
   request, estimate its token cost (input prompt length + reserved
   `max_output_tokens`) and check whether the window has room. If not,
   block until the oldest entry expires far enough to make room.
2. **Header-driven correction.** OpenAI returns `x-ratelimit-remaining-
   requests`, `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-
   requests`, `x-ratelimit-reset-tokens` on every response. After each
   call, snap the tracker's "effective remaining" to whichever is
   smaller — our local estimate or the server's reported remaining.
   This converges the tracker to the truth even when our token estimate
   is wrong (e.g. tool-use turns, structured output overhead).
3. **Per-model isolation.** One tracker per `(api_key, model)` pair so
   `gpt-5-nano`'s pressure can't block embedding traffic, and so a
   future addition of `gpt-4.1-mini` or `gpt-5` doesn't share buckets
   with `gpt-5-nano`.

#### Public surface

```python
# llm.py (additive — does not break existing callers)

class _ModelLimiter:
    """Per-model RPM + TPM tracker. Thread-safe."""
    def __init__(self, *, rpm: int, tpm: int): ...
    def acquire(self, *, est_tokens: int): ...     # blocks if needed
    def update_from_headers(self, headers: dict): ...

_LIMITERS: dict[tuple[str, str], _ModelLimiter] = {}

def call_with_openai_retry(do_call, *, label, model: str | None = None,
                           est_tokens: int | None = None,
                           backoff_base=_BACKOFF_BASE_DEFAULT,
                           semaphore=None):
    """If `model` is provided, also gate on the per-model tracker.
    `semaphore` becomes a backstop (mostly redundant when the tracker
    is enforcing both RPM and TPM accurately) but is kept for safety
    against unknown-model paths and tracker bugs."""
```

Embedder + `llm_call` start passing `model=` and a token estimate;
older callers that don't pass `model` keep the current
semaphore-only behaviour.

#### What this subsumes

- **Phase 1 semaphores** become safety nets, not the primary control.
  We could keep them at generous values (e.g. 100/200) since the tracker
  is doing the real budgeting.
- **The body-regex retry hint** stays, but should fire much less often
  because we're no longer over-issuing.

#### What this does NOT solve

- **Multi-process / multi-host coordination.** The tracker is
  in-process. `reddit-intel-closed` runs the engine in one process, so
  this is fine for the open repo. Distributed deployments would need
  to swap the trackers for a Redis-backed implementation (the same
  swap point already documented for the legacy semaphore in
  PIPELINE.md §12).
- **Token estimation accuracy.** We can't perfectly estimate output
  tokens before the call. We rely on (a) reserving `max_output_tokens`
  pessimistically, (b) header-driven correction after the fact. There
  will still be the occasional 429 when our estimate undershoots and
  several requests land in the same window — the existing retry layer
  handles those.

#### Estimated work

Roughly 1 day:

- ~150 LOC for `_ModelLimiter` + window math + header parsing
- ~30 LOC of wiring through `call_with_openai_retry` + the two callers
- ~150 LOC of tests (window expiry, header correction, blocking
  semantics, per-model isolation, race conditions on update vs acquire)
- Doc updates here + PIPELINE.md

#### Worth-doing-with-it

If we end up touching this layer, two adjacent improvements ride free:

- **Surface budget pressure as a metric.** Log
  `tokens_used_in_window / tpm_limit` per call so we can plot it
  against pipeline runtime. The `evaluation/agentic_eval/` snapshot machinery
  already collects programmatic metrics — adding a "rate limit
  pressure" series there would let us validate Phase 2 the same way
  we validated Phase 1.
- **Per-tier auto-sizing.** Read `OPENAI_TIER` env var (or first-call
  headers) and size `rpm` / `tpm` from a small lookup table instead
  of hard-coding. Avoids a class of "I bumped the tier but forgot to
  bump the constant" bugs.

---

<!-- Append future deferred-work entries above this line, newest first -->
