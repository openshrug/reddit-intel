"""Hermetic tests for the OpenAI retry layer in `llm.py`.

Covers:
- `_parse_retry_after`: header / body-regex / missing-hint paths.
- `_classify_for_retry`: per-exception-class retry budgets.
- `_compute_retry_delay`: hint-driven sleep vs full-jitter exponential.
- `call_with_openai_retry`: success on first try, hint-respected retry,
  budget exhaustion, immediate raise on non-retryable, semaphore is
  released on every path.

No network calls; OpenAI exception classes are constructed against
fake `httpx.Response` objects. `time.sleep` is monkeypatched so the
retry loop runs in milliseconds even with a 60s wall-time cap.
"""

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

import llm


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def no_sleep(monkeypatch):
    """Replace `time.sleep` with a recorder so retry tests run instantly.

    Returns the list of recorded sleep durations so tests can assert on
    the actual delays the retry loop would have requested.
    """
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


@pytest.fixture
def deterministic_jitter(monkeypatch):
    """Pin `random.uniform(a, b)` to its midpoint so jitter math is
    predictable. Returns nothing — fixture exists for its side-effect."""
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: (a + b) / 2)


def _make_rate_limit_error(*, retry_after=None, retry_after_ms=None,
                           body_hint=None, message=None):
    """Build a real `RateLimitError` against a fake httpx response.

    `retry_after` / `retry_after_ms` populate response headers; `body_hint`
    embeds a "Please try again in X" string into the body that the SDK
    stringifies into the exception message (mimicking the live 429 we
    saw against gpt-5-nano)."""
    headers = {}
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    if retry_after_ms is not None:
        headers["retry-after-ms"] = str(retry_after_ms)

    body_msg = body_hint or "Rate limit reached."
    body = {"error": {"message": body_msg, "type": "tokens",
                      "code": "rate_limit_exceeded"}}

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(429, headers=headers, request=request, json=body)
    return RateLimitError(
        message=(message if message is not None else f"Error code: 429 - {body}"),
        response=response,
        body=body,
    )


def _make_status_error(status_code: int):
    """Build an APIStatusError for a non-429 HTTP response."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code, request=request, json={"error": "x"})
    if status_code >= 500:
        return InternalServerError(
            message=f"Error code: {status_code}", response=response,
            body={"error": "x"},
        )
    return APIStatusError(
        message=f"Error code: {status_code}", response=response,
        body={"error": "x"},
    )


# ---------------------------------------------------------------------------
# _parse_retry_after
# ---------------------------------------------------------------------------


class TestParseRetryAfter:
    def test_prefers_retry_after_ms_over_seconds(self):
        # When both are present we should use the millisecond value
        # because it carries sub-second precision that the integer
        # `retry-after` would round away.
        exc = _make_rate_limit_error(retry_after=2, retry_after_ms=1573)
        assert llm._parse_retry_after(exc) == pytest.approx(1.573)

    def test_uses_retry_after_seconds_when_no_ms(self):
        exc = _make_rate_limit_error(retry_after=4)
        assert llm._parse_retry_after(exc) == pytest.approx(4.0)

    def test_falls_back_to_body_hint_when_headers_missing(self):
        exc = _make_rate_limit_error(
            body_hint="Rate limit reached. Please try again in 850ms.",
        )
        assert llm._parse_retry_after(exc) == pytest.approx(0.85)

    def test_body_hint_seconds_form(self):
        exc = _make_rate_limit_error(
            body_hint="Rate limit reached. Please try again in 1.573s. Visit ...",
        )
        assert llm._parse_retry_after(exc) == pytest.approx(1.573)

    def test_returns_none_when_no_hint_anywhere(self):
        # Generic exception with no headers and no matching message.
        assert llm._parse_retry_after(RuntimeError("boom")) is None

    def test_returns_none_for_unparseable_header(self):
        # HTTP-date form (which OpenAI doesn't use today, but the spec
        # allows). We deliberately don't depend on `email.utils` so this
        # falls through to the body regex / exponential backoff.
        exc = _make_rate_limit_error(retry_after="Wed, 21 Oct 2026 07:28:00 GMT")
        assert llm._parse_retry_after(exc) is None


# ---------------------------------------------------------------------------
# _classify_for_retry
# ---------------------------------------------------------------------------


class TestClassifyForRetry:
    def test_rate_limit_gets_largest_budget(self):
        exc = _make_rate_limit_error(retry_after=1)
        assert llm._classify_for_retry(exc) == llm._RATE_LIMIT_MAX_ATTEMPTS

    def test_5xx_status_error_is_retryable(self):
        assert llm._classify_for_retry(_make_status_error(500)) == \
            llm._TRANSIENT_MAX_ATTEMPTS
        assert llm._classify_for_retry(_make_status_error(503)) == \
            llm._TRANSIENT_MAX_ATTEMPTS

    def test_4xx_caller_errors_are_not_retryable(self):
        # 401, 403, 404, 422 — retrying just wastes the budget; the
        # request needs to be fixed by the caller, not delayed.
        for code in (400, 401, 403, 404, 422):
            assert llm._classify_for_retry(_make_status_error(code)) == 0, \
                f"status {code} should not retry"

    def test_connection_error_is_retryable(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        exc = APIConnectionError(request=request)
        assert llm._classify_for_retry(exc) == llm._TRANSIENT_MAX_ATTEMPTS

    def test_timeout_error_is_retryable(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        exc = APITimeoutError(request=request)
        assert llm._classify_for_retry(exc) == llm._TRANSIENT_MAX_ATTEMPTS

    def test_unknown_exception_is_not_retryable(self):
        # Programmer errors (KeyError, ValueError, RuntimeError) shouldn't
        # spend the retry budget — they'll just fail again identically.
        assert llm._classify_for_retry(RuntimeError("bug")) == 0
        assert llm._classify_for_retry(ValueError("bad arg")) == 0

    def test_rate_limit_takes_precedence_over_status_error(self):
        # RateLimitError extends APIStatusError, so the order of the
        # isinstance checks matters. This guards against accidentally
        # giving 429s the smaller transient budget if the checks are
        # ever reordered.
        exc = _make_rate_limit_error()
        assert llm._classify_for_retry(exc) == llm._RATE_LIMIT_MAX_ATTEMPTS
        assert llm._RATE_LIMIT_MAX_ATTEMPTS > llm._TRANSIENT_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# _compute_retry_delay
# ---------------------------------------------------------------------------


class TestComputeRetryDelay:
    def test_uses_hint_plus_jitter(self, deterministic_jitter):
        # Hint = 1.573s, midpoint jitter = 0.55s (between 0.1 and 1.0).
        exc = _make_rate_limit_error(retry_after_ms=1573)
        delay = llm._compute_retry_delay(exc, attempt=0, backoff_base=4)
        assert delay == pytest.approx(1.573 + 0.55)

    def test_falls_back_to_exponential_with_full_jitter(self, deterministic_jitter):
        # No hint -> full-jitter exponential. With midpoint jitter at
        # attempt=0: random.uniform(0, 4 * 1) midpoint = 2.0
        exc = RuntimeError("no hint here")
        delay = llm._compute_retry_delay(exc, attempt=0, backoff_base=4)
        assert delay == pytest.approx(2.0)

    def test_exponential_grows_with_attempt(self, deterministic_jitter):
        # Midpoints: attempt=0 -> base/2, attempt=1 -> base, attempt=2 -> 2*base.
        exc = RuntimeError("no hint")
        d0 = llm._compute_retry_delay(exc, attempt=0, backoff_base=4)
        d1 = llm._compute_retry_delay(exc, attempt=1, backoff_base=4)
        d2 = llm._compute_retry_delay(exc, attempt=2, backoff_base=4)
        assert d0 < d1 < d2
        assert d2 == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# call_with_openai_retry
# ---------------------------------------------------------------------------


class TestCallWithOpenAIRetry:
    def test_success_on_first_try_no_sleep(self, no_sleep):
        result = llm.call_with_openai_retry(lambda: "ok")
        assert result == "ok"
        assert no_sleep == []

    def test_retries_rate_limit_then_succeeds(self, no_sleep, deterministic_jitter):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise _make_rate_limit_error(retry_after_ms=200)
            return "finally"

        result = llm.call_with_openai_retry(flaky)
        assert result == "finally"
        assert attempts["n"] == 3
        # 2 retries -> 2 sleeps, each ~= hint(0.2) + midpoint jitter(0.55)
        assert len(no_sleep) == 2
        for s in no_sleep:
            assert s == pytest.approx(0.75, abs=0.01)

    def test_raises_after_exhausting_rate_limit_budget(self, no_sleep,
                                                      deterministic_jitter):
        def always_429():
            raise _make_rate_limit_error(retry_after_ms=100)

        with pytest.raises(RateLimitError):
            llm.call_with_openai_retry(always_429)
        # _RATE_LIMIT_MAX_ATTEMPTS attempts -> _RATE_LIMIT_MAX_ATTEMPTS - 1 sleeps.
        assert len(no_sleep) == llm._RATE_LIMIT_MAX_ATTEMPTS - 1

    def test_does_not_retry_4xx(self, no_sleep):
        def bad_request():
            raise _make_status_error(400)

        with pytest.raises(APIStatusError):
            llm.call_with_openai_retry(bad_request)
        # Non-retryable -> first failure raises, no sleeps recorded.
        assert no_sleep == []

    def test_does_not_retry_unknown_exception(self, no_sleep):
        def buggy():
            raise ValueError("caller passed bad args")

        with pytest.raises(ValueError):
            llm.call_with_openai_retry(buggy)
        assert no_sleep == []

    def test_5xx_uses_smaller_budget(self, no_sleep, deterministic_jitter):
        def always_500():
            raise _make_status_error(500)

        with pytest.raises(APIStatusError):
            llm.call_with_openai_retry(always_500)
        assert len(no_sleep) == llm._TRANSIENT_MAX_ATTEMPTS - 1

    def test_total_wait_capped(self, monkeypatch):
        # Force every retry to ask for a huge sleep so the cumulative
        # cap kicks in before the attempt budget. Without the cap the
        # loop would happily wait minutes; with the cap a single oversized
        # delay is clamped to the remaining budget and the next iteration
        # raises because the budget is exhausted.
        monkeypatch.setattr(llm, "_compute_retry_delay",
                            lambda exc, attempt, backoff_base: 1000.0)
        sleeps: list[float] = []
        monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))

        def always_429():
            raise _make_rate_limit_error()

        with pytest.raises(RateLimitError):
            llm.call_with_openai_retry(always_429)
        # First retry's sleep is clamped to the full _MAX_TOTAL_WAIT_S
        # budget; the second attempt then sees remaining <= 0 and raises.
        assert sleeps == [pytest.approx(llm._MAX_TOTAL_WAIT_S)]

    def test_releases_semaphore_on_exception(self, no_sleep):
        # If the helper leaked semaphore slots on every failure, after
        # OPENAI_COMPLETION_CONCURRENCY non-retryable failures the next
        # call would block forever. Run that many failures and then a
        # success — if we get the success back, the semaphore was
        # released cleanly.
        for _ in range(llm.OPENAI_COMPLETION_CONCURRENCY + 1):
            with pytest.raises(ValueError):
                llm.call_with_openai_retry(lambda: (_ for _ in ()).throw(ValueError("x")))
        assert llm.call_with_openai_retry(lambda: "ok") == "ok"

    def test_label_appears_in_log(self, no_sleep, deterministic_jitter, caplog):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise _make_rate_limit_error(retry_after_ms=10)
            return "ok"

        caplog.set_level("WARNING", logger="llm")
        llm.call_with_openai_retry(flaky, label="custom_label")
        assert any("custom_label attempt" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Per-model semaphore routing (Phase 1 split)
# ---------------------------------------------------------------------------


class TestSemaphoreRouting:
    """Completion and embedding paths must acquire DIFFERENT semaphores
    so a saturated completion fan-out can't starve embedding callers
    (and vice-versa). These tests use a tiny `RecordingSemaphore`
    wrapper — sufficient because the helper only needs ``__enter__`` /
    ``__exit__``."""

    class RecordingSemaphore:
        """Drop-in semaphore stand-in that records each acquire/release.
        Bound is irrelevant for these tests — we only assert WHICH
        semaphore got entered, not the bound itself (that's covered by
        the contention tests in test_painpoint_pipeline.py)."""

        def __init__(self):
            self.entered = 0
            self.exited = 0

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            self.exited += 1
            return False

    def test_completion_and_embedding_semaphores_are_distinct(self):
        # Trivial structural check: if the two module globals were
        # accidentally aliased the entire Phase 1 split is a no-op.
        assert llm.OPENAI_COMPLETION_SEMAPHORE is not llm.OPENAI_EMBEDDING_SEMAPHORE

    def test_default_is_completion_semaphore(self, no_sleep):
        # Swap in a recording stand-in for the COMPLETION bucket only;
        # the embedding bucket is left alone so any accidental routing
        # to it would surface as a failed assertion below.
        rec = self.RecordingSemaphore()
        original = llm.OPENAI_COMPLETION_SEMAPHORE
        llm.OPENAI_COMPLETION_SEMAPHORE = rec
        try:
            result = llm.call_with_openai_retry(lambda: "ok")
        finally:
            llm.OPENAI_COMPLETION_SEMAPHORE = original
        assert result == "ok"
        assert rec.entered == 1 and rec.exited == 1

    def test_explicit_semaphore_param_overrides_default(self, no_sleep):
        rec = self.RecordingSemaphore()
        result = llm.call_with_openai_retry(lambda: "ok", semaphore=rec)
        assert result == "ok"
        assert rec.entered == 1 and rec.exited == 1

    def test_embedder_routes_through_embedding_semaphore(self, no_sleep):
        # Verify the actual embedder wiring, not just the helper plumbing
        # — catches future regressions where someone removes the
        # `semaphore=` kwarg from `_embed_with_retry` and silently falls
        # back to the completion bucket.
        from db.embeddings import OpenAIEmbedder

        rec = self.RecordingSemaphore()
        original = llm.OPENAI_EMBEDDING_SEMAPHORE
        llm.OPENAI_EMBEDDING_SEMAPHORE = rec
        # The embedder caches its own reference to the semaphore name at
        # call-time via the module-level import in db.embeddings, so we
        # also have to swap that binding for the duration of the test.
        from db import embeddings as embeddings_mod
        original_emb = embeddings_mod.OPENAI_EMBEDDING_SEMAPHORE
        embeddings_mod.OPENAI_EMBEDDING_SEMAPHORE = rec
        try:
            class FakeData:
                embedding = [0.0] * 1536
            class FakeResp:
                data = [FakeData()]
            class FakeClient:
                class embeddings:
                    @staticmethod
                    def create(model, input):
                        return FakeResp()
            emb = OpenAIEmbedder(client=FakeClient())
            emb.embed("hello")
        finally:
            llm.OPENAI_EMBEDDING_SEMAPHORE = original
            embeddings_mod.OPENAI_EMBEDDING_SEMAPHORE = original_emb
        assert rec.entered == 1 and rec.exited == 1


# ---------------------------------------------------------------------------
# Backwards-compat aliases for downstream consumers
# ---------------------------------------------------------------------------


class TestDeprecatedAliases:
    """`reddit-intel-closed` imports `OPENAI_API_SEMAPHORE` /
    `OPENAI_CONCURRENCY` to swap in a distributed implementation at
    process boot (PIPELINE.md §12). Until that consumer migrates to the
    per-model names, the old symbols must keep pointing at the
    completion bucket so downstream behaviour is unchanged."""

    def test_old_semaphore_aliases_completion(self):
        assert llm.OPENAI_API_SEMAPHORE is llm.OPENAI_COMPLETION_SEMAPHORE

    def test_old_concurrency_aliases_completion(self):
        assert llm.OPENAI_CONCURRENCY == llm.OPENAI_COMPLETION_CONCURRENCY
