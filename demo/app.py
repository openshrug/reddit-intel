"""
FastAPI app for the public AI Pulse demo.

Routes:
  GET  /                    landing page with curated grid + form
  POST /analyze             kick off a background analysis job, returns job_id
  GET  /jobs/{job_id}       progress page with SSE stream
  GET  /jobs/{job_id}/stream  SSE event stream (used by the progress page)
  GET  /r/{subreddit}       insights dashboard (serves from cache if fresh)
  GET  /api/cached          JSON list of cached analyses (for the grid)

Jobs are tracked in an in-memory dict. On Fly/Railway single-instance
deploys this is fine; for multi-instance you'd swap to Redis.
"""

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from demo.analyzer import (
    analyze_subreddit,
    list_cached,
    _load_cache,
    load_cache_any,
    _normalize_name,
    mark_ideas_pending,
    generate_ideas_for_cached,
    get_ideas_log,
)


BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="AI Pulse Demo")


# ============================================================
# JOB QUEUE (in-memory)
# ============================================================

class Job:
    def __init__(self, job_id, subreddit):
        self.id = job_id
        self.subreddit = subreddit
        self.events = []  # list[dict]
        self.done = False
        self.error = None
        self.result = None
        self.started_at = time.time()
        self._cond = threading.Condition()

    def emit(self, event):
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def finish(self, result=None, error=None):
        with self._cond:
            self.result = result
            self.error = error
            self.done = True
            self._cond.notify_all()

    def wait_for_new(self, since_idx, timeout=1.0):
        """Block until a new event appears or timeout elapses."""
        with self._cond:
            if len(self.events) > since_idx or self.done:
                return
            self._cond.wait(timeout=timeout)


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

# Subreddits with an in-flight ideas thread. Prevents duplicate concurrent runs.
IDEAS_RUNNING: set[str] = set()
IDEAS_LOCK = threading.Lock()


def _cleanup_old_jobs(max_age_seconds=3600):
    """Drop jobs older than max_age to keep memory bounded."""
    now = time.time()
    with JOBS_LOCK:
        stale = [jid for jid, j in JOBS.items() if now - j.started_at > max_age_seconds]
        for jid in stale:
            JOBS.pop(jid, None)


def _run_job(job: Job):
    """Worker thread: runs analyze_subreddit and streams events into the job."""
    try:
        result = analyze_subreddit(job.subreddit, on_event=job.emit)
        job.finish(result=result)
    except Exception as e:
        job.emit({"stage": "error", "status": "error", "detail": str(e)})
        job.finish(error=str(e))
        return

    # Analyze succeeded — kick off idea generation in the background so ideas
    # start baking while the user is reading the painpoints on /r/{sub}.
    _start_ideas_background(job.subreddit)


def _start_ideas_background(subreddit, count=3):
    """Start ideas generation in a background thread if not already running.
    Safe to call multiple times — concurrent calls are deduped via IDEAS_RUNNING."""
    subreddit = _normalize_name(subreddit)

    with IDEAS_LOCK:
        if subreddit in IDEAS_RUNNING:
            return False
        IDEAS_RUNNING.add(subreddit)

    if not mark_ideas_pending(subreddit):
        with IDEAS_LOCK:
            IDEAS_RUNNING.discard(subreddit)
        return False

    def _worker():
        try:
            generate_ideas_for_cached(subreddit, count=count)
        finally:
            with IDEAS_LOCK:
                IDEAS_RUNNING.discard(subreddit)

    threading.Thread(target=_worker, daemon=True).start()
    return True


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    cached = list_cached()
    return TEMPLATES.TemplateResponse(
        request, "landing.html",
        {"cached": cached, "total_cached": len(cached)},
    )


@app.post("/analyze")
async def analyze_start(request: Request):
    """Start a new analysis job. Accepts form or JSON body."""
    try:
        form = await request.form()
        subreddit = form.get("subreddit", "")
    except Exception:
        body = await request.json()
        subreddit = body.get("subreddit", "")

    subreddit = _normalize_name(subreddit)
    if not subreddit:
        raise HTTPException(400, "subreddit required")

    # If fresh cache exists, skip the job and jump straight to results
    if _load_cache(subreddit):
        return RedirectResponse(f"/r/{subreddit}", status_code=303)

    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, subreddit)
    with JOBS_LOCK:
        JOBS[job_id] = job

    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()

    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return TEMPLATES.TemplateResponse(
        request, "progress.html",
        {"job_id": job_id, "subreddit": job.subreddit},
    )


@app.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    """SSE stream — yields each event as it arrives, closes on done."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    async def event_source():
        sent = 0
        while True:
            # Drain any new events without blocking the event loop too long
            while sent < len(job.events):
                ev = job.events[sent]
                sent += 1
                yield f"event: progress\ndata: {json.dumps(ev)}\n\n"

            if job.done:
                final = {
                    "ok": job.error is None,
                    "error": job.error,
                    "redirect": f"/r/{job.subreddit}" if job.error is None else None,
                }
                yield f"event: done\ndata: {json.dumps(final)}\n\n"
                return

            # Brief async sleep; thread-side notifies us via the events list
            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/r/{subreddit}", response_class=HTMLResponse)
async def insights_page(request: Request, subreddit: str):
    subreddit = _normalize_name(subreddit)
    data = load_cache_any(subreddit)
    if not data:
        return TEMPLATES.TemplateResponse(
            request, "not_found.html",
            {"subreddit": subreddit},
            status_code=404,
        )

    # Auto-kick ideas generation if it's never been started for this run
    if data.get("ideas_status") is None:
        _start_ideas_background(subreddit)
        data = load_cache_any(subreddit) or data

    return TEMPLATES.TemplateResponse(
        request, "insights.html",
        {"data": data, "subreddit": subreddit},
    )


@app.get("/api/r/{subreddit}/ideas")
async def api_ideas(subreddit: str):
    """Polling endpoint for the insights page. Returns the current ideas
    state plus status so the frontend can swap in results when ready."""
    subreddit = _normalize_name(subreddit)
    data = load_cache_any(subreddit)
    if not data:
        raise HTTPException(404, "subreddit not analyzed")

    return JSONResponse({
        "status": data.get("ideas_status") or "not_started",
        "ideas": data.get("ideas") or [],
        "error": data.get("ideas_error"),
        "generated_at": data.get("ideas_generated_at"),
        "log": get_ideas_log(subreddit),
    })


@app.post("/api/r/{subreddit}/ideas/start")
async def api_ideas_start(subreddit: str):
    """Manual kickoff of idea generation (used if auto-start was missed)."""
    subreddit = _normalize_name(subreddit)
    if not load_cache_any(subreddit):
        raise HTTPException(404, "subreddit not analyzed")
    started = _start_ideas_background(subreddit)
    return JSONResponse({"started": started})


@app.get("/api/cached")
async def api_cached():
    return JSONResponse(list_cached())
