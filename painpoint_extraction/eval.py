"""
Extraction quality evaluation — programmatic checks + LLM judge.

    python -m extraction.eval
    python -m extraction.eval --sample-size 30 --model gpt-5.4-mini
"""

import argparse
import asyncio
import logging
import random
import statistics
from collections import defaultdict

from pydantic import BaseModel, Field

from db import get_db, init_db
from db.posts import get_posts_by_ids, get_comments_for_post
from painpoint_extraction.extractor import _format_post
from llm import get_client, llm_call

log = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_CONCURRENCY = 20
JUDGE_SAMPLE_SIZE = 50


# ============================================================
# Pydantic models
# ============================================================

class PainpointVerdict(BaseModel):
    painpoint_id: int = Field(description="The pending_painpoints.id being evaluated")
    validity: int = Field(
        ge=1, le=5,
        description="1 = hallucinated/not a painpoint, 5 = clearly a real user pain",
    )
    title_quality: int = Field(
        ge=1, le=5,
        description="1 = misleading/wrong, 5 = accurately captures the pain",
    )
    severity_accuracy: int = Field(
        ge=1, le=5,
        description="1 = severity way off, 5 = spot-on given source text",
    )
    notes: str = Field(
        default="",
        description="Brief explanation for low scores, or empty if all good",
    )


class MissedPainpoint(BaseModel):
    post_id: int
    description: str = Field(description="Brief description of the missed painpoint")


class JudgeResult(BaseModel):
    verdicts: list[PainpointVerdict]
    missed_painpoints: list[MissedPainpoint] = Field(
        default_factory=list,
        description="Painpoints the extractor missed in the source material",
    )


# ============================================================
# Data loading
# ============================================================

def _load_painpoints():
    """Load all pending_painpoints with their source material."""
    conn = get_db()
    rows = conn.execute("""
        SELECT pp.id, pp.post_id, pp.comment_id, pp.title, pp.description,
               pp.quoted_text, pp.severity, pp.category_id,
               p.title AS post_title, p.selftext, p.subreddit,
               p.score AS post_score, p.num_comments AS post_num_comments,
               c.body AS comment_body, c.score AS comment_score,
               cat.name AS category_name
        FROM pending_painpoints pp
        JOIN posts p ON p.id = pp.post_id
        LEFT JOIN comments c ON c.id = pp.comment_id
        LEFT JOIN categories cat ON cat.id = pp.category_id
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# Layer 1: Programmatic checks
# ============================================================

QUOTE_VERBATIM = "QUOTE_VERBATIM"
QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND"
QUOTE_WRONG_SOURCE = "QUOTE_WRONG_SOURCE"
ATTRIBUTION_OK = "ATTRIBUTION_OK"
ATTRIBUTION_WRONG = "ATTRIBUTION_WRONG"


def _check_quote(pp, all_comments_by_post):
    """Check if quoted_text is a verbatim substring of the attributed source."""
    quote = (pp.get("quoted_text") or "").lower().strip()
    if not quote:
        return QUOTE_NOT_FOUND, ATTRIBUTION_WRONG

    post_text = (
        (pp.get("post_title") or "") + " " + (pp.get("selftext") or "")
    ).lower()

    if pp["comment_id"] is not None:
        attributed_text = (pp.get("comment_body") or "").lower()
        if quote in attributed_text:
            return QUOTE_VERBATIM, ATTRIBUTION_OK
        if quote in post_text:
            return QUOTE_WRONG_SOURCE, ATTRIBUTION_WRONG
        for c in all_comments_by_post.get(pp["post_id"], []):
            if c["id"] != pp["comment_id"] and quote in (c.get("body") or "").lower():
                return QUOTE_WRONG_SOURCE, ATTRIBUTION_WRONG
        return QUOTE_NOT_FOUND, ATTRIBUTION_WRONG
    else:
        if quote in post_text:
            return QUOTE_VERBATIM, ATTRIBUTION_OK
        for c in all_comments_by_post.get(pp["post_id"], []):
            if quote in (c.get("body") or "").lower():
                return QUOTE_WRONG_SOURCE, ATTRIBUTION_WRONG
        return QUOTE_NOT_FOUND, ATTRIBUTION_WRONG


def _programmatic_checks(painpoints):
    """Run all programmatic checks. Returns a list of result dicts."""
    post_ids = list({pp["post_id"] for pp in painpoints})
    all_comments_by_post = {}
    for pid in post_ids:
        all_comments_by_post[pid] = get_comments_for_post(pid)

    results = []
    for pp in painpoints:
        quote_status, attr_status = _check_quote(pp, all_comments_by_post)
        categorized = pp.get("category_name") is not None
        results.append({
            "id": pp["id"],
            "post_id": pp["post_id"],
            "title": pp["title"],
            "severity": pp["severity"],
            "quote_status": quote_status,
            "attribution_status": attr_status,
            "categorized": categorized,
        })
    return results


# ============================================================
# Layer 2: LLM judge
# ============================================================

JUDGE_INSTRUCTIONS = """\
You are an extraction quality judge for developer-tool product research. \
You will receive a Reddit post with its comments, followed by a list of \
painpoints that were extracted from it by an automated system.

The extractor's goal is to find painpoints that someone could build a \
developer product around (an app, tool, extension, API, or service). \
Painpoints should be ACTIONABLE, SPECIFIC, and TECHNICAL — related to \
software development, tooling, infrastructure, or developer workflow.

The extractor should skip: pure opinions/memes/jokes, pricing complaints \
(unless they reveal a feature gap), social/career anxieties, and platform \
politics or company drama.

Your job is to evaluate each extracted painpoint AND identify any \
productizable painpoints the system missed.

For each extracted painpoint, score:
- validity (1-5): Is this a real, productizable painpoint? \
1 = hallucinated/noise/not actionable, 5 = clearly expressed pain that \
someone could build a product to solve.
- title_quality (1-5): Does the title accurately and concisely capture the pain? \
1 = misleading, 5 = excellent summary.
- severity_accuracy (1-5): Is the assigned severity reasonable? \
1 = way off, 5 = well-calibrated.
- notes: Brief explanation only if any score is <= 3.

Also list any productizable painpoints the extractor missed \
(missed_painpoints). Only list genuine, specific, actionable pains that \
someone could build a developer tool to solve — not vague sentiments or \
non-technical complaints."""


def _format_judge_input(post_dict, comments, extracted_pps):
    """Build the user input for the judge: source material + extracted painpoints."""
    source = _format_post(post_dict, comments)

    pp_lines = ["\n---\n\n## Extracted painpoints to evaluate:\n"]
    for pp in extracted_pps:
        pp_lines.append(
            f"[PP {pp['id']}] title=\"{pp['title']}\" | "
            f"severity={pp['severity']} | "
            f"quoted_text=\"{pp.get('quoted_text', '')}\" | "
            f"comment_id={pp.get('comment_id', 'null')} | "
            f"description=\"{pp.get('description', '')}\""
        )

    return source + "\n".join(pp_lines)


async def _llm_judge(painpoints, *, judge_model=JUDGE_MODEL, sample_size=20):
    """Run the LLM judge on a sample of painpoints, batched by post_id."""
    by_post = defaultdict(list)
    for pp in painpoints:
        by_post[pp["post_id"]].append(pp)

    post_ids = list(by_post.keys())
    random.shuffle(post_ids)

    sampled_pps = []
    sampled_post_ids = []
    for pid in post_ids:
        if len(sampled_pps) >= sample_size:
            break
        sampled_pps.extend(by_post[pid])
        sampled_post_ids.append(pid)

    posts_map = get_posts_by_ids(sampled_post_ids)

    groups = []
    for pid in sampled_post_ids:
        if pid not in posts_map:
            continue
        post = posts_map[pid]
        comments = get_comments_for_post(pid)
        groups.append((post, comments, by_post[pid]))

    client = get_client()
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def _judge_group(post, comments, pps):
        async with sem:
            input_text = _format_judge_input(post, comments, pps)
            try:
                result = await asyncio.to_thread(
                    llm_call, client, JUDGE_INSTRUCTIONS, input_text,
                    response_model=JudgeResult, model=judge_model,
                    max_tokens=None, reasoning_effort="low",
                )
            except Exception as exc:
                log.warning("judge: failed for post %d: %s", post["id"], exc)
                return None
            return result

    results = await asyncio.gather(
        *(_judge_group(p, c, pps) for p, c, pps in groups)
    )

    all_verdicts = []
    all_missed = []
    for r in results:
        if r is None:
            continue
        all_verdicts.extend(r.verdicts)
        all_missed.extend(r.missed_painpoints)

    return {
        "verdicts": all_verdicts,
        "missed": all_missed,
        "posts_judged": len([r for r in results if r is not None]),
        "painpoints_judged": len(sampled_pps),
    }


# ============================================================
# Report
# ============================================================

def _print_report(prog_results, judge_results):
    """Print a formatted evaluation report."""
    total = len(prog_results)
    print(f"\n{'=' * 60}")
    print(f"  Extraction Quality Report ({total} painpoints)")
    print(f"{'=' * 60}")

    # --- Programmatic ---
    quote_ok = sum(1 for r in prog_results if r["quote_status"] == QUOTE_VERBATIM)
    quote_wrong_src = sum(1 for r in prog_results if r["quote_status"] == QUOTE_WRONG_SOURCE)
    quote_missing = sum(1 for r in prog_results if r["quote_status"] == QUOTE_NOT_FOUND)

    attr_ok = sum(1 for r in prog_results if r["attribution_status"] == ATTRIBUTION_OK)
    categorized = sum(1 for r in prog_results if r["categorized"])

    severities = [r["severity"] for r in prog_results]
    sev_mean = statistics.mean(severities) if severities else 0
    sev_stdev = statistics.stdev(severities) if len(severities) > 1 else 0

    print(f"\n  Programmatic Checks")
    print(f"  {'-' * 40}")
    print(f"  Quote verbatim:     {quote_ok:>3}/{total} ({100*quote_ok/total:.0f}%)")
    print(f"  Quote wrong source: {quote_wrong_src:>3}/{total} ({100*quote_wrong_src/total:.0f}%)")
    print(f"  Quote not found:    {quote_missing:>3}/{total} ({100*quote_missing/total:.0f}%)")
    print(f"  Attribution correct:{attr_ok:>3}/{total} ({100*attr_ok/total:.0f}%)")
    print(f"  Categorized:        {categorized:>3}/{total} ({100*categorized/total:.0f}%)")
    sev_dist = {i: 0 for i in range(1, 11)}
    for s in severities:
        sev_dist[s] = sev_dist.get(s, 0) + 1
    print(f"  Severity mean/std:  {sev_mean:.1f} / {sev_stdev:.1f}")
    print(f"  Severity dist:      ", end="")
    print("  ".join(f"{i}:{sev_dist[i]}" for i in range(1, 11)))

    flagged = [
        r for r in prog_results
        if r["quote_status"] != QUOTE_VERBATIM
        or r["attribution_status"] != ATTRIBUTION_OK
    ]
    if flagged:
        print(f"\n  Flagged ({len(flagged)}):")
        for r in flagged[:20]:
            flags = []
            if r["quote_status"] != QUOTE_VERBATIM:
                flags.append(r["quote_status"])
            if r["attribution_status"] != ATTRIBUTION_OK:
                flags.append(r["attribution_status"])
            print(f"    #{r['id']:>3}: {', '.join(flags):30s} {r['title'][:50]}")
        if len(flagged) > 20:
            print(f"    ... and {len(flagged) - 20} more")

    # --- LLM Judge ---
    if judge_results:
        verdicts = judge_results["verdicts"]
        missed = judge_results["missed"]
        n_judged = judge_results["painpoints_judged"]
        n_posts = judge_results["posts_judged"]

        print(f"\n  LLM Judge ({len(verdicts)} scored, {n_posts} posts)")
        print(f"  {'-' * 40}")

        if verdicts:
            avg_val = statistics.mean(v.validity for v in verdicts)
            avg_title = statistics.mean(v.title_quality for v in verdicts)
            avg_sev = statistics.mean(v.severity_accuracy for v in verdicts)
            print(f"  Validity:           {avg_val:.1f}/5 avg")
            print(f"  Title quality:      {avg_title:.1f}/5 avg")
            print(f"  Severity accuracy:  {avg_sev:.1f}/5 avg")

            low_scores = [v for v in verdicts if min(v.validity, v.title_quality, v.severity_accuracy) <= 2]
            if low_scores:
                print(f"\n  Low-scoring painpoints ({len(low_scores)}):")
                for v in low_scores[:10]:
                    print(f"    PP #{v.painpoint_id}: validity={v.validity} "
                          f"title={v.title_quality} severity={v.severity_accuracy}"
                          f"{f'  ({v.notes})' if v.notes else ''}")

        if missed:
            print(f"\n  Missed painpoints ({len(missed)}):")
            for m in missed[:15]:
                print(f"    Post {m.post_id}: {m.description[:70]}")
            if len(missed) > 15:
                print(f"    ... and {len(missed) - 15} more")

    print(f"\n{'=' * 60}\n")

    return {
        "total": total,
        "quote_verbatim": quote_ok,
        "quote_wrong_source": quote_wrong_src,
        "quote_not_found": quote_missing,
        "attribution_correct": attr_ok,
        "categorized": categorized,
        "severity_mean": round(sev_mean, 2),
        "severity_stdev": round(sev_stdev, 2),
        "severity_dist": sev_dist,
        "judge": judge_results,
    }


# ============================================================
# Public API
# ============================================================

async def run_eval(*, sample_size=JUDGE_SAMPLE_SIZE, judge_model=JUDGE_MODEL):
    """Run full evaluation: programmatic checks + LLM judge.

    Returns a metrics dict suitable for regression assertions.
    """
    init_db()
    painpoints = _load_painpoints()
    if not painpoints:
        print("No pending painpoints to evaluate.")
        return {}

    log.info("eval: loaded %d painpoints", len(painpoints))

    prog_results = _programmatic_checks(painpoints)

    judge_results = None
    if sample_size > 0:
        log.info("eval: running LLM judge on sample of %d", sample_size)
        judge_results = await _llm_judge(
            painpoints, judge_model=judge_model, sample_size=sample_size,
        )

    return _print_report(prog_results, judge_results)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate extraction quality of pending painpoints",
    )
    parser.add_argument(
        "--sample-size", type=int, default=JUDGE_SAMPLE_SIZE,
        help="Number of painpoints to sample for LLM judge (0 to skip)",
    )
    parser.add_argument(
        "--model", type=str, default=JUDGE_MODEL,
        help=f"Judge model (default: {JUDGE_MODEL})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from dotenv import load_dotenv
    load_dotenv(override=True)

    asyncio.run(run_eval(sample_size=args.sample_size, judge_model=args.model))


if __name__ == "__main__":
    main()
