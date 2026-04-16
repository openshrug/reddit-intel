"""
Benchmark extraction across multiple configs and compare eval results.

    python -m painpoint_extraction.bench
    python -m painpoint_extraction.bench --subreddit vibecoding --output-dir scrape_dumps
    python -m painpoint_extraction.bench --judge-sample 30
"""

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import time

import db
import painpoint_extraction.extractor as ext
from db.queries import run_sql
from painpoint_extraction.eval import run_eval

log = logging.getLogger(__name__)

# Per-million-token pricing (USD)
MODEL_PRICING = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}

DEFAULT_CONFIGS = [
    {"name": "budget1k_low",  "budget": 1_000, "reasoning": "low"},
    {"name": "budget2k_low",  "budget": 2_000, "reasoning": "low"},
]


def _compute_cost(token_usage, model="gpt-5-nano"):
    """Calculate USD cost from token usage dict and model pricing."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-5-nano"])
    input_cost = token_usage["input_tokens"] / 1_000_000 * pricing["input"]
    output_cost = token_usage["output_tokens"] / 1_000_000 * pricing["output"]
    return round(input_cost + output_cost, 6)


async def run_config(cfg, post_ids, *, judge_sample=20, output_dir=None):
    """Run extraction + eval for a single config. Returns a results dict."""
    name = cfg["name"]
    print(f"\n{'#' * 70}")
    print(f"# CONFIG: {name}  (budget={cfg['budget']}, reasoning={cfg['reasoning']})")
    print(f"{'#' * 70}\n")

    conn = db.get_db()
    conn.execute("DELETE FROM pending_painpoints")
    conn.commit()
    conn.close()

    orig_reasoning = ext.REASONING_EFFORT
    ext.REASONING_EFFORT = cfg["reasoning"]
    try:
        t0 = time.perf_counter()
        ids, token_usage = await ext.extract_painpoints(post_ids, batch_token_budget=cfg["budget"])
        extract_time = time.perf_counter() - t0
    finally:
        ext.REASONING_EFFORT = orig_reasoning

    cost = _compute_cost(token_usage, model=cfg.get("model", ext.MODEL))

    print(f"  Extraction: {len(ids)} painpoints in {extract_time:.1f}s")
    print(f"  Tokens:  input={token_usage['input_tokens']:,}  "
          f"output={token_usage['output_tokens']:,}  "
          f"(reasoning={token_usage['reasoning_tokens']:,}, "
          f"text={token_usage['text_tokens']:,})")
    print(f"  Cost:    ${cost:.4f}")

    buf = io.StringIO()
    t1 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        metrics = await run_eval(sample_size=judge_sample)
    eval_time = time.perf_counter() - t1
    report_text = buf.getvalue()

    print(report_text)

    dump = {
        "config": cfg,
        "extraction": {
            "posts": len(post_ids),
            "painpoints_extracted": len(ids),
            "time_seconds": round(extract_time, 1),
            "tokens": token_usage,
            "cost_usd": cost,
        },
        "eval_metrics": {k: v for k, v in metrics.items() if k != "judge"},
        "eval_time_seconds": round(eval_time, 1),
    }
    if metrics.get("judge"):
        judge = metrics["judge"]
        dump["judge_summary"] = {
            "posts_judged": judge["posts_judged"],
            "painpoints_judged": judge["painpoints_judged"],
            "verdicts": [v.model_dump() for v in judge["verdicts"]],
            "missed": [m.model_dump() for m in judge["missed"]],
        }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"eval_{name}.json")
        with open(json_path, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"  -> JSON: {json_path}")

        txt_path = os.path.join(output_dir, f"eval_{name}_report.txt")
        with open(txt_path, "w") as f:
            f.write(f"CONFIG: {name}  (budget={cfg['budget']}, reasoning={cfg['reasoning']})\n")
            f.write(f"Extraction: {len(ids)} painpoints in {extract_time:.1f}s\n")
            f.write(f"Tokens:  input={token_usage['input_tokens']:,}  "
                    f"output={token_usage['output_tokens']:,}  "
                    f"(reasoning={token_usage['reasoning_tokens']:,}, "
                    f"text={token_usage['text_tokens']:,})\n")
            f.write(f"Cost:    ${cost:.4f}\n\n")
            f.write(report_text)
        print(f"  -> Report: {txt_path}")

    return dump


def _print_comparison(results):
    print(f"\n{'=' * 90}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 90}")
    header = (
        f"{'Config':<20} {'PPs':>4} {'Time':>7} "
        f"{'InTok':>8} {'OutTok':>8} {'Cost':>8} "
        f"{'QuoteOK':>8} {'AttrOK':>8} {'Categ':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        c = r["config"]["name"]
        e = r["extraction"]
        m = r["eval_metrics"]
        t = m["total"]
        tok = e.get("tokens", {})
        cost = e.get("cost_usd", 0)
        print(
            f"{c:<20} {e['painpoints_extracted']:>4} "
            f"{e['time_seconds']:>6.1f}s "
            f"{tok.get('input_tokens', 0):>8,} "
            f"{tok.get('output_tokens', 0):>8,} "
            f"${cost:>6.4f} "
            f"{m['quote_verbatim']:>3}/{t} "
            f"{m['attribution_correct']:>3}/{t} "
            f"{m['categorized']:>3}/{t}"
        )


async def bench(*, subreddit=None, configs=None, judge_sample=20, output_dir=None):
    """Run benchmark across all configs.

    Args:
        subreddit: Filter posts to this subreddit (None = all posts).
        configs: List of config dicts. Uses DEFAULT_CONFIGS if None.
        judge_sample: Number of painpoints to sample for LLM judge per config.
        output_dir: Directory to dump JSON + text reports (None = skip files).

    Returns:
        List of result dicts.
    """
    db.init_db()
    configs = configs or DEFAULT_CONFIGS

    query = "SELECT id FROM posts"
    params = None
    if subreddit:
        query += " WHERE subreddit = ?"
        params = (subreddit,)
    rows = run_sql(query, params)
    post_ids = [r["id"] for r in rows]

    if not post_ids:
        print(f"No posts found{f' for r/{subreddit}' if subreddit else ''}.")
        return []

    log.info("bench: %d posts, %d configs", len(post_ids), len(configs))

    results = []
    for cfg in configs:
        r = await run_config(
            cfg, post_ids,
            judge_sample=judge_sample,
            output_dir=output_dir,
        )
        results.append(r)

    _print_comparison(results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark extraction configs and compare eval results",
    )
    parser.add_argument(
        "--subreddit", type=str, default=None,
        help="Filter to posts from this subreddit (default: all)",
    )
    parser.add_argument(
        "--judge-sample", type=int, default=20,
        help="Painpoints to sample for LLM judge per config (default: 20)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="scrape_dumps",
        help="Directory for report dumps (default: scrape_dumps)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from dotenv import load_dotenv
    load_dotenv(override=True)

    asyncio.run(bench(
        subreddit=args.subreddit,
        judge_sample=args.judge_sample,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
