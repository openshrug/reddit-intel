"""Scrape a subreddit and dump to scrape_dumps/ for manual inspection."""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from reddit_scraper import scrape_subreddit_full  # noqa: E402

DUMP_DIR = Path(__file__).parent / "scrape_dumps"


def _ts(utc: float | None) -> str:
    if not utc:
        return "?"
    return datetime.fromtimestamp(utc, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def dump_readable(posts: list[dict], subreddit: str, elapsed: float):
    DUMP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = DUMP_DIR / f"{subreddit}_{stamp}"

    # --- JSON (machine-readable, full fidelity) ---
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(posts, indent=2, ensure_ascii=False, default=str))

    # --- Markdown (human-readable) ---
    md_path = base.with_suffix(".md")
    lines = [
        f"# r/{subreddit} — scrape dump",
        "",
        f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Posts:** {len(posts)}",
        f"- **Posts with comments:** {sum(1 for p in posts if p.get('comments'))}",
        f"- **Total comments:** {sum(len(p.get('comments', [])) for p in posts)}",
        f"- **Scrape time:** {elapsed:.1f}s",
        "",
        "---",
        "",
    ]

    for i, p in enumerate(posts, 1):
        comments = p.get("comments", [])
        lines.append(f"## {i}. {p['title']}")
        lines.append("")
        lines.append(
            "| Field | Value |"
        )
        lines.append("|---|---|")
        lines.append(f"| Reddit ID | `{p.get('name', '')}` |")
        lines.append(f"| Author | u/{p.get('author', '?')} |")
        lines.append(f"| Score | {p.get('score', 0)} |")
        lines.append(f"| Comments (on reddit) | {p.get('num_comments', 0)} |")
        lines.append(f"| Comments (fetched) | {len(comments)} |")
        lines.append(f"| Upvote ratio | {p.get('upvote_ratio', '?')} |")
        lines.append(f"| Created | {_ts(p.get('created_utc'))} UTC |")
        lines.append(f"| Flair | {p.get('link_flair_text', '') or '—'} |")
        lines.append(f"| Permalink | {p.get('permalink', '')} |")
        lines.append("")

        body = p.get("selftext", "").strip()
        if body:
            lines.append(f"**Post body** ({len(body)} chars):")
            lines.append("")
            lines.append(f"> {body[:3000]}")
            if len(body) > 3000:
                lines.append(f"> ... (truncated, {len(body)} total chars)")
            lines.append("")

        if comments:
            lines.append(f"### Comments ({len(comments)})")
            lines.append("")
            for j, c in enumerate(comments, 1):
                cbody = c.get("body", "").strip()
                lines.append(
                    f"**{j}.** u/{c.get('author', '?')} · "
                    f"score {c.get('score', 0)} · "
                    f"depth {c.get('depth', 0)} · "
                    f"{_ts(c.get('created_utc'))} UTC"
                )
                lines.append("")
                lines.append(f"> {cbody[:1500]}")
                if len(cbody) > 1500:
                    lines.append(f"> ... (truncated, {len(cbody)} total chars)")
                lines.append("")

        lines.append("---")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("\nDumped to:")
    print(f"  Markdown: {md_path}")
    print(f"  JSON:     {json_path}")


async def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "AppBusiness"
    t0 = time.perf_counter()
    posts = await scrape_subreddit_full(sub)
    elapsed = time.perf_counter() - t0
    print(f"\nScrape completed in {elapsed:.1f}s — {len(posts)} posts")
    dump_readable(posts, sub, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
