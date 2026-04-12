#!/usr/bin/env python3
"""
Reddit Intel — run the subreddit analysis pipeline.

Usage:
    python main.py ExperiencedDevs
    python main.py ExperiencedDevs --min-score 50
"""

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from subreddit_pipeline import analyze


def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Reddit Intel — subreddit analysis pipeline")
    parser.add_argument("subreddit", help="Subreddit to analyze (without r/)")
    parser.add_argument("--min-score", type=int, default=None,
                        help="Only fetch comments for posts with score >= this value")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = asyncio.run(analyze(args.subreddit, min_score=args.min_score))

    print(f"\n--- r/{summary['subreddit']} ---")
    print(f"  Posts scraped:     {summary['posts_scraped']}")
    print(f"  Posts persisted:   {summary['posts_persisted']}")
    print(f"  Comments stored:   {summary['comments_persisted']}")
    print(f"  Painpoints found:  {summary['painpoints_extracted']}")


if __name__ == "__main__":
    main()
