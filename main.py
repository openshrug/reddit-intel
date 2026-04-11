#!/usr/bin/env python3
"""
Reddit Pulse — Reddit-focused trend intelligence CLI.

Two main commands:
  analyze    — scrape Reddit + Subriff, populate the facts DB, LLM curates
  idea       — read the DB, generate buildable product ideas
"""

import argparse
import os
from dotenv import load_dotenv

import database as db
from ingest import analyze
from ideas import propose_idea, print_ideas


def cmd_analyze(args):
    """Run the ingest pipeline."""
    if args.debug:
        os.environ["AIPULSE_DEBUG"] = "1"
    analyze(rounds=args.rounds)


def cmd_idea(args):
    """Generate product ideas from the DB."""
    if args.debug:
        os.environ["AIPULSE_DEBUG"] = "1"
    ideas = propose_idea(
        focus=args.focus,
        count=args.count,
        check_competition=not args.skip_competition,
    )
    if ideas:
        print_ideas(ideas)
    else:
        print("No ideas generated. Run `python main.py analyze` first to populate the DB.")


def cmd_demo(args):
    """Run the FastAPI demo app."""
    import uvicorn
    print(f"⚡ Reddit Pulse demo → http://{args.host}:{args.port}")
    uvicorn.run("demo.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_stats(_args):
    """Print DB stats."""
    stats = db.get_stats()
    print("\n📊 Database stats:\n")
    for k, v in stats.items():
        print(f"  {k:20} {v}")

    taxonomy = db.get_category_list_flat()
    print(f"\n🌲 Taxonomy ({len(taxonomy)} leaf categories):")
    for c in taxonomy[:20]:
        print(f"  {c['path']}")
    if len(taxonomy) > 20:
        print(f"  ... and {len(taxonomy) - 20} more")

    painpoints = db.get_top_painpoints(limit=10)
    if painpoints:
        print("\n🔥 Top painpoints:")
        for pp in painpoints:
            cats = pp.get("categories") or "uncategorized"
            print(f"  [{pp['signal_count']}x, sev {pp.get('severity', '?')}] {pp['title']} ({cats})")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Reddit Pulse — trend intelligence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    analyze_p = subparsers.add_parser("analyze", help="Scrape Reddit + Subriff, populate DB")
    analyze_p.add_argument("--rounds", type=int, default=1, help="Number of refinement rounds")
    analyze_p.add_argument("--debug", action="store_true", help="Show full LLM conversation")
    analyze_p.set_defaults(func=cmd_analyze)

    # idea
    idea_p = subparsers.add_parser("idea", help="Generate ideas from DB")
    idea_p.add_argument("--focus", type=str, default=None,
                        help="Category slug to focus on (e.g. 'ai-coding-tools')")
    idea_p.add_argument("--count", type=int, default=3, help="Number of ideas")
    idea_p.add_argument("--skip-competition", action="store_true",
                        help="Skip HN+GitHub competition check")
    idea_p.add_argument("--debug", action="store_true", help="Show full LLM conversation")
    idea_p.set_defaults(func=cmd_idea)

    # stats
    stats_p = subparsers.add_parser("stats", help="Show DB stats")
    stats_p.set_defaults(func=cmd_stats)

    # reset-cursors
    reset_p = subparsers.add_parser("reset-cursors",
                                    help="Reset Reddit pagination cursors (start over from page 1)")
    reset_p.set_defaults(func=lambda _a: (db.reset_cursor("reddit"),
                                           print("Reset all Reddit cursors")))

    # demo
    demo_p = subparsers.add_parser("demo", help="Run the public demo web app")
    demo_p.add_argument("--host", default="127.0.0.1")
    demo_p.add_argument("--port", type=int, default=8000)
    demo_p.add_argument("--reload", action="store_true", help="Auto-reload on code change")
    demo_p.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
