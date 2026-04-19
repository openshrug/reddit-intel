"""Pipeline quality evaluation harness.

Drives a clean-DB run on a fixed subreddit list, snapshots `trends.db`
after each stage, and emits per-snapshot markdown dumps + JSON metrics
so an evaluator agent can produce a grounded quality report.

Entry point: `python -m evaluation.agentic_eval.run_pipeline`.
"""
