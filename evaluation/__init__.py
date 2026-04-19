"""Quality / retrieval evaluation harnesses for reddit-intel.

This is a meta-package — see ``evaluation/README.md`` for the canonical
map of sub-packages, the qualitative -> quantitative loop, and the
``shared/`` promotion rule for utilities used across siblings.

Current contents:

* :mod:`evaluation.agentic_eval` -- snapshot-driven qualitative judge
  (drives a clean-DB pipeline run, snapshots ``trends.db`` per stage,
  hands off to an LLM evaluator that produces ``report.md``).
* :mod:`evaluation.painpoints_eval` -- quantitative pair-cosine harness
  for ``MERGE_COSINE_THRESHOLD`` / ``PENDING_MERGE_THRESHOLD`` tuning;
  consumes the gold pairs lifted from agentic_eval reports.

A future ``evaluation.category_eval`` package is reserved for category-
sweep judge calibration / sibling distinctness / runtime-cat lifecycle.
A ``evaluation.shared`` package will be created lazily the first time
two sibling packages need the same utility.
"""
