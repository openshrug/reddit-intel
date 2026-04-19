"""Quantitative painpoint-merge / pending-dedup evaluation.

Scope
-----

Pair-wise cosine harness for the two thresholds that govern painpoint
clustering in the engine:

* ``MERGE_COSINE_THRESHOLD`` -- pending -> painpoint link at promote
  time (``db/painpoints.py: promote_pending``).
* ``PENDING_MERGE_THRESHOLD`` -- pending dedup at extract time
  (``db/painpoints.py: save_pending_painpoints_batch``).

Layout
------

* :mod:`evaluation.painpoints_eval.cosine_lab` -- ad-hoc REPL / one-off
  cosine on arbitrary strings (no DB lookup, caches embeddings per
  session).
* :mod:`evaluation.painpoints_eval.pair_eval` -- load gold-pair YAML,
  embed, score precision / recall / F1 at a named threshold, dump JSON.
* :mod:`evaluation.painpoints_eval.threshold_sweep` -- sweep
  ``MERGE_COSINE_THRESHOLD`` over a range, emit CSV + ASCII curve.
* :mod:`evaluation.painpoints_eval.mega_merge_stress` -- replay the
  ``pp #48`` mega-merge cluster from a real snapshot at varying
  thresholds, report cluster count + sizes.

Gold pairs live under ``fixtures/`` and are derived from
``evaluation/agentic_eval`` ``report.md`` files following the protocol
in ``SEEDING.md``.
"""
