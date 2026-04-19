"""Shared math + IO helpers private to painpoints_eval/.

Kept small and unexported. If a sibling package (`category_eval/`,
…) ends up needing the same helper, promote it to
``evaluation/shared/`` per the rule in ``evaluation/README.md`` rather
than copy-pasting.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import yaml


def cosine_sim(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity between two embedding vectors.

    Mirrors ``db.category_clustering._cosine_sim`` (kept local instead
    of imported because that one is module-private and the math is
    five lines).
    """
    a = list(a)
    b = list(b)
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# YAML fixture IO (shared by pair_eval + threshold_sweep)
# ---------------------------------------------------------------------------

VALID_LABELS = ("positive", "negative")


def load_pairs(path: Path) -> dict:
    """Load + validate a fixture YAML produced per ``SEEDING.md``.

    Returns a dict with two keys:

    * ``threshold_under_test`` -- str, the constant *name* (looked up
      live by the caller).
    * ``pairs`` -- list of dicts with keys ``id``, ``a``, ``b``,
      ``label``, ``cite``, optional ``notes``.

    Raises :class:`ValueError` if the file is malformed (duplicate ids,
    bad label values, missing required fields). Validation is strict
    so a typo doesn't silently zero-out half the fixture.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "pairs" not in raw:
        raise ValueError(f"{path}: top-level YAML must be a mapping with a 'pairs' key")

    threshold_name = raw.get("threshold_under_test")
    if not isinstance(threshold_name, str) or not threshold_name:
        raise ValueError(f"{path}: 'threshold_under_test' must be a non-empty string")

    pairs = raw["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path}: 'pairs' must be a non-empty list")

    seen_ids: set[str] = set()
    cleaned: list[dict] = []
    for i, p in enumerate(pairs):
        if not isinstance(p, dict):
            raise ValueError(f"{path}: pair index {i} is not a mapping")
        for required in ("id", "a", "b", "label", "cite"):
            if required not in p:
                raise ValueError(f"{path}: pair index {i} missing required field '{required}'")
        pid = p["id"]
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"{path}: pair index {i} has empty id")
        if pid in seen_ids:
            raise ValueError(f"{path}: duplicate pair id {pid!r}")
        seen_ids.add(pid)
        if p["label"] not in VALID_LABELS:
            raise ValueError(
                f"{path}: pair {pid!r} label {p['label']!r} not in {VALID_LABELS}"
            )
        if not isinstance(p["a"], str) or not p["a"].strip():
            raise ValueError(f"{path}: pair {pid!r} field 'a' must be a non-empty string")
        if not isinstance(p["b"], str) or not p["b"].strip():
            raise ValueError(f"{path}: pair {pid!r} field 'b' must be a non-empty string")
        cleaned.append({
            "id": pid,
            "a": p["a"],
            "b": p["b"],
            "label": p["label"],
            "cite": p["cite"],
            "notes": p.get("notes", ""),
        })

    return {"threshold_under_test": threshold_name, "pairs": cleaned}


def resolve_threshold(name: str) -> float:
    """Look up a named threshold constant on ``db.embeddings``.

    Lets the YAML fixture say ``threshold_under_test:
    MERGE_COSINE_THRESHOLD`` and have ``pair_eval`` quote *whatever
    value the engine is using right now* — so retunes don't require
    re-editing every fixture file.
    """
    import db.embeddings as emb
    if not hasattr(emb, name):
        raise ValueError(
            f"db.embeddings has no constant named {name!r}; "
            "fixture's threshold_under_test does not match a live tunable."
        )
    val = getattr(emb, name)
    if not isinstance(val, (int, float)):
        raise ValueError(f"db.embeddings.{name} = {val!r} is not numeric")
    return float(val)
