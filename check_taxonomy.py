"""Flag taxonomy.yaml pairs whose anchor texts embed too close.

A pair over DISTINGUISHABILITY_THRESHOLD means the two categories will
routinely steal painpoints from each other at routing time — the fix
is to rewrite one of the descriptions so the semantic overlap drops.

Usage:
  python check_taxonomy.py              # real embeddings (needs OPENAI_API_KEY)
  python check_taxonomy.py --fake       # FakeEmbedder, for hermetic CI

Exit code is the number of pairs over threshold (0 on pass), so the
script can drop in as a pre-commit check or CI gate.
"""

import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

# Pair-level cosine above this == indistinguishable at the anchor level.
# Picked to sit above typical seed-sibling cosine (~0.55) and below the
# MERGE_CATEGORY_THRESHOLD (0.80) — catches design rot before it shows
# up as mis-routing in a sweep.
DISTINGUISHABILITY_THRESHOLD = 0.70

TAXONOMY_FILE = Path(__file__).parent / "taxonomy.yaml"


def _cosine(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _collect_anchors():
    """Return [(path, anchor_text), ...] — one entry per root and child.

    `path` is "Root" or "Root > Child" so a violation report points at
    exactly which row in the YAML to rewrite.
    """
    taxonomy = yaml.safe_load(TAXONOMY_FILE.read_text())
    out = []
    for root_name, root_data in taxonomy.items():
        out.append((root_name, f"{root_name} {root_data.get('desc', '')}".strip()))
        for child_name, child_desc in root_data.get("children", {}).items():
            path = f"{root_name} > {child_name}"
            out.append((path, f"{child_name} {child_desc}".strip()))
    return out


def main(use_fake):
    anchors = _collect_anchors()

    if use_fake:
        from db.embeddings import FakeEmbedder
        embedder = FakeEmbedder()
    else:
        from db.embeddings import OpenAIEmbedder
        embedder = OpenAIEmbedder()

    embeddings = embedder.embed_batch([text for _, text in anchors])

    violations = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            sim = _cosine(embeddings[i], embeddings[j])
            if sim >= DISTINGUISHABILITY_THRESHOLD:
                violations.append((sim, anchors[i][0], anchors[j][0]))

    violations.sort(reverse=True)

    print(f"Checked {len(anchors)} anchors "
          f"({len(anchors) * (len(anchors) - 1) // 2} pairs) "
          f"at threshold {DISTINGUISHABILITY_THRESHOLD}")
    if not violations:
        print("PASS — all pairs distinguishable")
        return 0

    print(f"FAIL — {len(violations)} pair(s) over threshold:")
    for sim, a, b in violations:
        print(f"  {sim:.3f}   {a!r:50}  vs  {b!r}")
    return len(violations)


if __name__ == "__main__":
    fake = "--fake" in sys.argv[1:]
    sys.exit(main(fake))
