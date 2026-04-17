"""LLM-naming wrapper used by the category worker (§5 of the plan).

The worker calls these from inside the merge lock when it needs the LLM
to name a new category or a set of split sub-categories. They're written
so tests can inject a fake namer (`namer=FakeNamer(...)`) without ever
touching the OpenAI API.
"""

import json
from typing import Callable, List, Literal, Optional

from pydantic import BaseModel, Field

# --- Structured output schemas ---

class SplitSubcategory(BaseModel):
    name: str = Field(description="Short sub-category name (2-4 words)")
    description: str = Field(
        description="Keyword-rich one-sentence description (30-50 words) covering "
                    "the technologies, tools, and specific complaint keywords that "
                    "future painpoints matching this sub-category would use"
    )
    cluster_indices: List[int] = Field(
        description="Which cluster indices (from the input) belong to this sub-category. "
                    "You may group multiple clusters under one sub-category."
    )


class SplitDecision(BaseModel):
    decision: Literal["split", "keep"]
    reason: str = Field(description="One-sentence explanation of why split or keep")
    subcategories: List[SplitSubcategory] = Field(
        default_factory=list,
        description="Proposed sub-categories — only populated when decision is 'split'. "
                    "Must cover all meaningful clusters; groupings should produce "
                    "semantically distinct, keyword-rich sub-categories."
    )


class UncatDecision(BaseModel):
    """LLM review of a single Uncategorized painpoint. Either mints a
    new category anchored at some existing parent, or leaves the
    painpoint where it is. Default should lean toward `keep`."""
    action: Literal["create", "keep"]
    reason: str = Field(description="One-sentence rationale")
    name: Optional[str] = Field(
        default=None,
        description="New category name (2-4 words). Required when action=create.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Keyword-rich description for the new category (30-50 words). "
                    "Required when action=create.",
    )
    parent: Optional[str] = Field(
        default=None,
        description="Exact name of the existing category to nest this under. "
                    "Required when action=create. Use a ROOT name if no existing "
                    "branch fits well.",
    )


class LLMNamer:
    """Default namer that calls the project's `llm.py` helpers.

    Tests should pass a fake instead of constructing this — calling the
    real LLM in unit tests would be slow, flaky, and expensive.
    """

    def __init__(self, client=None):
        self._client = client   # lazy: only fetched if actually called

    def _get_client(self):
        if self._client is None:
            from llm import get_client  # local import to avoid hard dep at module load
            self._client = get_client()
        return self._client

    def _call(self, system, user):
        """Plain JSON-mode call for name/description generation. Uses
        gpt-4.1-mini — same reason as _call_structured (non-reasoning,
        faster, reliable for naming tasks).

        NOTE: OpenAI's json_object mode requires the word "json" to
        appear in the input messages, otherwise a 400 is returned. We
        append a canonical reminder to the user payload to guarantee
        compliance regardless of what the caller serialized.
        """
        from llm import llm_call
        user_with_json = f"{user}\n\nReturn your answer as a single JSON object."
        return llm_call(self._get_client(), system, user_with_json, max_tokens=400,
                        model=self._STRUCTURED_MODEL)

    # Sweep-time LLM calls (naming categories, deciding splits) are
    # classification/naming tasks that don't need deep reasoning. Using a
    # non-reasoning model avoids the "structured output returned None"
    # issue (reasoning models burn the token budget on reasoning and
    # never emit the schema) and is ~3-5× faster than gpt-5-nano.
    _STRUCTURED_MODEL = "gpt-4.1-mini"

    def _call_structured(self, system, user, response_model, max_tokens=2000):
        """Call the LLM with a Pydantic schema for structured output. Returns
        a validated model instance instead of a JSON string.

        Uses gpt-4.1-mini (non-reasoning) — faster and more reliable for
        structured output than gpt-5-nano.
        """
        from llm import llm_call
        return llm_call(
            self._get_client(), system, user,
            max_tokens=max_tokens,
            response_model=response_model,
            model=self._STRUCTURED_MODEL,
        )

    def name_new_category(self, sample_titles: List[str], sample_descriptions: List[str],
                          existing_taxonomy: Optional[List[dict]] = None):
        """Propose a name, description, AND parent for a brand-new category.

        `existing_taxonomy` is the output of `db.categories.get_category_list_flat()`:
        a list of `{"path": "Parent > Child", "name": str, "description": str}` dicts
        representing the current taxonomy tree. The LLM picks a parent from this list.

        Returns a dict: {"name": str, "description": str, "parent": str or null}.
        `parent` is the name of the existing ROOT category this should live under,
        or null if the LLM thinks it deserves a new root.
        """
        taxonomy_str = ""
        if existing_taxonomy:
            paths = [e["path"] for e in existing_taxonomy[:60]]
            taxonomy_str = (
                "\n\nExisting taxonomy tree (each line is 'Parent > Child'):\n"
                + "\n".join(f"  - {p}" for p in paths)
                + "\n\nPick an existing category name from this tree as the "
                "parent for your new category. You can pick either a root "
                "(e.g., 'Cloud & Infrastructure') or a child (e.g., 'Databases'). "
                "The new category will become a child of whatever you pick. "
                "Only return null for parent if NOTHING in the tree fits."
            )

        system = (
            "You are a taxonomist organising user pain points scraped from Reddit. "
            "Given a cluster of related painpoint titles and descriptions, "
            "propose a short category name (2-4 words), a one-sentence description, "
            "and which existing category from the taxonomy it should be placed under. "
            "Reply with JSON: {\"name\": \"...\", \"description\": \"...\", \"parent\": \"ExactCategoryName\" or null}. "
            "The parent value must be the EXACT name of an existing category from the tree below, or null."
            + taxonomy_str
        )
        user = json.dumps(
            {
                "titles": sample_titles[:10],
                "descriptions": [d[:200] for d in sample_descriptions[:10]],
            }
        )
        raw = self._call(system, user)
        return json.loads(raw)

    def decide_split(self, category_name: str, category_description: str,
                     total_members: int, clusters: List[dict]) -> SplitDecision:
        """Ask the LLM whether a bloated category should be split.

        Uses OpenAI structured output — returns a validated `SplitDecision`
        Pydantic model instance (no JSON parsing).

        Args:
            category_name: e.g. "AI Safety"
            category_description: the current description text
            total_members: how many painpoints are in the category
            clusters: list of dicts with keys:
                - 'size': int
                - 'sample_titles': list of 2-3 strings

        Returns a `SplitDecision` with `.decision`, `.reason`, `.subcategories`.
        When `.decision == "split"`, `.subcategories[i].cluster_indices` references
        positions in the input `clusters` list. The LLM is allowed to group
        multiple clusters under one sub-category.
        """
        system = (
            "You are a taxonomist reviewing a painpoint category using common sense. "
            "You receive the category name, its description, total member count, and "
            "top clusters found by embedding similarity (each with size + sample titles). "
            "Use SEMANTIC JUDGMENT, not cluster-size balance. Decide split vs keep.\n\n"
            "SPLIT when ANY of these is true:\n"
            "  - The sample titles span multiple distinct topics that belong in different buckets\n"
            "  - Significant chunks of content DON'T MATCH the category name/description "
            "(the category has been 'hijacked' by mis-routed painpoints — split them out)\n"
            "  - There are coherent sub-topics that would make sense as their own categories, "
            "even if some are small\n\n"
            "KEEP when all of:\n"
            "  - Sample titles are all variations of one coherent topic\n"
            "  - The content genuinely matches the category description\n\n"
            "Cluster sizes DO NOT matter — a small but semantically distinct cluster is still "
            "a valid sub-category. Trust the titles, not the sizes.\n\n"
            "If you split, group clusters by topic (multiple cluster indices can go under one "
            "sub-category). Sub-category descriptions MUST be keyword-rich (30-50 words) "
            "covering specific technologies, tools, and complaint keywords — this is critical "
            "for embedding-based routing of future painpoints.\n\n"
            "Return your decision as a JSON object matching the schema."
        )
        user = json.dumps({
            "category_name": category_name,
            "category_description": category_description,
            "total_members": total_members,
            "clusters": [
                {"index": i, "size": c["size"], "sample_titles": c["sample_titles"][:3]}
                for i, c in enumerate(clusters)
            ],
        })
        return self._call_structured(system, user, SplitDecision, max_tokens=1500)

    def decide_uncategorized(self, title: str, description: str,
                              signal_count: int, severity: int,
                              existing_taxonomy: List[dict]) -> UncatDecision:
        """Review a single Uncategorized painpoint. Decide whether it
        warrants its own category in the tree (and if so, pick a parent
        + propose name/description), or whether it should stay put.

        Conservative by design — default is `keep`.
        """
        taxonomy_str = (
            "\n".join(f"  - {e['path']}" for e in existing_taxonomy[:80])
            if existing_taxonomy else "  (empty)"
        )
        system = (
            "You are a taxonomist reviewing ONE painpoint that landed in the "
            "Uncategorized bucket. Decide on the merits: does it represent a "
            "topic worth naming, or is it a one-off that should stay put.\n\n"
            "CREATE a new category when:\n"
            "  - The painpoint names a concrete product, workflow, or "
            "lifestyle concern that an app/service could address (developer "
            "tooling, dating, fitness, habits, consumer apps — all fair game).\n"
            "  - It's plausible that similar painpoints will show up (a "
            "named bucket would collect them, not sit idle).\n"
            "  - No existing branch is a reasonable home — if a branch "
            "already fits, reroute will move it there, so answer `keep`.\n\n"
            "KEEP in Uncategorized when:\n"
            "  - Pure philosophy, social commentary, or opinion with no "
            "product/service hook.\n"
            "  - Genuinely one-off and unlikely to recur.\n"
            "  - An existing branch in the taxonomy covers it (reroute "
            "will handle the placement).\n\n"
            "Judge each case on its merits — no default bias toward either "
            "action. The seed tree is tech-heavy but consumer / lifestyle / "
            "relationships are valid new roots when the topic warrants one.\n\n"
            "When creating, `parent` must EXACTLY match an existing taxonomy "
            "entry (use the last segment of the path, e.g. 'Databases' from "
            "'Cloud & Infrastructure > Databases', or a root name). Use null "
            "only if NO existing root fits — spawning a new root. The new "
            "category description must be keyword-rich (30-50 words) to help "
            "future embedding-based routing.\n\n"
            "EXISTING TAXONOMY TREE:\n" + taxonomy_str
        )
        user = json.dumps({
            "title": title,
            "description": description or "",
            "signal_count": signal_count,
            "severity": severity,
        })
        return self._call_structured(system, user, UncatDecision, max_tokens=600)

    def describe_merged_category(self, survivor_name: str, loser_name: str,
                                  sample_member_titles: List[str]):
        """After merging two categories, generate an updated description
        for the survivor that covers the combined scope. The description
        should be keyword-rich so embedding similarity works well.

        Returns a dict: {"description": str}.
        """
        system = (
            "You are a taxonomist. Two categories have been merged. Given the surviving "
            "category name, the absorbed category name, and sample painpoint titles from "
            "both, write a keyword-rich description (one paragraph, ~30-50 words) for the "
            "merged category. Include specific technologies, tools, and common complaint "
            "keywords so embedding similarity can match future painpoints to this category. "
            "Reply with JSON: {\"description\": \"...\"}."
        )
        user = json.dumps({
            "survivor": survivor_name,
            "absorbed": loser_name,
            "sample_titles": sample_member_titles[:10],
        })
        raw = self._call(system, user)
        return json.loads(raw)


class FakeNamer:
    """Test double that returns canned names without calling any API.

    Pass `name_fn` / `split_fn` callables for fine-grained control, or rely
    on the deterministic defaults (cluster-index based names).
    """

    def __init__(self, name_fn: Optional[Callable] = None, split_fn: Optional[Callable] = None):
        self._name_fn = name_fn
        self._split_fn = split_fn
        self._next_id = 0

    def _next(self):
        self._next_id += 1
        return self._next_id

    def name_new_category(self, sample_titles, sample_descriptions,
                          existing_taxonomy=None):
        if self._name_fn is not None:
            return self._name_fn(sample_titles, sample_descriptions)
        i = self._next()
        return {"name": f"AutoCat-{i}", "description": f"auto-named cluster #{i}",
                "parent": None}

    def describe_merged_category(self, survivor_name, loser_name,
                                 sample_member_titles):
        return {"description": f"Merged: {survivor_name} + {loser_name}"}

    def decide_uncategorized(self, title, description, signal_count, severity,
                              existing_taxonomy):
        """Test double: default to `keep` so hermetic tests don't spuriously
        spawn new categories. Override via name_fn/split_fn isn't wired here
        — callers that need create-paths should inject their own namer."""
        return UncatDecision(action="keep", reason="fake: default keep")

    def decide_split(self, category_name, category_description, total_members, clusters):
        """Test double: always returns 'keep' unless clusters has ≥2 clusters
        of size ≥5 (mimicking the old threshold behaviour so existing tests
        still pass). Returns a SplitDecision Pydantic model."""
        eligible = [i for i, c in enumerate(clusters) if c["size"] >= 5]
        if len(eligible) < 2:
            return SplitDecision(decision="keep", reason="fake: too few big clusters",
                                 subcategories=[])
        subcats = [
            SplitSubcategory(
                name=f"{category_name}/Sub-{k + 1}",
                description=f"fake sub-cluster {k + 1}",
                cluster_indices=[ci],
            )
            for k, ci in enumerate(eligible[:4])
        ]
        return SplitDecision(decision="split", reason="fake: multiple big clusters",
                             subcategories=subcats)
