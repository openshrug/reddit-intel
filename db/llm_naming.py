"""LLM-naming wrapper used by the category worker (§5 of the plan).

The worker calls these from inside the merge lock when it needs the LLM
to name a new category or a set of split sub-categories. They're written
so tests can inject a fake namer (`namer=FakeNamer(...)`) without ever
touching the OpenAI API.
"""

import json
from typing import Callable, List, Optional


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
        from llm import llm_call
        return llm_call(self._get_client(), system, user, max_tokens=400)

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

    def name_split_subcategories(
        self, parent_name: str, clusters: List[List[str]],
        existing_taxonomy: Optional[List[dict]] = None,
    ):
        """Given the painpoint titles in each sub-cluster, propose a name +
        description for each. Returns a list of dicts in the same order.

        The parent category is already known (the one being split), so the
        LLM doesn't need to pick a parent — just name the children.
        """
        system = (
            "You are a taxonomist splitting an over-broad category into sub-categories. "
            "Given the parent category and N sub-clusters of painpoint titles, propose "
            "a short name and one-sentence description for each sub-cluster. "
            "Reply with JSON: {\"subcategories\": [{\"name\": \"...\", \"description\": \"...\"}, ...]}."
        )
        user = json.dumps(
            {
                "parent": parent_name,
                "clusters": [c[:10] for c in clusters],
            }
        )
        raw = self._call(system, user)
        return json.loads(raw)["subcategories"]


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

    def name_split_subcategories(self, parent_name, clusters,
                                 existing_taxonomy=None):
        if self._split_fn is not None:
            return self._split_fn(parent_name, clusters)
        return [
            {"name": f"{parent_name}/Sub-{i + 1}", "description": f"sub-cluster #{i + 1}"}
            for i, _ in enumerate(clusters)
        ]

    def describe_merged_category(self, survivor_name, loser_name,
                                 sample_member_titles):
        return {"description": f"Merged: {survivor_name} + {loser_name}"}
