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
        client = self._get_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return llm_call(client, messages, max_tokens=400, json_mode=True)

    def name_new_category(self, sample_titles: List[str], sample_descriptions: List[str]):
        """Propose a name + description for a brand-new category.

        Returns a dict: {"name": str, "description": str}.
        """
        system = (
            "You are a taxonomist organising user pain points scraped from Reddit. "
            "Given a cluster of related painpoint titles and descriptions, "
            "propose a short category name (2-4 words) and a one-sentence description. "
            "Reply with JSON: {\"name\": \"...\", \"description\": \"...\"}."
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
        self, parent_name: str, clusters: List[List[str]]
    ):
        """Given the painpoint titles in each sub-cluster, propose a name +
        description for each. Returns a list of dicts in the same order.
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

    def name_new_category(self, sample_titles, sample_descriptions):
        if self._name_fn is not None:
            return self._name_fn(sample_titles, sample_descriptions)
        i = self._next()
        return {"name": f"AutoCat-{i}", "description": f"auto-named cluster #{i}"}

    def name_split_subcategories(self, parent_name, clusters):
        if self._split_fn is not None:
            return self._split_fn(parent_name, clusters)
        return [
            {"name": f"{parent_name}/Sub-{i + 1}", "description": f"sub-cluster #{i + 1}"}
            for i, _ in enumerate(clusters)
        ]
