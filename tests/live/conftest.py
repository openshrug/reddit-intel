"""Auto-mark every test in this subtree as `live`.

Live tests hit real Reddit + OpenAI APIs, take 15+ minutes total, cost
money, and require credentials. The `live` marker is excluded by default
via `addopts = "-m 'not live'"` in pyproject.toml — run them explicitly
with `pytest -m live` (or `pytest tests/live/<file>.py -m live` for a
single file).
"""

from pathlib import Path

import pytest

LIVE_DIR = Path(__file__).parent.resolve()


def pytest_collection_modifyitems(config, items):
    """Apply the `live` marker only to items whose file lives under tests/live/.

    pytest delivers the full collected items list to every conftest's
    `pytest_collection_modifyitems` hook, so we must filter by path —
    a naive `for item in items: item.add_marker(...)` would mark the
    entire test suite as live.
    """
    for item in items:
        if LIVE_DIR in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.live)
