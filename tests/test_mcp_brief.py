"""MCP-first opportunity brief surface: resources, prompt, server instructions."""

from pathlib import Path

import pytest


@pytest.fixture()
def mcp_module():
    pytest.importorskip("fastmcp")
    import mcp_server
    return mcp_server


def test_brief_instructions_resource_returns_agents_md(mcp_module):
    expected = (
        Path(mcp_module.__file__).parent / "opportunity_briefs" / "AGENTS.md"
    ).read_text()
    assert mcp_module.opportunity_brief_instructions() == expected


def test_brief_template_resource_returns_synthesis_template_md(mcp_module):
    expected = (
        Path(mcp_module.__file__).parent
        / "opportunity_briefs"
        / "SYNTHESIS_TEMPLATE.md"
    ).read_text()
    assert mcp_module.opportunity_brief_template() == expected


def test_brief_layout_resource_returns_brief_layout_md(mcp_module):
    expected = (
        Path(mcp_module.__file__).parent
        / "opportunity_briefs"
        / "BRIEF_LAYOUT.md"
    ).read_text()
    assert mcp_module.opportunity_brief_layout() == expected


def test_brief_prompt_references_three_resources_and_evidence_tool(mcp_module):
    body = mcp_module.opportunity_brief(subreddit="smallbusiness")
    assert "reddit-intel://opportunity-brief-instructions" in body
    assert "reddit-intel://opportunity-brief-template" in body
    assert "reddit-intel://opportunity-brief-layout" in body
    assert "get_opportunity_evidence" in body


def test_brief_prompt_has_no_count_param(mcp_module):
    with pytest.raises(TypeError):
        mcp_module.opportunity_brief(subreddit="smallbusiness", count=5)


def test_brief_prompt_uses_brief_evidence_limit(mcp_module):
    body = mcp_module.opportunity_brief(subreddit="smallbusiness")
    expected = f"limit={mcp_module.opportunities.BRIEF_EVIDENCE_LIMIT}"
    assert expected in body


def test_triggers_appear_in_server_instructions_and_prompt(mcp_module):
    instructions = mcp_module.mcp.instructions
    prompt_body = mcp_module.opportunity_brief(subreddit="smallbusiness")
    for trigger in mcp_module.OPPORTUNITY_BRIEF_TRIGGERS:
        assert trigger in instructions, (
            f"trigger {trigger!r} missing from server instructions"
        )
        assert trigger in prompt_body, (
            f"trigger {trigger!r} missing from opportunity_brief prompt body"
        )
