import os

import pytest

import db
import opportunities
from db.opportunity_queries import get_opportunity_evidence_rows
from db.painpoints import save_pending_painpoint
from db.posts import upsert_comment, upsert_post
from opportunities import get_opportunity_evidence


@pytest.fixture()
def opportunity_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "opportunities.db")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    db.init_db()
    _seed_opportunity_data()
    return db.DB_PATH


def test_query_utility_groups_local_and_cross_subreddit_evidence(opportunity_db):
    rows = get_opportunity_evidence_rows("smallbusiness", limit=5)

    assert rows[0]["title"] == "Manual follow-up loses small-business leads"
    assert rows[0]["local_signal_count"] == 2
    assert rows[0]["global_signal_count"] == 3
    assert rows[0]["subreddits_seen"] == ["Entrepreneur", "smallbusiness"]

    local = rows[0]["local_evidence"]
    cross = rows[0]["cross_subreddit_evidence"]
    assert len(local) == 2
    assert len(cross) == 1
    assert {item["subreddit"] for item in local} == {"smallbusiness"}
    assert {item["subreddit"] for item in cross} == {"Entrepreneur"}
    assert all(item["source_permalink"].startswith("https://reddit.com/") for item in local + cross)


def test_opportunity_response_is_agent_ready_without_openai(opportunity_db):
    assert "OPENAI_API_KEY" not in os.environ

    response = get_opportunity_evidence("r/smallbusiness", limit=5)

    assert response["requested_subreddit"] == "smallbusiness"
    assert response["total_painpoints"] == 2
    first = response["painpoints"][0]
    assert first["title"] == "Manual follow-up loses small-business leads"
    assert first["evidence_strength"]["local_signal_count"] == 2
    assert first["evidence_strength"]["global_signal_count"] == 3
    assert first["local_evidence"][0]["source_permalink"]
    assert first["cross_subreddit_evidence"][0]["source_permalink"]

    guidelines = response["agent_guidelines"]
    assert "evidence-only shortlist" in guidelines["synthesis_prompt"]
    assert "Render every evidence quote" in guidelines["core_rules"][3]
    assert "deduped sources" in response["caveats"][2]


def test_synthesis_prompt_loads_from_template(opportunity_db, tmp_path, monkeypatch):
    template = tmp_path / "template.md"
    template.write_text("Custom brief template for r/{subreddit}")
    monkeypatch.setattr(opportunities, "SYNTHESIS_TEMPLATE", template)

    response = get_opportunity_evidence("smallbusiness", limit=1)

    assert response["agent_guidelines"]["synthesis_prompt"] == (
        "Custom brief template for r/smallbusiness"
    )


def test_category_filter_limits_candidates(opportunity_db):
    response = get_opportunity_evidence(
        "smallbusiness", limit=5, category="CRM & Sales",
    )

    assert response["total_painpoints"] == 1
    assert response["painpoints"][0]["category"] == "CRM & Sales"


def test_mcp_wrapper_returns_same_evidence_shape(opportunity_db):
    pytest.importorskip("fastmcp")
    import mcp_server

    response = mcp_server.get_opportunity_evidence("smallbusiness", limit=5)

    assert response["requested_subreddit"] == "smallbusiness"
    assert response["painpoints"][0]["local_evidence"][0]["source_permalink"]
    assert "agent_guidelines" in response


def _seed_opportunity_data():
    crm_category = _category_id("CRM & Sales")

    local_post = _post(
        "t3_smallbiz_followup",
        "smallbusiness",
        "Customer follow-up keeps slipping",
        score=120,
    )
    local_comment = _comment(
        local_post,
        "t1_smallbiz_followup",
        "I keep forgetting to follow up with leads after estimates.",
        subreddit="smallbusiness",
        score=31,
    )
    local_pending = save_pending_painpoint(
        local_post,
        "Manual follow-up loses small-business leads",
        comment_id=local_comment,
        category_name="CRM & Sales",
        description="Small-business owners lose leads because follow-up is manual.",
        quoted_text="I keep forgetting to follow up with leads after estimates.",
        severity=8,
    )

    second_local_post = _post(
        "t3_smallbiz_crm",
        "smallbusiness",
        "Spreadsheet CRM is falling apart",
        score=75,
    )
    second_local_pending = save_pending_painpoint(
        second_local_post,
        "Manual follow-up loses small-business leads",
        category_name="CRM & Sales",
        description="Lead tracking is scattered across spreadsheets and memory.",
        quoted_text="My spreadsheet CRM is falling apart.",
        severity=7,
    )

    cross_post = _post(
        "t3_entrepreneur_followup",
        "Entrepreneur",
        "Following up with prospects is where I drop the ball",
        score=90,
    )
    cross_comment = _comment(
        cross_post,
        "t1_entrepreneur_followup",
        "The hard part is remembering who needs a nudge this week.",
        subreddit="Entrepreneur",
        score=22,
    )
    cross_pending = save_pending_painpoint(
        cross_post,
        "Manual follow-up loses small-business leads",
        comment_id=cross_comment,
        category_name="CRM & Sales",
        description="Operators miss warm prospects because reminders are ad hoc.",
        quoted_text="The hard part is remembering who needs a nudge this week.",
        severity=6,
    )

    _painpoint(
        "Manual follow-up loses small-business leads",
        "Small operators lose warm leads because follow-up lives in memory, inboxes, or spreadsheets.",
        severity=8,
        category_id=crm_category,
        pending_ids=[local_pending, second_local_pending, cross_pending],
    )

    booking_post = _post(
        "t3_smallbiz_booking",
        "smallbusiness",
        "No-shows wreck my schedule",
        score=40,
    )
    booking_pending = save_pending_painpoint(
        booking_post,
        "No-shows create scheduling waste",
        description="Appointment no-shows leave small teams with unusable gaps.",
        quoted_text="No-shows wreck my schedule.",
        severity=5,
    )
    _painpoint(
        "No-shows create scheduling waste",
        "Appointment no-shows leave small teams with unusable gaps.",
        severity=5,
        pending_ids=[booking_pending],
    )


def _post(name, subreddit, title, *, score):
    return upsert_post({
        "name": name,
        "subreddit": subreddit,
        "title": title,
        "selftext": "",
        "score": score,
        "num_comments": 1,
        "permalink": f"/r/{subreddit}/comments/{name.removeprefix('t3_')}/post/",
        "created_utc": 1712000000.0,
        "is_self": True,
    })


def _comment(post_id, name, body, *, subreddit, score):
    return upsert_comment(post_id, {
        "name": name,
        "body": body,
        "score": score,
        "permalink": f"/r/{subreddit}/comments/post/{name}/",
        "created_utc": 1712001000.0,
    })


def _painpoint(title, description, *, severity, pending_ids, category_id=None):
    conn = db.get_db()
    try:
        now = db._now()
        conn.execute(
            "INSERT INTO painpoints "
            "(title, description, severity, signal_count, category_id, first_seen, last_updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, description, severity, len(pending_ids), category_id, now, now),
        )
        painpoint_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for pending_id in pending_ids:
            conn.execute(
                "INSERT INTO painpoint_sources (painpoint_id, pending_painpoint_id) "
                "VALUES (?, ?)",
                (painpoint_id, pending_id),
            )
        conn.commit()
        return painpoint_id
    finally:
        conn.close()


def _category_id(name):
    conn = db.get_db()
    try:
        row = conn.execute("SELECT id FROM categories WHERE name = ?", (name,)).fetchone()
        return row["id"] if row else None
    finally:
        conn.close()
