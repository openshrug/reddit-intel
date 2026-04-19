"""Per-snapshot human-readable markdown dump.

Sections (kept aligned with the per-dimension files in
``instructions/`` so the evaluator agent can cross-reference):

1. Header + totals
2. New pendings since last snapshot     -> extraction quality (dim 1)
3. Pending dedup groups                  -> pending dedup (dim 2)
4. Painpoints with multiple sources      -> pending->painpoint merge (dim 3)
5. Category tree                         -> category assignment (dim 4)
6. Audit log (post-sweep snapshot only)

All SQL-bound work is delegated to ``inspect_db`` and the existing
``db/queries.py`` helpers; this module only formats.
"""

import textwrap

import db
from db.queries import get_painpoint_evidence, get_top_painpoints

from . import inspect_db

# Sampling caps -- keep dump.md scannable on screen, tunable in one place.
NEW_PENDINGS_PER_SUBREDDIT = 25
DEDUP_GROUPS_LIMIT = 30
MULTI_SOURCE_PAINPOINTS_LIMIT = 30
EVIDENCE_PER_PAINPOINT = 6
AUDIT_LOG_LIMIT = 200


def write_dump(snapshot_path, out_path, *, previous_path=None,
               extras=None):
    """Render the dump for ``snapshot_path`` and write it to ``out_path``.

    ``previous_path`` powers the "new pendings since last snapshot"
    section -- pass ``None`` for the clean snapshot. ``extras`` is an
    optional dict of stage-specific payloads (e.g. the analyze() summary
    for this subreddit, the sweep summary) that gets rendered in the
    header.
    """
    extras = extras or {}
    sections = []

    with inspect_db.open_snapshot(snapshot_path):
        sections.append(_render_header(snapshot_path, extras))
        sections.append(_render_new_pendings(previous_path))
        sections.append(_render_dedup_groups())
        sections.append(_render_multi_source_painpoints())
        sections.append(_render_category_tree())
        sections.append(_render_audit_log())

    out_path.write_text("\n\n".join(sections).rstrip() + "\n")


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header(snapshot_path, extras):
    """Title + global totals + any stage-specific extras (analyze /
    sweep summary)."""
    conn = db.get_db()
    try:
        totals = {
            "posts": conn.execute(
                "SELECT COUNT(*) FROM posts").fetchone()[0],
            "comments": conn.execute(
                "SELECT COUNT(*) FROM comments").fetchone()[0],
            "pending_painpoints": conn.execute(
                "SELECT COUNT(*) FROM pending_painpoints").fetchone()[0],
            "painpoints": conn.execute(
                "SELECT COUNT(*) FROM painpoints").fetchone()[0],
            "categories": conn.execute(
                "SELECT COUNT(*) FROM categories").fetchone()[0],
        }
    finally:
        conn.close()

    lines = [
        f"# Snapshot dump: `{snapshot_path.parent.name}`",
        "",
        f"DB file: `{snapshot_path}`",
        "",
        "## Totals",
        "",
        f"- posts: **{totals['posts']}**",
        f"- comments: **{totals['comments']}**",
        f"- pending_painpoints: **{totals['pending_painpoints']}**",
        f"- painpoints: **{totals['painpoints']}**",
        f"- categories: **{totals['categories']}**",
    ]

    if "analyze_summary" in extras and extras["analyze_summary"]:
        s = extras["analyze_summary"]
        lines += [
            "",
            f"### `analyze({s.get('subreddit')!r})` summary",
            "",
            f"- posts_scraped: {s.get('posts_scraped')}",
            f"- posts_persisted: {s.get('posts_persisted')}",
            f"- comments_persisted: {s.get('comments_persisted')}",
            f"- painpoints_extracted: {s.get('painpoints_extracted')}",
            f"- painpoints_linked: {s.get('painpoints_linked')}",
        ]
        if s.get("promote_error"):
            lines.append(f"- promote_error: `{s['promote_error']}`")

    if "sweep_summary" in extras and extras["sweep_summary"]:
        lines += [
            "",
            "### `run_sweep()` summary",
            "",
            "| event | proposed | accepted |",
            "| --- | ---: | ---: |",
        ]
        for k, v in sorted(extras["sweep_summary"].items()):
            lines.append(
                f"| {k} | {v.get('proposed', 0)} | {v.get('accepted', 0)} |"
            )

    return "\n".join(lines)


def _render_new_pendings(previous_path):
    """Section 2: per-subreddit, sample of pendings introduced in this
    stage (id > prev.max(id))."""
    lines = [
        "## 2. New pendings since previous snapshot (extraction quality)",
        "",
        "_Use this section to evaluate dimension (1): pending painpoint "
        "extraction from a single subreddit._",
        "",
    ]

    if previous_path is None:
        lines.append(
            "_No previous snapshot -- this is the clean / initial state._"
        )
        return "\n".join(lines)

    with inspect_db.open_snapshot(previous_path):
        prev_max_pp = inspect_db.get_max_pending_id()

    subreddits = inspect_db.list_distinct_subreddits()
    any_new = False

    for sr in subreddits:
        new_pendings = inspect_db.list_pending_painpoints_for_subreddit(
            sr, since_pp_id=prev_max_pp,
        )
        if not new_pendings:
            continue
        any_new = True
        lines += [
            "",
            f"### r/{sr} -- {len(new_pendings)} new pendings",
            "",
            f"_Showing up to {NEW_PENDINGS_PER_SUBREDDIT} (largest "
            f"`extra_source_count` first)._",
            "",
        ]
        sample = sorted(
            new_pendings,
            key=lambda r: (-r["extra_source_count"], r["pending_id"]),
        )[:NEW_PENDINGS_PER_SUBREDDIT]
        for pp in sample:
            lines.append(_format_pending_row(pp))

    if not any_new:
        lines.append("_No new pendings in this snapshot._")
    return "\n".join(lines)


def _render_dedup_groups():
    """Section 3: pendings whose ``pending_painpoint_sources`` row count
    > 0 -- the PENDING_MERGE_THRESHOLD collapses."""
    lines = [
        "## 3. Pending dedup groups (extra-source count >= 1)",
        "",
        "_Use this section to evaluate dimension (2): pending painpoint "
        "deduping. Each group is one pending row plus every additional "
        "(post, comment) the embedding-cosine merge attached to it. "
        "Judge same-pain vs different-pain by reading the source quotes._",
        "",
    ]
    groups = inspect_db.list_pending_dedup_groups(limit=DEDUP_GROUPS_LIMIT)
    if not groups:
        lines.append(
            "_No pending dedup groups in this snapshot "
            "(no observation-level merges fired)._"
        )
        return "\n".join(lines)

    lines.append(f"_Showing top {len(groups)} groups by extras._")
    lines.append("")
    for g in groups:
        lines.append(
            f"### Pending #{g['pending_id']} "
            f"({g['extra_source_count']} extras) -- {_quote(g['title'])}"
        )
        if g["description"]:
            lines.append(f"> {_short(g['description'], 250)}")
        lines.append("")
        lines.append("**Primary source:**")
        lines.append(_format_source_block(g["primary"]))
        lines.append("")
        lines.append("**Extra sources collapsed onto this pending:**")
        lines.append("")
        for extra in g["extras"]:
            lines.append(_format_source_block(extra))
        lines.append("")
    return "\n".join(lines)


def _render_multi_source_painpoints():
    """Section 4: painpoints with signal_count > 1 -- the merged-side
    deduping done by the promoter (and post-sweep painpoint_merge)."""
    lines = [
        "## 4. Painpoints with multiple linked pendings",
        "",
        "_Use this section to evaluate dimension (3): merging of pending "
        "painpoints into a single painpoint. Each painpoint is one row "
        "in `painpoints`; the linked pendings are listed below it. Judge "
        "whether the merger is real or a false positive by comparing the "
        "pendings' titles + source quotes._",
        "",
    ]
    top = get_top_painpoints(limit=MULTI_SOURCE_PAINPOINTS_LIMIT)
    multi = [r for r in top if r["signal_count"] > 1]
    if not multi:
        lines.append(
            "_No painpoint has signal_count > 1 in this snapshot._"
        )
        return "\n".join(lines)

    lines.append(
        f"_Showing top {len(multi)} painpoints by signal_count "
        "(then severity)._"
    )
    lines.append("")
    for pp in multi:
        cat = pp.get("category") or "(none)"
        lines.append(
            f"### Painpoint #{pp['id']} -- sig={pp['signal_count']} "
            f"sev={pp['severity']} -- cat=`{cat}`"
        )
        lines.append(f"**{_quote(pp['title'])}**")
        if pp.get("description"):
            lines.append(f"> {_short(pp['description'], 250)}")
        lines.append("")

        evidence = get_painpoint_evidence(pp["id"])[:EVIDENCE_PER_PAINPOINT]
        lines.append(f"**Linked pendings ({len(evidence)} of "
                     f"{pp['signal_count']} shown):**")
        lines.append("")
        for ev in evidence:
            lines.append(_format_evidence_block(ev))
        if len(evidence) < pp["signal_count"]:
            lines.append(
                f"_... and {pp['signal_count'] - len(evidence)} more "
                "linked pendings (use `get_painpoint_evidence(id)` "
                "for the full list)._"
            )
        lines.append("")
    return "\n".join(lines)


def _render_category_tree():
    """Section 5: full category tree, recursive member counts. Empty
    subtrees are pruned for readability."""
    lines = [
        "## 5. Category tree (post-promote / post-sweep state)",
        "",
        "_Use this section to evaluate dimension (4): category "
        "assignment. Walk the tree and ask: are leaf categories "
        "coherent? Are siblings genuinely distinct? Is anything a "
        "near-duplicate? In snapshots 1-3 this is the promote-time "
        "placement only (much will be in `Uncategorized`); in the "
        "post-sweep snapshot it reflects the final taxonomy._",
        "",
    ]
    roots = inspect_db.render_category_tree()
    rendered_any = False
    for root in roots:
        if root["total_painpoints"] == 0 and root["name"] != "Uncategorized":
            continue
        rendered_any = True
        lines.append(_format_tree_node(root, indent=0))

    if not rendered_any:
        lines.append("_No categories with painpoints._")
    return "\n".join(lines)


def _render_audit_log():
    """Section 6: rows from ``category_events`` -- only present in the
    post-sweep snapshot (the sweep is what writes to that table)."""
    events = inspect_db.get_category_events(limit=AUDIT_LOG_LIMIT)
    lines = [
        "## 6. Category worker audit log (`category_events`)",
        "",
    ]
    if not events:
        lines.append(
            "_No `category_events` rows -- this snapshot pre-dates the "
            "sweep (analyze() doesn't run the category worker)._"
        )
        return "\n".join(lines)

    lines.append(f"_Showing {len(events)} events (oldest first)._")
    lines.append("")
    lines.append("| id | event_type | accepted | metric | reason |")
    lines.append("| ---: | --- | :---: | --- | --- |")
    for e in events:
        verdict = "ACCEPT" if e["accepted"] else "REJECT"
        metric = (
            f"{e['metric_name']}={e['metric_value']:.3f} "
            f"(thr={e['threshold']:.3f})"
        )
        reason = (e["reason"] or "").replace("|", "\\|")
        lines.append(
            f"| {e['id']} | {e['event_type']} | {verdict} | {metric} "
            f"| {_short(reason, 80)} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------

def _format_pending_row(pp):
    """One pending painpoint as a markdown sub-section."""
    cat = pp.get("pending_category_name") or "(none)"
    parts = [
        f"#### Pending #{pp['pending_id']} -- sev={pp['pending_severity']} "
        f"-- cat=`{cat}` -- extras={pp['extra_source_count']}",
        f"**{_quote(pp['pending_title'])}**",
    ]
    if pp.get("pending_description"):
        parts.append(f"> {_short(pp['pending_description'], 250)}")
    if pp.get("quoted_text"):
        parts.append(f"- quoted: \"{_short(pp['quoted_text'], 120)}\"")
    parts.append(
        f"- post: r/{pp['subreddit']} -- {_quote(pp['post_title'])} "
        f"({pp['post_permalink']})"
    )
    if pp.get("comment_id"):
        parts.append(
            f"- comment ({pp['comment_score']} pts): "
            f"\"{_short(pp.get('comment_body') or '', 200)}\" "
            f"({pp['comment_permalink']})"
        )
    return "\n".join(parts) + "\n"


def _format_source_block(src):
    """One (post, comment?) tuple as an indented bullet block."""
    parts = [
        f"- r/{src.get('subreddit')} post {src.get('post_id')}: "
        f"{_quote(src.get('post_title') or '')} ({src.get('post_permalink')})"
    ]
    if src.get("comment_id"):
        parts.append(
            f"  - comment {src['comment_id']}: "
            f"\"{_short(src.get('comment_body') or '', 200)}\" "
            f"({src.get('comment_permalink') or ''})"
        )
    return "\n".join(parts)


def _format_evidence_block(ev):
    """One row from ``get_painpoint_evidence``."""
    parts = [
        f"- pending #{ev['pending_id']} (sev={ev['pending_severity']}): "
        f"{_quote(ev['pending_title'])}"
    ]
    if ev.get("pending_description"):
        parts.append(f"  - desc: {_short(ev['pending_description'], 200)}")
    if ev.get("quoted_text"):
        parts.append(f"  - quoted: \"{_short(ev['quoted_text'], 120)}\"")
    parts.append(
        f"  - r/{ev['subreddit']} post: "
        f"{_quote(ev['post_title'])} ({ev['post_permalink']})"
    )
    if ev.get("comment_body"):
        parts.append(
            f"  - comment: \"{_short(ev['comment_body'], 200)}\""
        )
    return "\n".join(parts)


def _format_tree_node(node, *, indent):
    """Recursive markdown bullet rendering of one category subtree."""
    pad = "  " * indent
    seed_tag = " (seed)" if node.get("is_seed") else ""
    lines = [
        f"{pad}- **{node['name']}**{seed_tag} -- "
        f"id={node['id']} direct={node['direct_painpoints']} "
        f"total={node['total_painpoints']}"
    ]
    if node.get("description"):
        lines.append(f"{pad}  - _{_short(node['description'], 200)}_")
    for child in node["children"]:
        if (child["total_painpoints"] == 0
                and child["name"] != "Uncategorized"):
            continue
        lines.append(_format_tree_node(child, indent=indent + 1))
    return "\n".join(lines)


def _quote(text):
    return (text or "").replace("\n", " ").strip()


def _short(text, n):
    return textwrap.shorten(_quote(text), width=n, placeholder="...")
