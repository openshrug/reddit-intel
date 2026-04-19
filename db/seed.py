from pathlib import Path

import yaml

from . import _now, get_db, in_clause_placeholders
from .category_retrieval import sync_category_fts

TAXONOMY_FILE = Path(__file__).parents[1] / "taxonomy.yaml"


def _taxonomy_names():
    """Names of every category declared in taxonomy.yaml (roots + children)."""
    if not TAXONOMY_FILE.exists():
        return set()
    taxonomy = yaml.safe_load(TAXONOMY_FILE.read_text()) or {}
    names = set()
    for root_name, root_data in taxonomy.items():
        names.add(root_name)
        for child_name in (root_data.get("children") or {}).keys():
            names.add(child_name)
    return names


def backfill_is_seed():
    """Flag is_seed=1 on any category whose name matches the current
    taxonomy.yaml, plus the Uncategorized sentinel.

    Runs every init_db() — idempotent, O(one UPDATE) on the category
    count. Needed because the is_seed column was added as a migration
    with DEFAULT 0: an existing DB created before the migration has
    every historically-seeded category flagged 0. Matching by name here
    lets us retro-stamp them without needing a one-shot script.

    If taxonomy.yaml gains a new entry whose name collides with an
    existing runtime-minted category, that category gets promoted to
    seed — intentional: the taxonomist has decided this IS part of the
    curated tree now.
    """
    names = _taxonomy_names() | {"Uncategorized"}
    if not names:
        return
    names = list(names)
    conn = get_db()
    try:
        conn.execute(
            f"UPDATE categories SET is_seed = 1 "
            f"WHERE name IN ({in_clause_placeholders(len(names))}) "
            f"AND is_seed = 0",
            names,
        )
        conn.commit()
    finally:
        conn.close()


def seed_taxonomy():
    """Ensure every entry in taxonomy.yaml exists in the categories table.

    Idempotent and additive: runs on every init_db(). INSERT OR IGNORE on
    the UNIQUE name constraint means already-present categories stay
    untouched (description + parent preserved even if taxonomy.yaml has
    since been edited — we don't silently rewrite curated state that
    might have custom edits in the DB). New entries in taxonomy.yaml
    land as seed rows on the next init_db().

    Previously this short-circuited if the table had any rows at all,
    which meant adding a new root/child to taxonomy.yaml never
    propagated to an existing DB — users had to drop trends.db to see
    new seed categories. Running as INSERT OR IGNORE per row fixes that
    without risk to already-seeded rows.

    Concurrency: `BEGIN IMMEDIATE` serializes concurrent init_db() calls
    so the INSERT OR IGNORE block runs atomically; any second caller
    waits, then runs its own no-op inserts.
    """
    if not TAXONOMY_FILE.exists():
        return

    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            taxonomy = yaml.safe_load(TAXONOMY_FILE.read_text()) or {}
            now = _now()

            for parent_name, parent_data in taxonomy.items():
                parent_desc = parent_data.get("desc", "")
                conn.execute(
                    "INSERT OR IGNORE INTO categories "
                    "(name, parent_id, description, created_at, is_seed) "
                    "VALUES (?,?,?,?,1)",
                    (parent_name, None, parent_desc, now),
                )
                parent_id = conn.execute(
                    "SELECT id FROM categories WHERE name = ?", (parent_name,),
                ).fetchone()[0]
                sync_category_fts(conn, parent_id, parent_name, parent_desc)

                for child_name, child_desc in (parent_data.get("children") or {}).items():
                    conn.execute(
                        "INSERT OR IGNORE INTO categories "
                        "(name, parent_id, description, created_at, is_seed) "
                        "VALUES (?,?,?,?,1)",
                        (child_name, parent_id, child_desc, now),
                    )
                    child_id = conn.execute(
                        "SELECT id FROM categories WHERE name = ?", (child_name,),
                    ).fetchone()[0]
                    sync_category_fts(conn, child_id, child_name, child_desc)

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
