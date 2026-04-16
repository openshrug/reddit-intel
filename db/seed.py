from pathlib import Path

import yaml

from . import _now, get_db

TAXONOMY_FILE = Path(__file__).parents[1] / "taxonomy.yaml"


def seed_taxonomy():
    """Populate the categories table from taxonomy.yaml if empty.

    Wrapped in BEGIN IMMEDIATE so two concurrent init_db() invocations
    can't both pass the empty-table check and double-seed. The second
    caller will block on BEGIN IMMEDIATE, then re-check the count and
    short-circuit.
    """
    if not TAXONOMY_FILE.exists():
        return

    conn = get_db()
    try:
        # BEGIN IMMEDIATE acquires the write lock — second concurrent
        # caller waits here. After acquisition we re-check the count
        # so we don't seed twice if the other caller already did.
        conn.execute("BEGIN IMMEDIATE")
        try:
            if conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0] > 0:
                conn.execute("ROLLBACK")
                return

            taxonomy = yaml.safe_load(TAXONOMY_FILE.read_text())
            now = _now()

            for parent_name, parent_data in taxonomy.items():
                conn.execute(
                    "INSERT INTO categories (name, parent_id, description, created_at) "
                    "VALUES (?,?,?,?)",
                    (parent_name, None, parent_data.get("desc", ""), now),
                )
                parent_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                for child_name, child_desc in parent_data.get("children", {}).items():
                    conn.execute(
                        "INSERT INTO categories (name, parent_id, description, created_at) "
                        "VALUES (?,?,?,?)",
                        (child_name, parent_id, child_desc, now),
                    )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
