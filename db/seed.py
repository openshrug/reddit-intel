from pathlib import Path

import yaml

from . import get_db, _now

TAXONOMY_FILE = Path(__file__).parents[1] / "taxonomy.yaml"


def seed_taxonomy():
    """Populate the categories table from taxonomy.yaml if empty."""
    conn = get_db()
    if conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0] > 0:
        conn.close()
        return

    if not TAXONOMY_FILE.exists():
        conn.close()
        return

    taxonomy = yaml.safe_load(TAXONOMY_FILE.read_text())
    now = _now()

    for parent_name, parent_data in taxonomy.items():
        conn.execute(
            "INSERT INTO categories (name, parent_id, description, created_at) VALUES (?,?,?,?)",
            (parent_name, None, parent_data.get("desc", ""), now),
        )
        parent_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for child_name, child_desc in parent_data.get("children", {}).items():
            conn.execute(
                "INSERT INTO categories (name, parent_id, description, created_at) VALUES (?,?,?,?)",
                (child_name, parent_id, child_desc, now),
            )

    conn.commit()
    conn.close()
