from . import get_db


def get_category_by_name(name):
    """Return a category row by name, or None."""
    if not name:
        return None
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM categories WHERE name = ?", (name,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_category_id_by_name(name):
    """Return category id for a name, or None if not found."""
    if not name:
        return None
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM categories WHERE name = ?", (name,)
    ).fetchone()
    conn.close()
    return row["id"] if row else None


def get_category_list_flat():
    """Return leaf categories as 'Parent > Child' for prompts."""
    conn = get_db()
    rows = conn.execute("""
        SELECT c.name AS child, p.name AS parent, c.description
        FROM categories c
        LEFT JOIN categories p ON c.parent_id = p.id
        WHERE c.parent_id IS NOT NULL
        ORDER BY p.name, c.name
    """).fetchall()
    conn.close()
    return [
        {
            "path": f"{r['parent']} > {r['child']}",
            "name": r["child"],
            "description": r["description"],
        }
        for r in rows
    ]


def get_all_categories():
    """Return all categories (roots + children) ordered hierarchically."""
    conn = get_db()
    rows = conn.execute("""
        SELECT c.*, p.name AS parent_name
        FROM categories c
        LEFT JOIN categories p ON c.parent_id = p.id
        ORDER BY COALESCE(p.name, c.name), c.parent_id IS NULL DESC, c.name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_root_categories():
    """Return top-level (root) categories only."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM categories WHERE parent_id IS NULL ORDER BY name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
