import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parents[1] / "trends.db"
SCHEMA_FILE = Path(__file__).parent / "schema.sql"


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA_FILE.read_text())
    conn.commit()
    conn.close()

    from .seed import seed_taxonomy
    seed_taxonomy()
