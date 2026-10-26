"""
database/db_setup.py

Creates the SQLite schema used by the web app: a `users` table
(passwords stored as salted hashes, never plain text — see
database/db_operations.py's use of werkzeug.security) and a
`predictions` table that logs every prediction made through the app,
optionally linked to the logged-in user.

Run once before starting the app (app.py also calls this automatically on
startup, so manual execution is only needed if you want to reset the DB).

Run directly:
    python database/db_setup.py
"""

import os
import sqlite3
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE,
    email           TEXT UNIQUE,
    password_hash   TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER,               -- NULL for guest predictions
    product         TEXT,
    category        TEXT NOT NULL,
    region          TEXT NOT NULL,
    segment         TEXT NOT NULL,
    ship_mode       TEXT NOT NULL,
    store           TEXT,
    promotion       TEXT,
    price           REAL,
    quantity        REAL NOT NULL,
    discount        REAL NOT NULL,
    order_date      TEXT NOT NULL,          -- the order date supplied as input
    predicted_sales REAL NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),  -- date/time the prediction was made
    FOREIGN KEY (user_id) REFERENCES users (id)
);
"""


def _ensure_user_id_column(conn: sqlite3.Connection) -> None:
    """
    Migration safety net: if a `predictions` table already exists from
    before the `users` table was introduced, add the `user_id` column
    rather than failing or requiring a manual DB reset.
    """
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    if existing_columns and "user_id" not in existing_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER REFERENCES users(id)")
        conn.commit()
        print("Migrated existing 'predictions' table: added 'user_id' column")


def init_db(db_path: str = None) -> None:
    """Create the users/predictions tables if they don't already exist."""
    db_path = db_path or config.DATABASE_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        conn.commit()
        _ensure_user_id_column(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        if "inputs_json" not in columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN inputs_json TEXT")
            conn.commit()
    finally:
        conn.close()

    print(f"Database ready at '{db_path}'")


if __name__ == "__main__":
    init_db()
