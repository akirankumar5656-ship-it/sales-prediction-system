"""
database/db_operations.py

CRUD helpers around the `users` and `predictions` tables. app.py imports
these instead of writing raw SQL inline.

Passwords are never stored or compared as plain text: create_user() and
verify_user() use werkzeug.security's generate_password_hash /
check_password_hash (salted hashing, bundled with Flask — no extra
dependency needed).
"""

import os
import sqlite3
import sys
from typing import Dict, List, Optional

from werkzeug.security import generate_password_hash, check_password_hash

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# --------------------------------------------------------------------------
# USERS
# --------------------------------------------------------------------------
def create_user(username: str, password: str, email: Optional[str] = None) -> Optional[int]:
    """
    Create a new user with a salted password hash. Returns the new user's
    id, or None if the username/email is already taken.
    """
    password_hash = generate_password_hash(password)
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None  # username or email already exists
    finally:
        conn.close()


def verify_user(username: str, password: str) -> Optional[Dict]:
    """
    Check a login attempt against the stored hash. Returns the user row
    (without the hash) on success, or None on a bad username/password.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()

    if row is None or not check_password_hash(row["password_hash"], password):
        return None

    user = dict(row)
    user.pop("password_hash")  # never hand the hash back to the caller/session
    return user


def get_user_by_id(user_id: int) -> Optional[Dict]:
    conn = _get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    user = dict(row)
    user.pop("password_hash")
    return user


# --------------------------------------------------------------------------
# PREDICTIONS
# --------------------------------------------------------------------------
def insert_prediction(record: Dict) -> int:
    """
    Insert one logged prediction. `record` keys must match the
    `predictions` table columns (see database/db_setup.py); optional
    fields (user_id, product, store, promotion, price) may be None —
    user_id is None for a guest (not-logged-in) prediction.
    Returns the new row's id.
    """
    record = {"user_id": None, "inputs_json": None, **record}  # default so callers can omit it
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (user_id, product, category, region, segment, ship_mode, store,
                 promotion, price, quantity, discount, order_date, predicted_sales, inputs_json)
            VALUES (:user_id, :product, :category, :region, :segment, :ship_mode, :store,
                    :promotion, :price, :quantity, :discount, :order_date, :predicted_sales, :inputs_json)
            """,
            record,
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_recent_predictions(limit: int = 20) -> List[Dict]:
    """Return the most recent N logged predictions, newest first."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id IS NULL ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_prediction_history(user_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
    """
    Full prediction history for the history page. If user_id is given,
    only that user's predictions are returned (each account sees its own
    history); if None, only guest predictions are
    returned — used when nobody is logged in.
    """
    conn = _get_connection()
    try:
        if user_id is not None:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE user_id IS NULL ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_dashboard_stats(user_id: Optional[int] = None) -> Dict:
    """
    Aggregate stats over the current account or guest predictions, for the dashboard page:
    total count, average/min/max predicted sales, and totals by category.
    Returns sensible zeroed defaults if no predictions have been logged yet.
    """
    conn = _get_connection()
    try:
        summary = conn.execute(
            """
            SELECT
                COUNT(*)            AS total_predictions,
                AVG(predicted_sales) AS avg_predicted_sales,
                MIN(predicted_sales) AS min_predicted_sales,
                MAX(predicted_sales) AS max_predicted_sales
            FROM predictions WHERE user_id IS ?
            """, (user_id,)
        ).fetchone()

        by_category = conn.execute(
            """
            SELECT category, COUNT(*) AS count, AVG(predicted_sales) AS avg_sales
            FROM predictions WHERE user_id IS ?
            GROUP BY category
            ORDER BY count DESC
            """, (user_id,)
        ).fetchall()

        return {
            "total_predictions": summary["total_predictions"] or 0,
            "avg_predicted_sales": summary["avg_predicted_sales"] or 0.0,
            "min_predicted_sales": summary["min_predicted_sales"] or 0.0,
            "max_predicted_sales": summary["max_predicted_sales"] or 0.0,
            "by_category": [dict(row) for row in by_category],
        }
    finally:
        conn.close()
