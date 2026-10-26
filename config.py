"""
config.py

Centralized path configuration for the Flask app. Every module (app.py,
database/db_operations.py) imports paths from here instead of hardcoding
strings, so moving a folder means editing one file.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(BASE_DIR, "data", "database.db")

MODEL_PATH = os.path.join(BASE_DIR, "models", "sales_model.pkl")
FEATURE_LIST_PATH = os.path.join(BASE_DIR, "models", "feature_columns.json")

REPORTS_FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
REPORTS_EVAL_FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures", "evaluation")
MODEL_COMPARISON_PATH = os.path.join(BASE_DIR, "reports", "metrics", "model_comparison.csv")
EVALUATION_METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics", "evaluation_metrics.csv")

# Dropdown choices shown on the prediction form. These reflect the
# categorical values the model was actually trained on (Superstore-style
# schema) — keep in sync with your dataset if it differs.
CATEGORY_CHOICES = ["Furniture", "Office Supplies", "Technology"]
REGION_CHOICES = ["East", "West", "Central", "South"]
SEGMENT_CHOICES = ["Consumer", "Corporate", "Home Office"]
SHIP_MODE_CHOICES = ["Standard Class", "Second Class", "First Class", "Same Day"]

# Keep local sessions stable without shipping a public signing key.
from pathlib import Path
import secrets
_secret_file = Path(BASE_DIR) / "data" / ".session_secret"
_secret_file.parent.mkdir(parents=True, exist_ok=True)
if not os.environ.get("SECRET_KEY") and not _secret_file.exists():
    try:
        with _secret_file.open("x") as _stream:
            _stream.write(secrets.token_hex(32))
        _secret_file.chmod(0o600)
    except FileExistsError:
        pass
SECRET_KEY = os.environ.get("SECRET_KEY") or _secret_file.read_text().strip()
