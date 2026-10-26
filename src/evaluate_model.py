"""
src/evaluate_model.py

Model evaluation module for the Sales Prediction System.

Reuses the exact same data loading, feature selection, and time-based
train/test split as src/train_model.py (imported, not duplicated) so the
numbers here are guaranteed to match what train_model.py saw. For every
one of the four candidate models it:

  1. Computes MAE, MSE, RMSE, R^2 on the held-out test set
  2. Saves an Actual vs Predicted scatter plot
  3. Saves a Residual plot (residuals vs predicted values)
  4. Saves a Feature Importance plot (tree-based models only —
     Linear Regression has no `.feature_importances_`, so it's skipped
     for that plot but still gets metrics + the other two plots)

All results are written to disk:
  - reports/metrics/evaluation_metrics.csv   (one row per model)
  - reports/figures/evaluation/*.png         (3 plots per applicable model)

Run directly:
    python src/evaluate_model.py
"""

import os
import sys
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Support running both as a standalone script (`python src/evaluate_model.py`)
# and as part of the `src` package (e.g. `from src.evaluate_model import ...`
# from app.py) without duplicating the data/model-loading logic.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.train_model import (
        load_data, select_feature_columns, time_based_split,
        get_models, evaluate_model, CONFIG as TRAIN_CONFIG,
    )
except ImportError:
    from train_model import (
        load_data, select_feature_columns, time_based_split,
        get_models, evaluate_model, CONFIG as TRAIN_CONFIG,
    )

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CONFIG = {
    "figures_dir": "reports/figures/evaluation",
    "metrics_path": "reports/metrics/evaluation_metrics.csv",
    "top_n_features": 15,
    "fig_dpi": 150,
}

# Resolve defaults relative to this project, independent of the launch directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _key, _value in CONFIG.items():
    if isinstance(_value, str) and _value.startswith(("data/", "models/", "reports/")):
        CONFIG[_key] = os.path.join(_PROJECT_ROOT, _value)
if "raw_data_path" in CONFIG and not os.path.exists(CONFIG["raw_data_path"]):
    CONFIG["raw_data_path"] = os.path.join(_PROJECT_ROOT, "data/raw/sales data file.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def _safe_name(model_name: str) -> str:
    """Turn a model display name into a filesystem-safe slug."""
    return model_name.lower().replace(" ", "_")


def _save_fig(fig, filename: str) -> str:
    os.makedirs(CONFIG["figures_dir"], exist_ok=True)
    path = os.path.join(CONFIG["figures_dir"], filename)
    fig.savefig(path, dpi=CONFIG["fig_dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> '%s'", path)
    return path


# --------------------------------------------------------------------------
# DATA + MODELS
# --------------------------------------------------------------------------
def load_train_test_data():
    """
    Reuse train_model.py's pipeline pieces so the evaluation split is
    identical to what the training script used — this is what lets the
    comparison table here be trusted alongside train_model.py's own.
    """
    df = load_data(TRAIN_CONFIG["input_path"])
    feature_cols = select_feature_columns(df)
    X_train, X_test, y_train, y_test = time_based_split(df, feature_cols)
    return X_train, X_test, y_train, y_test, feature_cols


def train_all_models(X_train, y_train) -> Dict[str, object]:
    """Fit every candidate model on the training set."""
    models = get_models()
    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)
    return models


# --------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------
def compute_all_metrics(models: Dict[str, object], X_test, y_test) -> pd.DataFrame:
    """Compute MAE/MSE/RMSE/R2 for every model and return a sorted table."""
    rows = []
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        metrics = evaluate_model(y_test, y_pred)   # {"MAE":..,"MSE":..,"RMSE":..,"R2":..}
        metrics["Model"] = name
        rows.append(metrics)

    table = pd.DataFrame(rows)[["Model", "MAE", "MSE", "RMSE", "R2"]]
    table = table.sort_values("RMSE").reset_index(drop=True)
    return table, predictions


# --------------------------------------------------------------------------
# PLOTS
# --------------------------------------------------------------------------
def plot_actual_vs_predicted(y_test, y_pred, model_name: str) -> str:
    """Scatter of actual vs predicted values with a y=x reference line."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, s=15, color="steelblue")

    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color="red", linestyle="--", label="Perfect prediction (y = x)")

    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.legend()
    return _save_fig(fig, f"actual_vs_predicted_{_safe_name(model_name)}.png")


def plot_residuals(y_test, y_pred, model_name: str) -> str:
    """Residuals (actual - predicted) vs predicted values, with a zero line."""
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, s=15, color="darkorange")
    ax.axhline(0, color="red", linestyle="--")

    ax.set_title(f"Residual Plot — {model_name}")
    ax.set_xlabel("Predicted Sales")
    ax.set_ylabel("Residual (Actual - Predicted)")
    return _save_fig(fig, f"residuals_{_safe_name(model_name)}.png")


def plot_feature_importance(model, feature_cols: List[str], model_name: str) -> Optional[str]:
    """
    Bar plot of feature importances for tree-based models. Models without
    a `.feature_importances_` attribute (e.g. Linear Regression) are
    skipped here — their contribution to metrics/other plots is unaffected.
    """
    if not hasattr(model, "feature_importances_"):
        logger.info("'%s' has no feature_importances_ — skipping importance plot", model_name)
        return None

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_n = CONFIG["top_n_features"]
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    return _save_fig(fig, f"feature_importance_{_safe_name(model_name)}.png")


# --------------------------------------------------------------------------
# SAVE METRICS
# --------------------------------------------------------------------------
def save_metrics_table(table: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(CONFIG["metrics_path"]), exist_ok=True)
    table.to_csv(CONFIG["metrics_path"], index=False)
    logger.info("Saved evaluation metrics -> '%s'", CONFIG["metrics_path"])


# --------------------------------------------------------------------------
# PIPELINE
# --------------------------------------------------------------------------
def run_evaluation() -> pd.DataFrame:
    """
    Run the full evaluation pipeline: train every model, compute metrics
    for all of them, generate and save all plots, save the metrics table.
    Returns the metrics comparison table.
    """
    X_train, X_test, y_train, y_test, feature_cols = load_train_test_data()
    models = train_all_models(X_train, y_train)
    table, predictions = compute_all_metrics(models, X_test, y_test)

    print("\nEvaluation Metrics (sorted by RMSE, best first):")
    print(table.to_string(index=False))

    for name, model in models.items():
        y_pred = predictions[name]
        plot_actual_vs_predicted(y_test, y_pred, name)
        plot_residuals(y_test, y_pred, name)
        plot_feature_importance(model, feature_cols, name)

    save_metrics_table(table)

    print(f"\nAll plots saved to '{CONFIG['figures_dir']}/'")
    print(f"Metrics table saved to '{CONFIG['metrics_path']}'")

    return table


if __name__ == "__main__":
    run_evaluation()
