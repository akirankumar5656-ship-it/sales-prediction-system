"""
src/analytics.py

Computes the numbers shown on the Analytics Dashboard directly from the
actual dataset and the actual saved model results — nothing here is
hardcoded or simulated. If a required file is missing, the corresponding
section returns None (or an explicit "unavailable" marker) rather than a
fabricated placeholder, and the template is responsible for showing an
honest "no data yet" state instead of guessing.
"""

import os
import sys
import logging
from typing import Dict, Optional

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.file_utils import read_csv_robust
except ImportError:
    from file_utils import read_csv_robust

CONFIG = {
    # Prefer the raw dataset (has Product Name / human-readable labels);
    # fall back to the engineered dataset if raw isn't present.
    "raw_data_path": "data/raw/sales.csv",
    "processed_data_path": "data/processed/sales_processed.csv",

    "date_column": "Order Date",
    "target_column": "Sales",
    "product_column": "Product Name",
    "category_column": "Category",
    "region_column": "Region",

    "model_comparison_path": "reports/metrics/model_comparison.csv",
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


def _load_dataset() -> Optional[pd.DataFrame]:
    """Load whichever dataset file actually exists; return None if neither does."""
    for path in [CONFIG["raw_data_path"], CONFIG["processed_data_path"]]:
        if os.path.exists(path):
            df = read_csv_robust(path)
            date_col = CONFIG["date_column"]
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            logger.info("Analytics loaded '%s' -> shape %s", path, df.shape)
            return df
    logger.warning("No dataset found at '%s' or '%s'", CONFIG["raw_data_path"], CONFIG["processed_data_path"])
    return None


def compute_summary_stats(df: pd.DataFrame) -> Dict:
    """Total sales, average sales, number of distinct products, best-selling product."""
    target = CONFIG["target_column"]
    product_col = CONFIG["product_column"]

    stats = {
        "total_sales": None,
        "average_sales": None,
        "number_of_products": None,
        "best_selling_product": None,
        "best_selling_product_sales": None,
    }

    if target in df.columns:
        stats["total_sales"] = float(df[target].sum())
        stats["average_sales"] = float(df[target].mean())

    if product_col in df.columns:
        stats["number_of_products"] = int(df[product_col].nunique())

        if target in df.columns:
            totals_by_product = df.groupby(product_col)[target].sum().sort_values(ascending=False)
            if not totals_by_product.empty:
                stats["best_selling_product"] = totals_by_product.index[0]
                stats["best_selling_product_sales"] = float(totals_by_product.iloc[0])

    return stats


def compute_sales_trend(df: pd.DataFrame) -> Optional[Dict]:
    """Monthly total sales — returned as parallel label/value lists for Chart.js."""
    date_col, target = CONFIG["date_column"], CONFIG["target_column"]
    if date_col not in df.columns or target not in df.columns:
        return None

    series = (
        df.dropna(subset=[date_col])
        .set_index(date_col)[target]
        .resample("ME")
        .sum()
    )
    if series.empty:
        return None

    return {
        "labels": [d.strftime("%b %Y") for d in series.index],
        "values": [round(v, 2) for v in series.values.tolist()],
    }


def compute_group_totals(df: pd.DataFrame, group_col: str) -> Optional[Dict]:
    """Total sales grouped by an arbitrary column (Category / Region), sorted descending."""
    target = CONFIG["target_column"]
    if group_col not in df.columns or target not in df.columns:
        return None

    totals = df.groupby(group_col)[target].sum().sort_values(ascending=False)
    if totals.empty:
        return None

    return {
        "labels": totals.index.tolist(),
        "values": [round(v, 2) for v in totals.values.tolist()],
    }


def load_model_accuracy() -> Optional[Dict]:
    """
    Read the actual saved model comparison table (produced by
    src/train_model.py) and return the best model's real metrics.
    Returns None if that file doesn't exist yet — never invents numbers.
    """
    path = CONFIG["model_comparison_path"]
    if not os.path.exists(path):
        logger.warning("Model comparison file not found at '%s'", path)
        return None

    table = pd.read_csv(path)
    if table.empty:
        return None

    table = table.sort_values("RMSE")
    best = table.iloc[0]

    return {
        "model_name": best["Model"],
        "mae": float(best["MAE"]),
        "mse": float(best["MSE"]),
        "rmse": float(best["RMSE"]),
        "r2": float(best["R2"]),
        "all_models": table.to_dict(orient="records"),
    }


def get_analytics_data() -> Dict:
    """
    Single entry point the Flask route calls. Assembles every section
    independently so a missing piece (e.g. no model trained yet) doesn't
    prevent the rest of the dashboard from rendering with real data.
    """
    df = _load_dataset()

    if df is None:
        return {
            "data_available": False,
            "summary": None,
            "trend": None,
            "by_category": None,
            "by_region": None,
            "model_accuracy": load_model_accuracy(),
        }

    return {
        "data_available": True,
        "summary": compute_summary_stats(df),
        "trend": compute_sales_trend(df),
        "by_category": compute_group_totals(df, CONFIG["category_column"]),
        "by_region": compute_group_totals(df, CONFIG["region_column"]),
        "model_accuracy": load_model_accuracy(),
        "advertising": compute_advertising_stats(df),
    }


def compute_advertising_stats(df):
    """Describe the observed advertising data without assuming currency units."""
    if df is None or df.empty or not {"TV", "Radio", "Newspaper", "Sales"}.issubset(df.columns):
        return None
    channels = []
    total = float(df[["TV", "Radio", "Newspaper"]].sum().sum())
    for field in ["TV", "Radio", "Newspaper"]:
        amount = float(df[field].sum())
        channels.append({"name": field, "min": float(df[field].min()),
                         "max": float(df[field].max()), "median": float(df[field].median()),
                         "mean": float(df[field].mean()), "share": amount / total * 100 if total else 0})
    return {"rows": len(df), "channels": channels, "average_sales": float(df["Sales"].mean()),
            "min_sales": float(df["Sales"].min()), "max_sales": float(df["Sales"].max())}


def prediction_insight(df, inputs, prediction):
    stats = compute_advertising_stats(df)
    if stats is None:
        return None
    warnings = [f"{channel['name']} spend is outside the observed range "
                f"({channel['min']:.1f}–{channel['max']:.1f})."
                for channel in stats["channels"]
                if not channel["min"] <= inputs[channel["name"]] <= channel["max"]]
    average = stats["average_sales"]
    return {"total_budget": sum(inputs.values()), "average_sales": average,
            "difference": prediction - average,
            "percent_difference": (prediction - average) / average * 100 if average else None,
            "warnings": warnings}
