"""
src/eda.py

Exploratory Data Analysis module for the Sales Prediction System.

Each analysis is a standalone function that takes a dataframe, draws a
plot, and saves it to reports/figures/. This lets the same functions be
called from a script, from the notebook (notebooks/01_eda.ipynb), or later
from a Flask "analytics" route that regenerates charts on demand.

All functions check whether their required column(s) exist before running,
so the module degrades gracefully instead of crashing if your sales.csv
uses different column names than the defaults in CONFIG.

Run directly:
    python src/eda.py
"""

import os
import sys
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless/script execution; notebook overrides this
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.file_utils import read_csv_robust
except ImportError:
    from file_utils import read_csv_robust

# --------------------------------------------------------------------------
# CONFIG — keep column names in sync with src/data_preprocessing.py
# --------------------------------------------------------------------------
CONFIG = {
    # EDA reads from the RAW file, not the encoded/processed one, because
    # human-readable labels (product names, category names) are needed for
    # meaningful charts. The processed CSV drops those in favor of numeric
    # encodings meant for the model.
    "data_path": "data/raw/sales.csv",
    "figures_dir": "reports/figures",

    "date_column": "Order Date",
    "target_column": "Sales",          # what we're predicting
    "quantity_column": "Quantity",
    "product_column": "Product Name",
    "category_column": "Category",
    "region_column": "Region",

    "top_n_products": 10,
    "fig_dpi": 150,
}

# Resolve defaults relative to this project, independent of the launch directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _key, _value in CONFIG.items():
    if isinstance(_value, str) and _value.startswith(("data/", "models/", "reports/")):
        CONFIG[_key] = os.path.join(_PROJECT_ROOT, _value)
if not os.path.exists(CONFIG["data_path"]):
    CONFIG["data_path"] = os.path.join(_PROJECT_ROOT, "data/raw/sales data file.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the dataset for EDA and parse the date column if present."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Update CONFIG['data_path'] or place "
            f"your dataset there."
        )
    df = read_csv_robust(path)
    date_col = CONFIG["date_column"]
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    logger.info("Loaded '%s' -> shape %s", path, df.shape)
    return df


def _save_fig(fig, filename: str) -> str:
    """Save a matplotlib figure to CONFIG['figures_dir'] and return the path."""
    os.makedirs(CONFIG["figures_dir"], exist_ok=True)
    path = os.path.join(CONFIG["figures_dir"], filename)
    fig.savefig(path, dpi=CONFIG["fig_dpi"], bbox_inches="tight")
    logger.info("Saved figure -> '%s'", path)
    return path


def _has_columns(df: pd.DataFrame, *cols) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning("Skipping plot — missing column(s): %s", missing)
        return False
    return True


# --------------------------------------------------------------------------
# 1. SALES DISTRIBUTION
# --------------------------------------------------------------------------
def plot_sales_distribution(df: pd.DataFrame) -> Optional[str]:
    """Histogram + KDE of the target variable to check skew/outliers."""
    target = CONFIG["target_column"]
    if not _has_columns(df, target):
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[target], kde=True, bins=40, color="steelblue", ax=ax)
    ax.set_title("Sales Distribution")
    ax.set_xlabel(target)
    ax.set_ylabel("Frequency")
    return _save_fig(fig, "01_sales_distribution.png")


# --------------------------------------------------------------------------
# 2. SALES OVER TIME
# --------------------------------------------------------------------------
def plot_sales_over_time(df: pd.DataFrame) -> Optional[str]:
    """Daily/monthly aggregated sales trend line."""
    date_col, target = CONFIG["date_column"], CONFIG["target_column"]
    if not _has_columns(df, date_col, target):
        return None

    ts = (
        df.dropna(subset=[date_col])
        .set_index(date_col)[target]
        .resample("ME")   # monthly totals — daily is usually too noisy to read
        .sum()
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ts.plot(ax=ax, color="darkorange", marker="o")
    ax.set_title("Total Sales Over Time (Monthly)")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Total {target}")
    return _save_fig(fig, "02_sales_over_time.png")


# --------------------------------------------------------------------------
# 3. SALES BY PRODUCT
# --------------------------------------------------------------------------
def plot_sales_by_product(df: pd.DataFrame) -> Optional[str]:
    """Top-N products by total sales (horizontal bar — full product list is unreadable)."""
    product_col, target = CONFIG["product_column"], CONFIG["target_column"]
    if not _has_columns(df, product_col, target):
        return None

    top_n = CONFIG["top_n_products"]
    top_products = (
        df.groupby(product_col)[target].sum().sort_values(ascending=False).head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_products.values, y=top_products.index, palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Products by Total Sales")
    ax.set_xlabel(f"Total {target}")
    ax.set_ylabel("Product")
    return _save_fig(fig, "03_sales_by_product.png")


# --------------------------------------------------------------------------
# 4. SALES BY CATEGORY
# --------------------------------------------------------------------------
def plot_sales_by_category(df: pd.DataFrame) -> Optional[str]:
    """Total sales grouped by product category."""
    category_col, target = CONFIG["category_column"], CONFIG["target_column"]
    if not _has_columns(df, category_col, target):
        return None

    by_category = df.groupby(category_col)[target].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=by_category.index, y=by_category.values, palette="mako", ax=ax)
    ax.set_title("Total Sales by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel(f"Total {target}")
    ax.tick_params(axis="x", rotation=30)
    return _save_fig(fig, "04_sales_by_category.png")


# --------------------------------------------------------------------------
# 5. SALES BY REGION
# --------------------------------------------------------------------------
def plot_sales_by_region(df: pd.DataFrame) -> Optional[str]:
    """Total sales grouped by region."""
    region_col, target = CONFIG["region_column"], CONFIG["target_column"]
    if not _has_columns(df, region_col, target):
        return None

    by_region = df.groupby(region_col)[target].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=by_region.index, y=by_region.values, palette="crest", ax=ax)
    ax.set_title("Total Sales by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel(f"Total {target}")
    return _save_fig(fig, "05_sales_by_region.png")


# --------------------------------------------------------------------------
# 6. CORRELATION HEATMAP
# --------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame) -> Optional[str]:
    """Correlation matrix across all numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        logger.warning("Skipping correlation heatmap — fewer than 2 numeric columns")
        return None

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    return _save_fig(fig, "06_correlation_heatmap.png")


# --------------------------------------------------------------------------
# 7. QUANTITY VS SALES
# --------------------------------------------------------------------------
def plot_quantity_vs_sales(df: pd.DataFrame) -> Optional[str]:
    """Scatter plot with regression line: does volume drive revenue?"""
    qty_col, target = CONFIG["quantity_column"], CONFIG["target_column"]
    if not _has_columns(df, qty_col, target):
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(x=df[qty_col], y=df[target], scatter_kws={"alpha": 0.4, "s": 15},
                line_kws={"color": "red"}, ax=ax)
    ax.set_title("Quantity vs Sales")
    ax.set_xlabel(qty_col)
    ax.set_ylabel(target)
    return _save_fig(fig, "07_quantity_vs_sales.png")


# --------------------------------------------------------------------------
# 8. PRICE VS SALES
# --------------------------------------------------------------------------
def plot_price_vs_sales(df: pd.DataFrame) -> Optional[str]:
    """
    Scatter plot of unit price vs sales. If no explicit price column
    exists (common in datasets like Superstore), it's derived as
    Sales / Quantity for this plot only.
    """
    target, qty_col = CONFIG["target_column"], CONFIG["quantity_column"]

    if "Price" in df.columns:
        price = df["Price"]
    elif _has_columns(df, target, qty_col):
        with np.errstate(divide="ignore", invalid="ignore"):
            price = df[target] / df[qty_col].replace(0, np.nan)
    else:
        logger.warning("Skipping price vs sales — no 'Price' column and cannot derive one")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=price, y=df[target], alpha=0.4, s=15, ax=ax)
    ax.set_title("Price vs Sales")
    ax.set_xlabel("Price (derived: Sales / Quantity)" if "Price" not in df.columns else "Price")
    ax.set_ylabel(target)
    return _save_fig(fig, "08_price_vs_sales.png")


# --------------------------------------------------------------------------
# PIPELINE
# --------------------------------------------------------------------------
def run_eda(data_path: Optional[str] = None) -> dict:
    """
    Run every EDA plot in sequence and return a dict of {name: saved_path}
    for whichever plots successfully ran (entries are skipped, not None,
    if the required columns were missing).
    """
    df = load_data(data_path or CONFIG["data_path"])

    steps = {
        "sales_distribution": plot_sales_distribution,
        "sales_over_time": plot_sales_over_time,
        "sales_by_product": plot_sales_by_product,
        "sales_by_category": plot_sales_by_category,
        "sales_by_region": plot_sales_by_region,
        "correlation_heatmap": plot_correlation_heatmap,
        "quantity_vs_sales": plot_quantity_vs_sales,
        "price_vs_sales": plot_price_vs_sales,
    }

    results = {}
    for name, func in steps.items():
        path = func(df)
        if path:
            results[name] = path
        plt.close("all")  # free memory between plots

    logger.info("EDA complete: %d/%d plots generated", len(results), len(steps))
    return results


if __name__ == "__main__":
    run_eda()
