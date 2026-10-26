"""
src/feature_engineering.py

Feature engineering module for the Sales Prediction System.

Takes the cleaned dataset produced by src/data_preprocessing.py and builds
the feature set the models in ml/train_model.py will actually be trained
on: date-part features, numeric derived features, categorical encodings,
and time-aware lag / rolling-window features.

LEAKAGE POLICY (read this before changing anything below):
  1. Lag and rolling features are computed per group, sorted chronologically,
     and the rolling window is always taken on a series that has already
     been shifted forward by at least 1 period. This guarantees a feature
     for row t only ever uses information from rows strictly before t.
  2. No target-encoding / mean-encoding of categorical variables is done
     here. Computing a category's average Sales across the WHOLE dataset
     and using it as a feature leaks test-set information into training
     rows (the average itself was partly computed from future/test data).
     If you want target encoding later, it must be fit on the training
     fold only, inside the train/test split — not in this shared module.
  3. Rows at the start of each group's time series will have NaN lag/
     rolling values (no history yet exists for them). These are dropped
     rather than filled, because filling them with e.g. 0 or the global
     mean would inject information that did not genuinely exist at that
     point in time.
  4. The train/test split downstream MUST be time-based (train on earlier
     dates, test on later dates), never a random shuffle-split — a random
     split would let the model train on lag features that were computed
     using rows chronologically after the row being predicted.

Run directly:
    python src/feature_engineering.py
"""

import os
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG — keep column names in sync with src/data_preprocessing.py
# --------------------------------------------------------------------------
CONFIG = {
    "input_path": "data/processed/sales_processed.csv",   # output of data_preprocessing.py
    "output_path": "data/processed/sales_features.csv",

    "date_column": "Order Date",
    "target_column": "Sales",
    "quantity_column": "Quantity",
    "discount_column": "Discount",
    "profit_column": "Profit",

    # Series each set of lag/rolling features is computed within. Using
    # Category keeps each product category's own time series separate so
    # a lag feature for "Furniture" never leaks information from
    # "Technology". Change to a Product/Store identifier if your dataset
    # has one and you want item-level lags instead of category-level.
    "group_column": "Category",

    # Low-cardinality columns to one-hot encode. High-cardinality columns
    # (e.g. hundreds of product names) should stay label-encoded — the
    # *_encoded columns already produced by data_preprocessing.py are fine
    # as-is and are left untouched here.
    "onehot_columns": ["Segment", "Ship Mode"],

    "lag_periods": [1, 7, 30],
    "rolling_windows": [7, 30],
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


# --------------------------------------------------------------------------
# LOAD
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the preprocessed dataset and re-parse the date column."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Run src/data_preprocessing.py first, "
            f"or update CONFIG['input_path']."
        )
    df = pd.read_csv(path)
    date_col = CONFIG["date_column"]
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    logger.info("Loaded '%s' -> shape %s", path, df.shape)
    return df


# --------------------------------------------------------------------------
# 1. DATE FEATURES
# --------------------------------------------------------------------------
def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the date column, create: Year, Month, Day, Day_of_week,
    Week_of_year, Quarter — plus a couple of standard extras
    (Is_weekend, Is_month_start, Is_month_end) that are cheap to add and
    commonly useful for sales seasonality.
    """
    df = df.copy()
    date_col = CONFIG["date_column"]
    if date_col not in df.columns:
        logger.warning("Date column '%s' not found — skipping date features", date_col)
        return df

    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    df["Day_of_week"] = df[date_col].dt.dayofweek          # Monday=0 ... Sunday=6
    df["Week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["Quarter"] = df[date_col].dt.quarter

    df["Is_weekend"] = (df["Day_of_week"] >= 5).astype(int)
    df["Is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["Is_month_end"] = df[date_col].dt.is_month_end.astype(int)

    logger.info("Created date features: Year, Month, Day, Day_of_week, "
                "Week_of_year, Quarter, Is_weekend, Is_month_start, Is_month_end")
    return df


# --------------------------------------------------------------------------
# 2. NUMERICAL FEATURES
# --------------------------------------------------------------------------
def create_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived numeric features that use only same-row information (so no
    leakage risk) — unit price, profit margin, discount amount, and a
    monotonic time-trend feature the model can use to learn long-run
    growth/decline independent of seasonality.
    """
    df = df.copy()
    target = CONFIG["target_column"]
    qty_col = CONFIG["quantity_column"]
    discount_col = CONFIG["discount_column"]
    profit_col = CONFIG["profit_column"]
    date_col = CONFIG["date_column"]

    if target in df.columns and qty_col in df.columns:
        df["Price"] = df[target] / df[qty_col].replace(0, np.nan)
        df["Price"] = df["Price"].fillna(df["Price"].median())
        logger.info("Created 'Price' = %s / %s", target, qty_col)

    if profit_col in df.columns and target in df.columns:
        df["Profit_margin"] = df[profit_col] / df[target].replace(0, np.nan)
        df["Profit_margin"] = df["Profit_margin"].fillna(0)
        logger.info("Created 'Profit_margin' = %s / %s", profit_col, target)

    if discount_col in df.columns and target in df.columns:
        # Discount is stored as a fraction (0-1) upstream; amount is in
        # currency units, useful alongside the raw fraction.
        df["Discount_amount"] = df[discount_col] * df[target]
        logger.info("Created 'Discount_amount' = %s * %s", discount_col, target)

    if date_col in df.columns:
        # Days since the earliest date in the dataset — a simple numeric
        # trend feature. Safe: it's derived purely from the row's own date,
        # not from any target value.
        min_date = df[date_col].min()
        df["Days_since_start"] = (df[date_col] - min_date).dt.days
        logger.info("Created 'Days_since_start' trend feature")

    return df


# --------------------------------------------------------------------------
# 3. CATEGORICAL FEATURES
# --------------------------------------------------------------------------
def create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode configured low-cardinality categorical columns.
    High-cardinality columns should rely on the *_encoded label encoding
    already produced by data_preprocessing.py rather than one-hot (which
    would blow up column count).
    """
    df = df.copy()
    cols = [c for c in CONFIG["onehot_columns"] if c in df.columns]
    if not cols:
        logger.info("No configured one-hot columns present — skipping")
        return df

    df = pd.get_dummies(df, columns=cols, prefix=cols, drop_first=True)
    logger.info("One-hot encoded columns: %s", cols)
    return df


# --------------------------------------------------------------------------
# 4. LAG FEATURES  (leakage-safe: only past rows, per group, sorted by date)
# --------------------------------------------------------------------------
def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag_N columns of the target variable, computed within each
    group (CONFIG['group_column']) and sorted chronologically first, so
    lag_1 for a given row is strictly the previous period's value for
    that same group — never a future value and never another group's data.
    """
    df = df.copy()
    date_col = CONFIG["date_column"]
    target = CONFIG["target_column"]
    group_col = CONFIG["group_column"]

    if date_col not in df.columns or target not in df.columns:
        logger.warning("Missing date/target column — skipping lag features")
        return df

    has_group = group_col in df.columns
    sort_cols = [group_col, date_col] if has_group else [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    grouped = df.groupby(group_col)[target] if has_group else None

    for lag in CONFIG["lag_periods"]:
        col_name = f"lag_{lag}"
        if has_group:
            df[col_name] = grouped.shift(lag)
        else:
            df[col_name] = df[target].shift(lag)
        logger.info("Created '%s'%s", col_name, f" (grouped by {group_col})" if has_group else "")

    return df


# --------------------------------------------------------------------------
# 5. ROLLING FEATURES (leakage-safe: shift(1) BEFORE the rolling window)
# --------------------------------------------------------------------------
def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling mean/std columns of the target, computed on a series
    that has already been shifted by 1 period. Shifting first is what
    prevents the current row's own target value from being included in
    "its own" rolling average — a classic and easy-to-miss leakage bug.
    """
    df = df.copy()
    date_col = CONFIG["date_column"]
    target = CONFIG["target_column"]
    group_col = CONFIG["group_column"]

    if date_col not in df.columns or target not in df.columns:
        logger.warning("Missing date/target column — skipping rolling features")
        return df

    has_group = group_col in df.columns
    # df should already be sorted by create_lag_features; sort again defensively.
    sort_cols = [group_col, date_col] if has_group else [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if has_group:
        shifted = df.groupby(group_col)[target].shift(1)
    else:
        shifted = df[target].shift(1)

    for window in CONFIG["rolling_windows"]:
        mean_col, std_col = f"rolling_mean_{window}", f"rolling_std_{window}"
        if has_group:
            df[mean_col] = shifted.groupby(df[group_col]).transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            df[std_col] = shifted.groupby(df[group_col]).transform(
                lambda s: s.rolling(window, min_periods=1).std()
            )
        else:
            df[mean_col] = shifted.rolling(window, min_periods=1).mean()
            df[std_col] = shifted.rolling(window, min_periods=1).std()
        logger.info("Created '%s' and '%s'", mean_col, std_col)

    return df


# --------------------------------------------------------------------------
# 6. CLEANUP — drop rows with no history yet (start of each series)
# --------------------------------------------------------------------------
def drop_incomplete_history_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where lag/rolling features are NaN because not enough
    history existed yet at that point in the series. This is preferable
    to filling with 0 or a global mean, which would fabricate information
    the model would not genuinely have had access to at prediction time.
    """
    lag_cols = [f"lag_{lag}" for lag in CONFIG["lag_periods"]]
    lag_cols = [c for c in lag_cols if c in df.columns]

    if not lag_cols:
        return df

    before = len(df)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    dropped = before - len(df)
    logger.info("Dropped %d rows with incomplete lag history (%d -> %d)",
                dropped, before, len(df))
    return df


# --------------------------------------------------------------------------
# SAVE
# --------------------------------------------------------------------------
def save_features(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved feature-engineered dataset -> '%s' (shape %s)", path, df.shape)


# --------------------------------------------------------------------------
# PIPELINE
# --------------------------------------------------------------------------
def run_pipeline(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline end to end."""
    input_path = input_path or CONFIG["input_path"]
    output_path = output_path or CONFIG["output_path"]

    df = load_data(input_path)

    df = create_date_features(df)
    df = create_numerical_features(df)
    df = create_categorical_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = drop_incomplete_history_rows(df)

    print("\nFinal feature-engineered dataset shape:", df.shape)
    print("\nColumns:", list(df.columns))

    save_features(df, output_path)
    return df


if __name__ == "__main__":
    run_pipeline()
