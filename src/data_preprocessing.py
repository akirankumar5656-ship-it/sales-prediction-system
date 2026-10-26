"""
src/data_preprocessing.py

Data preprocessing module for the Sales Prediction System.

This module is written to be dataset-agnostic: point CONFIG at the actual
column names in your sales.csv (Superstore-style or Rossmann-style) and the
pipeline will adapt. Every step is a standalone function so it can be
unit-tested, reused in a Jupyter notebook, or called from the Flask app /
training pipeline independently.

Run directly:
    python src/data_preprocessing.py
"""

import os
import sys
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.file_utils import read_csv_robust
except ImportError:
    from file_utils import read_csv_robust

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
# Edit this block to match your actual sales.csv column names. Nothing else
# in the file needs to change if your schema differs from the default
# (Superstore-style) assumption below.

CONFIG: Dict = {
    "raw_data_path": "data/raw/sales.csv",
    "processed_data_path": "data/processed/sales_processed.csv",

    # Column(s) that hold dates. First one found in the dataframe is used.
    "date_columns": ["Order Date", "Date"],

    # Target variable you are ultimately predicting.
    "target_column": "Sales",

    # Categorical columns to label-encode. Only columns that actually exist
    # in the loaded dataframe are processed — extras are ignored safely.
    "categorical_columns": [
        "Category", "Sub-Category", "Region", "Segment",
        "Ship Mode", "State", "City", "Store Type", "Assortment",
    ],

    # Columns that should never be negative (used for invalid-value checks).
    "non_negative_columns": ["Sales", "Quantity", "Profit", "Discount"],

    # Columns dropped outright before processing (IDs / free text that add
    # no predictive value and would otherwise choke the encoder).
    "drop_columns": ["Row ID", "Order ID", "Customer ID", "Customer Name",
                      "Product ID", "Product Name", "Postal Code"],

    "label_encoders_path": "models/label_encoders.pkl",
}

# Resolve defaults relative to this project, independent of the launch directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _key, _value in CONFIG.items():
    if isinstance(_value, str) and _value.startswith(("data/", "models/", "reports/")):
        CONFIG[_key] = os.path.join(_PROJECT_ROOT, _value)
if "raw_data_path" in CONFIG and not os.path.exists(CONFIG["raw_data_path"]):
    CONFIG["raw_data_path"] = os.path.join(_PROJECT_ROOT, "data/raw/sales data file.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. LOAD
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the raw sales CSV into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Place your dataset there or update "
            f"CONFIG['raw_data_path']."
        )
    df = read_csv_robust(path)
    logger.info("Loaded '%s' -> shape %s", path, df.shape)
    return df


# --------------------------------------------------------------------------
# 2. INSPECT
# --------------------------------------------------------------------------
def display_shape(df: pd.DataFrame) -> None:
    """Print row/column counts."""
    print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")


def display_dtypes(df: pd.DataFrame) -> None:
    """Print the dtype of every column."""
    print("\nColumn data types:")
    print(df.dtypes)


# --------------------------------------------------------------------------
# 3. MISSING VALUES
# --------------------------------------------------------------------------
def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table of missing values per column
    (count + percentage), sorted descending. Prints it too.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct.round(2),
    })
    summary = summary[summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )

    print("\nMissing value summary:")
    print(summary if not summary.empty else "No missing values detected.")
    return summary


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill/drop missing values with column-type-appropriate strategies:
      - Numeric columns  -> median (robust to outliers, unlike mean)
      - Categorical/text -> mode, or 'Unknown' if no mode exists
      - Date columns     -> rows with missing dates are dropped, since a
                             sales record with no date is not usable for a
                             time-based forecasting model
      - Rows missing the target variable are dropped (can't train on them)
    """
    df = df.copy()

    # Drop rows with a missing target — we can't learn from unlabeled rows.
    target = CONFIG["target_column"]
    if target in df.columns:
        before = len(df)
        df = df.dropna(subset=[target])
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d rows with missing target '%s'", dropped, target)

    date_cols = [c for c in CONFIG["date_columns"] if c in df.columns]
    for col in date_cols:
        before = len(df)
        df = df.dropna(subset=[col])
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d rows with missing date column '%s'", dropped, col)

    for col in df.columns:
        if col in date_cols or col == target:
            continue
        if df[col].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            logger.info("Filled numeric column '%s' missing values with median=%s", col, fill_value)
        else:
            mode_series = df[col].mode()
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            logger.info("Filled categorical column '%s' missing values with mode='%s'", col, fill_value)

    return df


# --------------------------------------------------------------------------
# 4. DUPLICATES
# --------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicated rows and report how many were removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    logger.info("Removed %d duplicate rows (%d -> %d)", removed, before, len(df))
    return df


# --------------------------------------------------------------------------
# 5. INVALID VALUES
# --------------------------------------------------------------------------
def detect_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag and remove obviously invalid rows:
      - Negative values in columns that should never be negative
        (Sales, Quantity, Profit magnitude, Discount)
      - Discount outside the [0, 1] range, if a Discount column exists
      - Future-dated rows (date later than today), if a date column exists
    Returns the cleaned dataframe; logs how many rows were removed per rule.
    """
    df = df.copy()

    for col in CONFIG["non_negative_columns"]:
        if col not in df.columns or col == "Profit":
            # Profit CAN legitimately be negative (a loss), so it's excluded
            # from the "must be non-negative" rule but still numeric-checked
            # elsewhere via missing-value handling.
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            invalid_mask = df[col] < 0
            n_invalid = invalid_mask.sum()
            if n_invalid:
                logger.info("Removing %d rows with negative '%s'", n_invalid, col)
                df = df[~invalid_mask]

    if "Discount" in df.columns and pd.api.types.is_numeric_dtype(df["Discount"]):
        invalid_mask = (df["Discount"] < 0) | (df["Discount"] > 1)
        n_invalid = invalid_mask.sum()
        if n_invalid:
            logger.info("Removing %d rows with Discount outside [0, 1]", n_invalid)
            df = df[~invalid_mask]

    date_cols = [c for c in CONFIG["date_columns"] if c in df.columns]
    for col in date_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            future_mask = df[col] > pd.Timestamp.now()
            n_future = future_mask.sum()
            if n_future:
                logger.info("Removing %d rows with future date in '%s'", n_future, col)
                df = df[~future_mask]

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 6. DATE CONVERSION
# --------------------------------------------------------------------------
def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert configured date columns to proper datetime dtype, then engineer
    standard date-part features used throughout the ML workflow (year,
    month, day, day_of_week, is_weekend, quarter). The original date
    column is kept for reference/sorting.
    """
    df = df.copy()
    date_cols = [c for c in CONFIG["date_columns"] if c in df.columns]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        prefix = col.lower().replace(" ", "_")

        df[f"{prefix}_year"] = df[col].dt.year
        df[f"{prefix}_month"] = df[col].dt.month
        df[f"{prefix}_day"] = df[col].dt.day
        df[f"{prefix}_day_of_week"] = df[col].dt.dayofweek       # 0=Mon
        df[f"{prefix}_is_weekend"] = (df[f"{prefix}_day_of_week"] >= 5).astype(int)
        df[f"{prefix}_quarter"] = df[col].dt.quarter

        logger.info("Converted '%s' to datetime and engineered date-part features", col)

    return df


# --------------------------------------------------------------------------
# 7. CATEGORICAL ENCODING
# --------------------------------------------------------------------------
def encode_categorical_variables(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """
    Label-encode configured categorical columns in place, keeping the
    original column (renamed with an `_encoded` suffix for the numeric
    version) so both the human-readable and model-ready forms are
    available downstream (e.g. for the dashboard vs. the model).

    Returns the fitted encoders (one per column) so the caller can persist
    them. This is what src/predict.py loads later — reusing the SAME
    fitted LabelEncoder is the only way a new value like "Furniture" is
    guaranteed to map to the exact same code it had during training.
    Fitting a fresh LabelEncoder at prediction time would not give this
    guarantee (encoding order depends on what data it sees).
    """
    cols_to_encode = [c for c in CONFIG["categorical_columns"] if c in df.columns]
    encoders: Dict[str, LabelEncoder] = {}

    for col in cols_to_encode:
        encoder = LabelEncoder()
        df[f"{col}_encoded"] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
        logger.info("Encoded '%s' -> '%s_encoded' (%d classes)",
                    col, col, len(encoder.classes_))

    return encoders


def save_label_encoders(encoders: Dict[str, LabelEncoder], path: str) -> None:
    """Persist fitted LabelEncoders with joblib so predict.py can reuse them."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoders, path)
    logger.info("Saved %d label encoders -> '%s'", len(encoders), path)


# --------------------------------------------------------------------------
# 8. DROP UNUSED COLUMNS
# --------------------------------------------------------------------------
def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID/free-text columns configured in CONFIG['drop_columns']."""
    cols_to_drop = [c for c in CONFIG["drop_columns"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped unused columns: %s", cols_to_drop)
    return df


# --------------------------------------------------------------------------
# 9. SAVE
# --------------------------------------------------------------------------
def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """Save the processed dataframe to CSV, creating folders if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved processed dataset -> '%s' (shape %s)", path, df.shape)


# --------------------------------------------------------------------------
# PIPELINE
# --------------------------------------------------------------------------
def run_pipeline(
    raw_path: Optional[str] = None,
    processed_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline end to end and return the
    resulting dataframe. This is the single entry point the training
    script / Flask app should import and call.
    """
    raw_path = raw_path or CONFIG["raw_data_path"]
    processed_path = processed_path or CONFIG["processed_data_path"]

    df = load_data(raw_path)

    if CONFIG["target_column"] not in df.columns:
        raise ValueError("Dataset must contain a Sales target column. The car purchasing dataset is not a sales dataset.")
    if df.empty:
        raise ValueError("The sales dataset is empty.")
    display_shape(df)
    display_dtypes(df)
    detect_missing_values(df)

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = convert_date_columns(df)
    df = detect_invalid_values(df)
    encoders = encode_categorical_variables(df)   # mutates df in place, returns fitted encoders
    save_label_encoders(encoders, CONFIG["label_encoders_path"])
    df = drop_unused_columns(df)

    print("\nFinal processed dataset shape:", df.shape)
    save_processed_data(df, processed_path)

    return df


if __name__ == "__main__":
    run_pipeline()
