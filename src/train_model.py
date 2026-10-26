"""
src/train_model.py

Model training pipeline for the Sales Prediction System.

Loads the feature-engineered dataset from src/feature_engineering.py,
performs a time-based train/test split, trains and compares four
regression models, evaluates each with MAE / MSE / RMSE / R^2, prints a
comparison table, and saves the best-performing model (by RMSE) to disk
with joblib for the Flask app to load at inference time.

WHY A TIME-BASED SPLIT (not train_test_split with shuffle=True):
The feature set includes lag_1/7/30 and rolling_mean/std features, which
are only valid if a row's "past" genuinely occurred before it. A random
shuffle split would let training rows sit chronologically after some test
rows (or vice versa), and — more subtly — since lag/rolling values were
computed using neighbouring rows in time, a random split can put a row and
the very rows its lag features were derived from on opposite sides of the
split, leaking information. Splitting by date (earliest X% = train, latest
(1-X)% = test) mirrors how the model will actually be used: predicting
sales it hasn't seen yet, using only what happened before.

Run directly:
    python src/train_model.py
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split as sk_train_test_split

# --------------------------------------------------------------------------
# CONFIG — keep in sync with src/feature_engineering.py
# --------------------------------------------------------------------------
CONFIG = {
    "input_path": "data/processed/sales_features.csv",
    "date_column": "Order Date",
    "target_column": "Sales",

    # Columns that must never be used as model inputs even though they're
    # numeric-looking or present in the dataframe (IDs, raw text columns
    # that have an *_encoded counterpart, leakage-risk columns derived
    # directly from the target for the SAME row).
    "exclude_columns": [
        "Order Date", "Ship Date", "Sales",   # date raw col + target
        "Profit", "Profit_margin",            # derived from Sales for the same row -> would leak
        # Price and Discount_amount are ALSO derived from Sales
        # (Price = Sales / Quantity, Discount_amount = Discount * Sales).
        # Training on them would let the model reconstruct Sales almost
        # perfectly, inflating test metrics — but at real prediction time
        # Sales is exactly what's unknown, so these can never be computed
        # for a genuinely new input. This is target leakage, not time
        # leakage, and it would silently break src/predict.py if left in.
        "Price", "Discount_amount",
        # The prediction form doesn't collect these, so the model must not
        # depend on them (at inference they'd be constants, not real signal).
        "Sub-Category_encoded", "State_encoded", "City_encoded",
        "Store Type_encoded", "Assortment_encoded",
        "Category", "Sub-Category", "Region", "Segment", "Ship Mode",
        "State", "City", "Store Type", "Assortment",  # raw text (encoded versions used instead)
    ],

    "test_size": 0.2,          # last 20% of the time-ordered data -> test set
    "random_state": 42,

    "model_output_path": "models/sales_model.pkl",
    "feature_list_path": "models/feature_columns.json",
    "comparison_table_path": "reports/metrics/model_comparison.csv",
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
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Run src/feature_engineering.py first, "
            f"or update CONFIG['input_path']."
        )
    df = pd.read_csv(path)
    date_col = CONFIG["date_column"]
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    logger.info("Loaded '%s' -> shape %s", path, df.shape)
    return df


# --------------------------------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------------------------------
def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Build the final list of model input columns: every numeric column
    except the target and anything explicitly excluded in CONFIG. Any
    remaining non-numeric column is dropped with a warning rather than
    silently breaking the model fit.
    """
    exclude = set(CONFIG["exclude_columns"])
    candidate_cols = [c for c in df.columns if c not in exclude]

    numeric_cols, dropped_non_numeric = [], []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)
        else:
            dropped_non_numeric.append(col)

    if dropped_non_numeric:
        logger.warning("Dropping non-numeric columns not covered by CONFIG['exclude_columns']: %s",
                        dropped_non_numeric)

    logger.info("Selected %d feature columns", len(numeric_cols))
    return numeric_cols


# --------------------------------------------------------------------------
# TIME-BASED SPLIT
# --------------------------------------------------------------------------
def time_based_split(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sort by date and take the earliest (1 - test_size) fraction as train,
    the latest test_size fraction as test. See module docstring for why
    this is used instead of a random shuffle split WHEN a date column and
    time-derived features (lags/rolling windows) are actually present.

    Some datasets (e.g. a plain "ad spend -> sales" regression dataset
    with no date/time dimension at all) have no temporal structure to
    split on and no leakage-prone lag features to protect. For those,
    this falls back to a normal random train_test_split — there is no
    "before/after" concept to respect, so a time-based split would just
    be an arbitrary row split with an extra, meaningless sort.
    """
    date_col = CONFIG["date_column"]
    target = CONFIG["target_column"]

    if date_col not in df.columns:
        logger.warning(
            "No '%s' column found in this dataset — there's no time "
            "dimension to split on, so falling back to a random "
            "train_test_split (test_size=%.2f). This is expected and safe "
            "for a non-time-series dataset (e.g. plain feature -> Sales "
            "regression with no lag/rolling features).",
            date_col, CONFIG["test_size"],
        )
        X = df[feature_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = sk_train_test_split(
            X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"]
        )
        logger.info("Random split -> train=%d rows, test=%d rows", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - CONFIG["test_size"]))

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_train, y_test = train_df[target], test_df[target]

    logger.info("Time-based split -> train=%d rows (up to %s), test=%d rows (from %s)",
                len(train_df), train_df[date_col].max(),
                len(test_df), test_df[date_col].min())

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# MODELS
# --------------------------------------------------------------------------
def get_models() -> Dict[str, object]:
    """Instantiate the four candidate regressors with sensible defaults."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=CONFIG["random_state"]),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=CONFIG["random_state"], n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=CONFIG["random_state"]),
    }


# --------------------------------------------------------------------------
# TRAIN + EVALUATE
# --------------------------------------------------------------------------
def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, R^2 for one set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def train_and_compare(
    X_train, X_test, y_train, y_test
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Train every candidate model, evaluate on the held-out test set, and
    return (comparison_table, fitted_models_dict).
    """
    models = get_models()
    fitted_models = {}
    rows = []

    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)
        metrics["Model"] = name
        rows.append(metrics)
        fitted_models[name] = model

        logger.info("%s -> MAE=%.3f  RMSE=%.3f  R2=%.4f",
                    name, metrics["MAE"], metrics["RMSE"], metrics["R2"])

    comparison = pd.DataFrame(rows)[["Model", "MAE", "MSE", "RMSE", "R2"]]
    comparison = comparison.sort_values("RMSE").reset_index(drop=True)
    return comparison, fitted_models


# --------------------------------------------------------------------------
# SAVE
# --------------------------------------------------------------------------
def save_best_model(comparison: pd.DataFrame, fitted_models: Dict[str, object],
                     feature_cols: List[str]) -> str:
    """
    Save the model with the lowest RMSE on the test set, plus the exact
    feature column list/order it expects — the Flask inference code needs
    that list to build a correctly-ordered input row.
    """
    best_name = comparison.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    os.makedirs(os.path.dirname(CONFIG["model_output_path"]), exist_ok=True)
    joblib.dump(best_model, CONFIG["model_output_path"])
    logger.info("Saved best model ('%s') -> '%s'", best_name, CONFIG["model_output_path"])

    os.makedirs(os.path.dirname(CONFIG["feature_list_path"]), exist_ok=True)
    with open(CONFIG["feature_list_path"], "w") as f:
        json.dump({"model_name": best_name, "feature_columns": feature_cols}, f, indent=2)
    logger.info("Saved feature column list -> '%s'", CONFIG["feature_list_path"])

    return best_name


def save_comparison_table(comparison: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(CONFIG["comparison_table_path"]), exist_ok=True)
    comparison.to_csv(CONFIG["comparison_table_path"], index=False)
    logger.info("Saved model comparison table -> '%s'", CONFIG["comparison_table_path"])


# --------------------------------------------------------------------------
# PIPELINE
# --------------------------------------------------------------------------
def run_pipeline(input_path: Optional[str] = None) -> pd.DataFrame:
    """Run the full training pipeline end to end and return the comparison table."""
    df = load_data(input_path or CONFIG["input_path"])
    feature_cols = select_feature_columns(df)

    X_train, X_test, y_train, y_test = time_based_split(df, feature_cols)

    comparison, fitted_models = train_and_compare(X_train, X_test, y_train, y_test)

    print("\nModel Comparison (sorted by RMSE, best first):")
    print(comparison.to_string(index=False))

    best_name = save_best_model(comparison, fitted_models, feature_cols)
    save_comparison_table(comparison)

    print(f"\nBest model: {best_name}")
    return comparison


if __name__ == "__main__":
    run_pipeline()
