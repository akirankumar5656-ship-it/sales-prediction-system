"""
src/predict.py

Prediction / inference module for the Sales Prediction System.

Loads the trained model and reproduces — exactly, not approximately — the
same feature transformations used in training, then returns a predicted
Sales value for new, user-supplied input.

WHY THIS IS HARDER THAN IT LOOKS
A trained regression model expects an input row shaped exactly like the
rows it was trained on. Getting that row is not just "re-run
preprocessing on one row" for three reasons specific to this pipeline:

  1. Categorical encoding must use the SAME fitted LabelEncoder objects
     from training (src/data_preprocessing.py), not a freshly-fit one —
     otherwise "Furniture" could map to a different integer code than it
     did during training, and the model would silently misread the input.
     Fix: data_preprocessing.py now saves the fitted encoders to
     models/label_encoders.pkl; this module loads and reuses them.

  2. One-hot encoded columns (Segment, Ship Mode) only exist because
     pd.get_dummies saw every category across the whole training set. A
     single new row can't reproduce that on its own. Fix: this module
     starts from the exact feature column list saved by train_model.py
     (models/feature_columns.json) and sets the one matching dummy column
     to 1, leaving the rest at 0 — which is mathematically identical to
     what get_dummies would have produced for that row.

  3. Lag / rolling features (lag_1, lag_7, lag_30, rolling_mean/std) are
     NOT computable from the new input alone — they describe what
     happened *before* this prediction date for this category. Fix: this
     module looks up recent historical rows for the requested category
     from the feature-engineered dataset and computes them the same way
     src/feature_engineering.py did (shift-before-rolling, no peeking at
     rows after the prediction date).

Any column in models/feature_columns.json that doesn't fall into one of
the above is filled from the user's direct input (Quantity, Discount).

Run directly for a quick manual test:
    python src/predict.py
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.feature_engineering import create_date_features, CONFIG as FE_CONFIG
except ImportError:
    from feature_engineering import create_date_features, CONFIG as FE_CONFIG

try:
    from src.data_preprocessing import convert_date_columns
except ImportError:
    from data_preprocessing import convert_date_columns

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CONFIG = {
    "model_path": "models/sales_model.pkl",
    "feature_list_path": "models/feature_columns.json",
    "label_encoders_path": "models/label_encoders.pkl",

    # Historical data used to compute lag/rolling features for a new
    # prediction request. Must be the SAME file feature_engineering.py
    # produced, so history is consistent with how the model was trained.
    "history_path": "data/processed/sales_features.csv",

    # Used only to recover the earliest date seen in training, which the
    # Days_since_start feature is measured from.
    "processed_path": "data/processed/sales_processed.csv",

    "date_column": "Order Date",
    "target_column": "Sales",
    "group_column": "Category",       # must match feature_engineering.CONFIG
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
# LOAD ARTIFACTS (model, feature list, encoders, history) — cached module-level
# --------------------------------------------------------------------------
class PredictionService:
    """
    Wraps the trained model plus every artifact needed to reproduce
    training-time preprocessing. Instantiate once (loading is not free)
    and call .predict(user_input) as many times as needed — this is the
    pattern the Flask app should use (load at app startup, not per request).
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or CONFIG
        self.model = self._load_model()
        self.feature_columns = self._load_feature_columns()
        self.label_encoders = self._load_label_encoders()
        self.history = self._load_history()
        self.is_advertising = set(self.feature_columns) == {"TV", "Radio", "Newspaper"}
        self.min_date = None if self.is_advertising else self._load_min_date()

    # ---- artifact loading -------------------------------------------------
    def _load_model(self):
        path = self.config["model_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Trained model not found at '{path}'. Run src/train_model.py first."
            )
        model = joblib.load(path)
        logger.info("Loaded model from '%s'", path)
        return model

    def _load_feature_columns(self):
        path = self.config["feature_list_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Feature column list not found at '{path}'. Run src/train_model.py first."
            )
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded %d expected feature columns (model: %s)",
                    len(data["feature_columns"]), data.get("model_name", "unknown"))
        return data["feature_columns"]

    def _load_label_encoders(self):
        path = self.config["label_encoders_path"]
        if not os.path.exists(path):
            logger.warning("Label encoders not found at '%s' — categorical "
                            "*_encoded features will fall back to -1 (unknown)", path)
            return {}
        encoders = joblib.load(path)
        logger.info("Loaded %d label encoders", len(encoders))
        return encoders

    def _load_history(self) -> Optional[pd.DataFrame]:
        path = self.config["history_path"]
        if not os.path.exists(path):
            logger.warning("History file not found at '%s' — lag/rolling features "
                            "will be filled with 0 (equivalent to 'no known history')", path)
            return None
        df = pd.read_csv(path)
        date_col = self.config["date_column"]
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        logger.info("Loaded history -> shape %s", df.shape)
        return df

    def _load_min_date(self):
        """Earliest order date in the training data (Days_since_start's origin)."""
        path = self.config["processed_path"]
        date_col = self.config["date_column"]
        if not os.path.exists(path):
            logger.warning("'%s' not found — Days_since_start cannot be reproduced", path)
            return None
        dates = pd.read_csv(path, usecols=[date_col], parse_dates=[date_col])[date_col]
        return dates.min()

    def _mode_from_history(self, col: str):
        """Most common value of a raw column in the training history (or None)."""
        if self.history is not None and col in self.history.columns:
            mode = self.history[col].mode()
            if not mode.empty:
                return str(mode.iloc[0])
        return None

    # ---- feature construction ---------------------------------------------
    def _encode_categoricals(self, row: Dict, output: Dict) -> None:
        """
        Apply the SAME fitted LabelEncoders used in training. Unseen
        category values (not present in the training data) are logged
        and mapped to -1 rather than raising, so a single bad/new input
        value degrades gracefully instead of crashing the whole request.
        """
        for col, encoder in self.label_encoders.items():
            target_feature = f"{col}_encoded"
            if target_feature not in self.feature_columns:
                continue
            raw_value = row.get(col)
            if raw_value is None or raw_value == "":
                # Field not collected by the form (e.g. State, City): use the
                # most common training value instead of an invalid -1 code.
                raw_value = self._mode_from_history(col)
                if raw_value is None:
                    logger.warning("No value or history for '%s' — encoding as -1", col)
                    output[target_feature] = -1
                    continue
                logger.info("'%s' not supplied — using most common training value '%s'", col, raw_value)
            raw_value = str(raw_value)
            if raw_value in encoder.classes_:
                output[target_feature] = encoder.transform([raw_value])[0]
            else:
                logger.warning("Unseen value '%s' for column '%s' — encoding as -1", raw_value, col)
                output[target_feature] = -1

    def _encode_onehot(self, row: Dict, output: Dict) -> None:
        """
        Reconstruct one-hot columns by matching '<column>_<value>' against
        the known feature column list, rather than calling pd.get_dummies
        on a single row (which cannot reproduce the training-time set of
        dummy columns on its own). If the row's value was the baseline
        category dropped by drop_first=True during training, every
        matching dummy correctly stays 0 — no action needed.
        """
        for col in FE_CONFIG.get("onehot_columns", []):
            raw_value = row.get(col)
            if raw_value is None:
                continue
            dummy_col = f"{col}_{raw_value}"
            if dummy_col in self.feature_columns:
                output[dummy_col] = 1

    def _compute_date_features(self, row: Dict, output: Dict) -> None:
        """
        Reuse the exact functions used in training so nothing drifts:
        convert_date_columns (order_date_year, ... from data_preprocessing)
        and create_date_features (Year, Month, ... from feature_engineering),
        plus Days_since_start measured from the training data's first date.
        """
        date_col = self.config["date_column"]
        tmp = pd.DataFrame({date_col: [pd.to_datetime(row[date_col])]})
        tmp = convert_date_columns(tmp)
        tmp = create_date_features(tmp)
        for col in tmp.columns:
            if col != date_col and col in self.feature_columns:
                output[col] = tmp.iloc[0][col]

        if "Days_since_start" in self.feature_columns and self.min_date is not None:
            output["Days_since_start"] = (tmp.iloc[0][date_col] - self.min_date).days

    def _compute_lag_rolling_features(self, row: Dict, output: Dict) -> None:
        """
        Look up this category's historical Sales values strictly BEFORE
        the requested prediction date, and compute lag_N / rolling_mean_N
        / rolling_std_N exactly as feature_engineering.py did (shift-based,
        no use of same-day or future data).
        """
        target = self.config["target_column"]
        date_col = self.config["date_column"]
        group_col = self.config["group_column"]
        prediction_date = pd.to_datetime(row[date_col])

        if self.history is None:
            for lag in self.config["lag_periods"]:
                output.setdefault(f"lag_{lag}", 0)
            for window in self.config["rolling_windows"]:
                output.setdefault(f"rolling_mean_{window}", 0)
                output.setdefault(f"rolling_std_{window}", 0)
            return

        group_value = row.get(group_col)
        subset = self.history
        if group_col in subset.columns and group_value is not None:
            subset = subset[subset[group_col] == group_value]

        subset = subset[subset[date_col] < prediction_date].sort_values(date_col)

        if subset.empty:
            logger.warning("No history found for %s='%s' before %s — lag/rolling "
                            "features default to 0 (cold start)", group_col, group_value, prediction_date.date())
            recent_sales = pd.Series(dtype=float)
        else:
            recent_sales = subset[target]

        for lag in self.config["lag_periods"]:
            col_name = f"lag_{lag}"
            if col_name in self.feature_columns:
                output[col_name] = recent_sales.iloc[-lag] if len(recent_sales) >= lag else 0

        for window in self.config["rolling_windows"]:
            mean_col, std_col = f"rolling_mean_{window}", f"rolling_std_{window}"
            window_slice = recent_sales.tail(window)
            if mean_col in self.feature_columns:
                output[mean_col] = window_slice.mean() if not window_slice.empty else 0
            if std_col in self.feature_columns:
                output[std_col] = window_slice.std() if len(window_slice) > 1 else 0

    def _direct_numeric_fields(self, row: Dict, output: Dict) -> None:
        """Copy simple user-supplied numeric fields straight through when the model expects them."""
        for field in ["Quantity", "Discount"]:
            if field in self.feature_columns and field in row:
                output[field] = row[field]

    # ---- public API ---------------------------------------------------
    def build_feature_row(self, user_input: Dict) -> pd.DataFrame:
        """
        Transform raw user input into a single-row DataFrame with columns
        in exactly the order the model expects, applying every
        transformation used at training time.
        """
        if self.is_advertising:
            values = {}
            for field in self.feature_columns:
                try:
                    value = float(user_input[field])
                except (KeyError, TypeError, ValueError):
                    raise ValueError(f"{field} must be a number.") from None
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{field} must be a finite, non-negative number.")
                values[field] = value
            return pd.DataFrame([values], columns=self.feature_columns)
        required = [self.config["date_column"], self.config["group_column"],
                    "Quantity", "Discount"]
        missing = [f for f in required if f not in user_input]
        if missing:
            raise ValueError(f"Missing required input field(s): {missing}")

        # Start every expected column at 0 — this is what correctly
        # handles one-hot columns whose category wasn't provided (or was
        # the training-time baseline category).
        output = {col: 0 for col in self.feature_columns}

        self._compute_date_features(user_input, output)
        self._direct_numeric_fields(user_input, output)
        self._encode_categoricals(user_input, output)
        self._encode_onehot(user_input, output)
        self._compute_lag_rolling_features(user_input, output)

        row = pd.DataFrame([output])[self.feature_columns]  # enforce exact training column order
        return row

    def predict(self, user_input: Dict) -> float:
        """
        Accept a dict of raw user input, e.g.:
            {
                "Order Date": "2019-08-15",
                "Category": "Furniture",
                "Region": "West",
                "Segment": "Consumer",
                "Ship Mode": "Second Class",
                "Quantity": 3,
                "Discount": 0.1,
            }
        and return the predicted Sales value as a float.
        """
        feature_row = self.build_feature_row(user_input)
        prediction = self.model.predict(feature_row)[0]
        if not np.isfinite(prediction):
            raise ValueError("The model returned a non-finite prediction.")
        predicted_sales = float(max(prediction, 0))  # sales can't be negative
        logger.info("Predicted sales for input %s -> %.2f", user_input, predicted_sales)
        return predicted_sales


# --------------------------------------------------------------------------
# MODULE-LEVEL CONVENIENCE (what Flask will import)
# --------------------------------------------------------------------------
_service: Optional[PredictionService] = None


def get_service() -> PredictionService:
    """Lazily instantiate a single shared PredictionService (loaded once)."""
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


def predict_sales(user_input: Dict) -> float:
    """Convenience function: app.py can just `from src.predict import predict_sales`."""
    return get_service().predict(user_input)


if __name__ == "__main__":
    sample_input = {
        "Order Date": datetime.today().strftime("%Y-%m-%d"),
        "Category": "Furniture",
        "Region": "West",
        "Segment": "Consumer",
        "Ship Mode": "Second Class",
        "Quantity": 3,
        "Discount": 0.1,
    }
    result = predict_sales(sample_input)
    print(f"\nPredicted Sales: {result:.2f}")
