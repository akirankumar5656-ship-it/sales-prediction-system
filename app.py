"""Flask UI for trained sales models, accounts, analytics, and prediction history.

The supplied advertising model uses TV, Radio, and Newspaper inputs. Retail
models retain the order-entry form. The app never retrains models on requests.
"""

import os
import sys
import math
import json
import csv
import io
from datetime import datetime

from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for, session, g, Response
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from database import db_setup, db_operations
from src.predict import PredictionService
from src import analytics

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Initialize DB schema on startup (safe to call repeatedly — CREATE TABLE IF NOT EXISTS).
db_setup.init_db()

# Load the trained model / encoders / feature list ONCE at startup, not per
# request — model loading involves disk I/O and should not happen on every
# form submission.
try:
    prediction_service = PredictionService()
    MODEL_LOAD_ERROR = None
except Exception as exc:  # noqa: BLE001 - surfaced to the UI deliberately
    prediction_service = None
    MODEL_LOAD_ERROR = str(exc)


@app.before_request
def load_logged_in_user():
    """Make the current user (if any) available as g.user on every request."""
    user_id = session.get("user_id")
    g.user = db_operations.get_user_by_id(user_id) if user_id else None


# --------------------------------------------------------------------------
# AUTH
# --------------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip() or None
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")

    if not username or not password:
        flash("Username and password are required.", "error")
        return redirect(url_for("register"))
    if password != confirm:
        flash("Passwords do not match.", "error")
        return redirect(url_for("register"))

    # Passwords are hashed inside db_operations.create_user (werkzeug's
    # generate_password_hash) — the plain-text password never reaches SQLite.
    user_id = db_operations.create_user(username, password, email)
    if user_id is None:
        flash("That username or email is already taken.", "error")
        return redirect(url_for("register"))

    session["user_id"] = user_id
    flash("Account created — you're logged in.", "success")
    return redirect(url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    user = db_operations.verify_user(username, password)
    if user is None:
        flash("Invalid username or password.", "error")
        return redirect(url_for("login"))

    session["user_id"] = user["id"]
    flash(f"Welcome back, {user['username']}.", "success")
    return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You've been logged out.", "success")
    return redirect(url_for("home"))


# --------------------------------------------------------------------------
# HOME
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", model_ready=prediction_service is not None)


# --------------------------------------------------------------------------
# PREDICTION FORM
# --------------------------------------------------------------------------
@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template(
        "predict.html",
        advertising=prediction_service is not None and prediction_service.is_advertising,
        categories=config.CATEGORY_CHOICES,
        regions=config.REGION_CHOICES,
        segments=config.SEGMENT_CHOICES,
        ship_modes=config.SHIP_MODE_CHOICES,
        today=datetime.today().strftime("%Y-%m-%d"),
        model_ready=prediction_service is not None,
        model_error=MODEL_LOAD_ERROR,
        advertising_stats=analytics.compute_advertising_stats(prediction_service.history)
        if prediction_service is not None and prediction_service.is_advertising else None,
        values=session.pop("prediction_values", {}),
    )


@app.route("/predict", methods=["POST"])
def predict_submit():
    """
    Input -> Preprocessing -> ML Model -> Prediction -> Result.

    "Preprocessing" here means src/predict.py's PredictionService, which
    applies the exact same date-feature engineering, label encoding,
    one-hot encoding, and lag/rolling lookups used at training time (see
    src/predict.py's module docstring). This route never re-implements
    any of that logic itself — it only collects form input and calls it.
    """
    if prediction_service is None:
        flash(f"Model is not available: {MODEL_LOAD_ERROR}", "error")
        return redirect(url_for("predict_form"))

    form = request.form
    try:
        if prediction_service.is_advertising:
            model_input = {key: form.get(key) for key in ("TV", "Radio", "Newspaper")}
            predicted_sales = prediction_service.predict(model_input)
            model_input = {key: float(value) for key, value in model_input.items()}
            record_extras = {"product": "Advertising campaign", "store": None,
                             "promotion": None, "price": None}
            record = {**record_extras, "category": "Advertising", "region": "—",
                      "segment": "—", "ship_mode": "—", "quantity": 0,
                      "discount": 0, "order_date": datetime.today().date().isoformat()}
        else:
            model_input = {"Order Date": form.get("order_date"),
                           "Category": form.get("category"), "Region": form.get("region"),
                           "Segment": form.get("segment"), "Ship Mode": form.get("ship_mode"),
                           "Quantity": float(form.get("quantity", "")),
                           "Discount": float(form.get("discount", ""))}
            datetime.strptime(model_input["Order Date"], "%Y-%m-%d")
            for field, choices in [("Category", config.CATEGORY_CHOICES), ("Region", config.REGION_CHOICES),
                                   ("Segment", config.SEGMENT_CHOICES), ("Ship Mode", config.SHIP_MODE_CHOICES)]:
                if model_input[field] not in choices:
                    raise ValueError(f"Select a valid {field}.")
            quantity, discount = model_input["Quantity"], model_input["Discount"]
            if not math.isfinite(quantity) or quantity < 1 or not quantity.is_integer():
                raise ValueError("Quantity must be a positive whole number.")
            if not math.isfinite(discount) or not 0 <= discount <= 1:
                raise ValueError("Discount must be between 0 and 1.")
            price = float(form["price"]) if form.get("price") else None
            if price is not None and (not math.isfinite(price) or price < 0):
                raise ValueError("Price must be a finite, non-negative number.")
            record_extras = {"product": form.get("product") or None, "store": form.get("store") or None,
                             "promotion": form.get("promotion") or None, "price": price}
            predicted_sales = prediction_service.predict(model_input)
            record = {**record_extras, **{key: model_input[source] for key, source in
                      [("category", "Category"), ("region", "Region"), ("segment", "Segment"),
                       ("ship_mode", "Ship Mode"), ("quantity", "Quantity"),
                       ("discount", "Discount"), ("order_date", "Order Date")]}}
    except (ValueError, TypeError) as exc:
        session["prediction_values"] = {key: value[:100] for key, value in form.items()}
        flash(f"Invalid prediction input: {exc}", "error")
        return redirect(url_for("predict_form"))
    record.update(user_id=g.user["id"] if g.user else None,
                  predicted_sales=predicted_sales, inputs_json=json.dumps(model_input))
    db_operations.insert_prediction(record)

    return render_template(
        "result.html",
        predicted_sales=predicted_sales,
        model_input=model_input,
        record_extras=record_extras,
        advertising=prediction_service.is_advertising,
        insight=analytics.prediction_insight(prediction_service.history, model_input, predicted_sales)
        if prediction_service.is_advertising else None,
    )


# --------------------------------------------------------------------------
# DASHBOARD
# --------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    stats = db_operations.get_dashboard_stats(g.user["id"] if g.user else None)
    recent = db_operations.get_prediction_history(user_id=g.user["id"] if g.user else None, limit=10)

    eda_figures = []
    if os.path.isdir(config.REPORTS_FIGURES_DIR):
        eda_figures = sorted(
            f for f in os.listdir(config.REPORTS_FIGURES_DIR)
            if f.lower().endswith(".png")
        )

    return render_template(
        "dashboard.html",
        stats=stats,
        recent=recent,
        eda_figures=eda_figures,
    )


# --------------------------------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------------------------------
@app.route("/performance")
def performance():
    comparison_table = []
    if os.path.exists(config.MODEL_COMPARISON_PATH):
        comparison_table = pd.read_csv(config.MODEL_COMPARISON_PATH).to_dict(orient="records")

    evaluation_table = []
    if os.path.exists(config.EVALUATION_METRICS_PATH):
        evaluation_table = pd.read_csv(config.EVALUATION_METRICS_PATH).to_dict(orient="records")

    eval_figures = []
    if os.path.isdir(config.REPORTS_EVAL_FIGURES_DIR):
        eval_figures = sorted(
            f for f in os.listdir(config.REPORTS_EVAL_FIGURES_DIR)
            if f.lower().endswith(".png")
        )

    return render_template(
        "performance.html",
        comparison_table=comparison_table,
        evaluation_table=evaluation_table,
        eval_figures=eval_figures,
    )


# --------------------------------------------------------------------------
# ANALYTICS DASHBOARD
# --------------------------------------------------------------------------
@app.route("/analytics")
def analytics_dashboard():
    """
    Real dataset + model analytics — every number here is computed live
    from data/raw/sales.csv (or the processed dataset) and the saved
    model comparison table in src/analytics.py. Nothing is hardcoded.
    """
    data = analytics.get_analytics_data()
    return render_template("analytics.html", data=data)


# --------------------------------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------------------------------
@app.route("/history")
def history():
    """
    Logged-in users see only their own predictions; a guest sees every
    guest-made prediction (there's no per-guest identity to filter by).
    """
    user_id = g.user["id"] if g.user else None
    records = db_operations.get_prediction_history(user_id=user_id, limit=100)
    return render_template("history.html", records=records, viewing_own=g.user is not None)


@app.route("/history/export")
def export_history():
    """Export all history visible to the current account or guest."""
    records = db_operations.get_prediction_history(
        user_id=g.user["id"] if g.user else None, limit=-1)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Created at (UTC)", "Model inputs", "Predicted sales"])
    for record in records:
        inputs = record["inputs_json"] or record["category"]
        # Prevent spreadsheet software from interpreting legacy text as a formula.
        if inputs.lstrip().startswith(("=", "+", "-", "@")):
            inputs = "'" + inputs
        writer.writerow([record["created_at"], inputs, record["predicted_sales"]])
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=prediction-history.csv"})


# --------------------------------------------------------------------------
# STATIC FIGURE SERVING (reports/figures is outside static/, so it needs
# its own route rather than Flask's default static handler)
# --------------------------------------------------------------------------
@app.route("/figures/<path:filename>")
def serve_figure(filename):
    return send_from_directory(config.REPORTS_FIGURES_DIR, filename)


@app.route("/figures/evaluation/<path:filename>")
def serve_eval_figure(filename):
    return send_from_directory(config.REPORTS_EVAL_FIGURES_DIR, filename)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
