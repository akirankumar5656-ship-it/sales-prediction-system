# Sales Prediction System

Flask app with sales prediction, model comparison, diagnostic charts, analytics,
accounts, and prediction history.

## Quick start

Python 3.11 or newer is recommended. Create a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
python app.py
```

Open http://127.0.0.1:5000. On this machine, if Apple's `python3` asks for an
Xcode license, use `/opt/homebrew/bin/python3` to create the environment.

## Dataset and predictions

The supplied `data/raw/sales data file.csv` contains 200 advertising observations:
`TV`, `Radio`, `Newspaper`, and target `Sales`. The app uses those three advertising
budgets to predict sales. Enter values in the CSV's original units. No currency
or sales unit is inferred from the file. This is regression on advertising spend,
not a dated product-sales forecast or evidence of causal advertising effects.

The unrelated `car_purchasing.csv` is not used. A `data/raw/sales.csv`, if present,
takes precedence. Switching to retail data requires matching the documented
retail schema and rerunning the entire pipeline, then restarting the app.

The pipeline preprocesses data, constructs features, compares four regressors,
and saves the best model, metrics, and figures. Advertising data uses a reproducible
80/20 random split. Comparison metrics are used for model selection, so they are
validation estimates rather than an independent final test of the chosen model.
Retail data uses a chronological split and the existing date/category features.
Predictions outside the training input range may be unreliable.

## App features

- Prediction form matching the trained advertising model, with server validation.
- Example budgets, dataset median presets, and a live total-spend summary.
- Results compared with dataset average sales and observed-budget range checks.
- Dataset analytics; unavailable date/product breakdowns are explicitly omitted.
- Advertising channel spend distribution and average/median benchmarks.
- Model comparison, residual plots, and feature importances.
- Registration/login with hashed passwords.
- Personal history and dashboard for each account; guests share guest-only records.
- Search the latest 100 saved predictions and export all visible history to CSV.
- SQLite storage created automatically; existing records are preserved.

Model files are loaded once per app process. Restart after retraining. Default
paths work independently of the current working directory. Set `SECRET_KEY` for
deployment; local use generates a private key in `data/.session_secret`. Debug mode
is off unless `FLASK_DEBUG=1`. Use a production WSGI server for public hosting.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests use a temporary database and exercise the generated model artifacts.
