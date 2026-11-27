"""Read the XGBoost results saved in model/xgboost/outputs/ (for the dashboard / report). Nothing is computed here.

    from model.xgboost.xgboost_results import (load_metrics, load_predictions, load_feature_importance,
                                               load_segment_errors, load_model, predict_units, compare_models)

    m = load_metrics()                      # dict, see metrics.json
    m["tuned"]["test"]["rmse"]              # also "mae", "mse", "r2", "wape", "bias"
    compare_models()                        # one table: reference models, default, tuned and ensemble
    load_predictions()                      # DataFrame: Date, Transaction_ID, Product_ID, Units_Sold, pred_tuned, error ...
    predict_units(df)                       # forecast for new rows (see below)

Charts are ready-made PNG files in outputs/ (list_plots()).
Run `python model/xgboost/xgboost_model.py` first to create the files.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
ROOT_DIR = HERE.parents[1]
if str(ROOT_DIR / "src" / "backend") not in sys.path:        # so predict_units can import data_modeling
    sys.path.insert(0, str(ROOT_DIR / "src" / "backend"))


def _need(name: str) -> Path:
    """Path of a result file, or a clear error telling the user how to create it."""
    path = OUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first: python model/xgboost/xgboost_model.py")
    return path


# ─────────────────────────────────────────────
# 1. RESULT TABLES
# ─────────────────────────────────────────────
def load_metrics() -> dict:
    return json.loads(_need("metrics.json").read_text())


def load_best_params() -> dict:
    """The tuned hyper-parameters chosen by Optuna."""
    return json.loads(_need("best_params.json").read_text())


def load_predictions() -> pd.DataFrame:
    """Test-period predictions of the default (pred_default) and tuned (pred_tuned) model next to the actual Units_Sold."""
    return pd.read_csv(_need("predictions.csv"), parse_dates=["Date"])


def load_worst_predictions() -> pd.DataFrame:
    return pd.read_csv(_need("worst_predictions.csv"))


def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(_need("feature_importance.csv"), index_col="feature")


def load_segment_errors() -> pd.DataFrame:
    return pd.read_csv(_need("segment_errors.csv"))


def load_tuning_trials() -> pd.DataFrame:
    """Every Optuna trial (settings tried and their CV RMSE)."""
    return pd.read_csv(_need("tuning_trials.csv"))


def compare_models() -> pd.DataFrame:
    """Test-period scores of every model in this folder, one row each (best RMSE first)."""
    m = load_metrics()
    rows = {"Mean predictor": m["reference"]["mean_predictor"],
            "Single decision tree": m["reference"]["single_decision_tree"],
            "XGBoost (default)": m["default"]["test"],
            "XGBoost (tuned)": m["tuned"]["test"]}
    if "ensemble" in m:                                       # absent in results made by older runs
        rows[f"XGBoost ensemble (best {m['ensemble']['members']} settings)"] = m["ensemble"]["test"]
    return pd.DataFrame(rows).T.sort_values("rmse")


# ─────────────────────────────────────────────
# 2. CHARTS AND MODELS
# ─────────────────────────────────────────────
def list_plots() -> dict:
    """{plot name: path} for every PNG the training run drew."""
    return {p.stem: p for p in sorted(OUT_DIR.glob("*.png"))}


def load_model():
    """The final tuned XGBoost model."""
    return joblib.load(_need("model.joblib"))


def load_default_model():
    """The default (untuned) XGBoost model, kept for comparison."""
    return joblib.load(_need("model_default.joblib"))


# ─────────────────────────────────────────────
# 3. FORECAST FOR NEW ROWS
# ─────────────────────────────────────────────
def predict_units(df: pd.DataFrame) -> np.ndarray:
    """Forecast Units_Sold for new rows.

    `df` must be in the PROCESSED format (the columns of data/processed/retail_store_inventory_processed.csv:
    Promotion, Holiday, Month, Unit_Price, Discount_Percent, Marketing_Spend, Customer_Footfall, Product_Category,
    Product_Subcategory, Weather_Condition, Store_Type). The pipeline's saved preprocessor does the scaling and
    one-hot encoding, so the new rows are treated exactly like the training rows.
    """
    from data_modeling import FEATURES, PREPROCESSOR_FILE
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    X = joblib.load(PREPROCESSOR_FILE).transform(df[FEATURES])
    return np.clip(load_model().predict(X), 0, None)          # units sold cannot be negative
