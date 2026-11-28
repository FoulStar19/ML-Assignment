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
    return json.loads(_need("best_params.json").read_text())


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(_need("predictions.csv"), parse_dates=["Date"])


def load_worst_predictions() -> pd.DataFrame:
    return pd.read_csv(_need("worst_predictions.csv"))


def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(_need("feature_importance.csv"), index_col="feature")


def load_segment_errors() -> pd.DataFrame:
    return pd.read_csv(_need("segment_errors.csv"))


def load_tuning_trials() -> pd.DataFrame:
    return pd.read_csv(_need("tuning_trials.csv"))


def compare_models() -> pd.DataFrame:
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
    return {p.stem: p for p in sorted(OUT_DIR.glob("*.png"))}


def load_model():
    return joblib.load(_need("model.joblib"))


def load_default_model():
    return joblib.load(_need("model_default.joblib"))


# ─────────────────────────────────────────────
# 3. FORECAST FOR NEW ROWS
# ─────────────────────────────────────────────
def predict_units(df: pd.DataFrame) -> np.ndarray:
    from data_modeling import FEATURES, PREPROCESSOR_FILE
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    X = joblib.load(PREPROCESSOR_FILE).transform(df[FEATURES])
    return np.clip(load_model().predict(X), 0, None)          # units sold cannot be negative
