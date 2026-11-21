"""The trained models the recommender uses, and how it scores the current records with them.

1. Status model: the classifier with the best tuned test macro-F1. It calls each record Understock / Healthy / Overstock.
2. Demand model: the XGBoost regressor that forecasts the units sold in a record. It sets cover days, reorder point and order size.

"Best" means the highest tuned test macro-F1, the same rule pipeline.py and the Model results tab use.
The pipeline saves each tuned model as model/<name>/outputs/model.joblib. If that file is missing
(for example on a fresh clone, because joblib files are git-ignored), the model is refitted once from
its saved best settings and stored, so the dashboard always starts.
"""
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import DEMAND_DIR, MODEL_DIR, ROOT_DIR
from planning import load_data

for _path in (ROOT_DIR / "src" / "backend", MODEL_DIR):   # the pipeline code lives outside the frontend folder
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from data_modeling import CATEGORICAL_FEATURES, CLASS_NAMES, NUMERIC_FEATURES  # noqa: E402

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
STATUS_NAMES = {"Understock": "Understock risk", "Healthy": "Healthy", "Overstock": "Overstock"}


def best_model_info() -> dict | None:
    """Name, folder and test scores of the tuned model with the best macro-F1 (None if nothing is trained)."""
    best = None
    for path in sorted(MODEL_DIR.glob("*/outputs/metrics.json")):
        metrics = json.loads(path.read_text())
        test = metrics["tuned"]["test"]
        info = {
            "name": metrics["model"], "folder": path.parent, "metrics": metrics,
            "f1": test["f1_macro"], "accuracy": test["accuracy"], "mcc": test["mcc"],
            "under_recall": test["per_class"]["Understock"]["recall"],
            "over_recall": test["per_class"]["Overstock"]["recall"],
        }
        if best is None or info["f1"] > best["f1"]:
            best = info
    return best


def _refit(info: dict):
    """Fit the best model again on the training period using the settings the tuning found."""
    from data_modeling import _load_model_script, prepare_data

    model_dir = info["folder"].parent
    module = _load_model_script(model_dir / f"{model_dir.name}_model.py")
    ds = prepare_data()
    pipe = module.build_pipeline(ds)
    pipe.set_params(**info["metrics"]["tuned"]["best_params"])
    if "clf__n_jobs" in pipe.get_params():
        pipe.set_params(clf__n_jobs=-1)
    extra = module.fit_params(ds) if hasattr(module, "fit_params") else {}
    return pipe.fit(ds.X_train, ds.y_train, **extra)


@st.cache_resource(show_spinner=False)
def _load(path_str: str, mtime: float, name: str):
    """Cached per model file, so a new training run (new mtime) loads the new model."""
    return joblib.load(path_str)


def load_best_model(info: dict):
    """The fitted best model. Loads model.joblib, or refits and saves it if it is missing or unreadable."""
    path = info["folder"] / "model.joblib"
    if path.exists():
        try:
            return _load(str(path), path.stat().st_mtime, info["name"])
        except Exception:        # e.g. saved with another scikit-learn version
            pass
    with st.spinner(f"Preparing {info['name']} (first start only, about a minute)..."):
        model = _refit(info)
        joblib.dump(model, path, compress=3)
    return model


def add_predictions(lines: pd.DataFrame, model, review: int) -> pd.DataFrame:
    """Score every record with the model and let its call drive status and order sizes.

    The model only says Understock / Healthy / Overstock, so quantities still come from the
    reorder-point policy: an understocked line is topped up to its target level (at least one
    review period of demand), an overstocked line has everything above target as surplus.
    """
    proba = model.predict_proba(lines[FEATURES])

    out = lines.copy()
    out["p_under"], out["p_healthy"], out["p_over"] = proba[:, 0], proba[:, 1], proba[:, 2]
    out["confidence"] = proba.max(axis=1)
    out["status"] = [STATUS_NAMES[CLASS_NAMES[i]] for i in proba.argmax(axis=1)]

    under, over = out["status"] == "Understock risk", out["status"] == "Overstock"
    gap = out["target_level"] - out["inventory_level"]
    top_up = np.maximum(gap, out["mu"] * review)
    out["order_qty"] = np.where(under, np.ceil(top_up), 0)
    out["order_value"] = out["order_qty"] * out["price"]
    out["excess_units"] = np.where(over, np.floor(-gap).clip(lower=0), 0)
    out["excess_value"] = out["excess_units"] * out["price"]
    return out


# ─────────────────────────────────────────────
# XGBoost demand model
# ─────────────────────────────────────────────
def demand_model_info() -> dict | None:
    """Test scores of the tuned XGBoost demand model (None if it has not been trained)."""
    path = DEMAND_DIR / "demand_metrics.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    t = m["tuned"]["test"]
    return {"name": "XGBoost", "folder": DEMAND_DIR, "metrics": m, "wape": t["wape"], "mae": t["mae"],
            "baseline_wape": m["baseline"]["wape"]}


def _refit_demand(info: dict):
    """Fit the demand model again on the training period using the settings the tuning found."""
    from data_modeling import _load_model_script, prepare_demand_data

    module = _load_model_script(info["folder"].parent / "xgboost_demand_model.py")
    ds = prepare_demand_data()
    pipe = module.build_pipeline(ds)
    pipe.set_params(**info["metrics"]["tuned"]["best_params"])
    if "clf__n_jobs" in pipe.get_params():
        pipe.set_params(clf__n_jobs=-1)
    return pipe.fit(ds.X_train, ds.y_train)


def load_demand_model(info: dict):
    """The fitted demand model. Loads demand_model.joblib, or refits and saves it if it is missing or unreadable."""
    path = info["folder"] / "demand_model.joblib"
    if path.exists():
        try:
            return _load(str(path), path.stat().st_mtime, "demand")
        except Exception:
            pass
    with st.spinner("Preparing the XGBoost demand model (first start only)..."):
        model = _refit_demand(info)
        joblib.dump(model, path, compress=3)
    return model


@st.cache_data(show_spinner="Forecasting demand...")
def _forecast_all(stamp: float, _model) -> pd.DataFrame:
    """Expected units sold for every record. `stamp` refreshes the cache after a new training run."""
    df = load_data()
    out = df[["transaction_id"]].copy()
    out["mu_model"] = np.clip(_model.predict(df[FEATURES]), 1, None)
    return out


def with_model_demand(info: dict) -> pd.DataFrame:
    """Every record with the XGBoost forecast as its expected demand (mu); the policy numbers then follow from it.

    sigma is the model's test RMSE: how far its forecasts typically miss.
    """
    model = load_demand_model(info)
    stamp = (info["folder"] / "demand_model.joblib").stat().st_mtime
    out = load_data().merge(_forecast_all(stamp, model), on="transaction_id", how="left")
    out = out.rename(columns={"mu_model": "mu"})
    out["sigma"] = float(info["metrics"]["tuned"]["test"]["rmse"])
    return out
