"""The best trained model: which one it is, loading it, and scoring store-product lines with it.

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

from config import MODEL_DIR, ROOT_DIR

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
    """Score every line with the model and let its call drive status and order sizes.

    The model only says Understock / Healthy / Overstock, so quantities still come from the
    reorder-point policy: an understocked line is topped up to its target level (at least one
    review period of demand), an overstocked line has everything above target as surplus.
    """
    X = lines[FEATURES].copy()
    X["holiday_promotion"] = X["holiday_promotion"].astype(str)
    proba = model.predict_proba(X)

    out = lines.copy()
    out["p_under"], out["p_healthy"], out["p_over"] = proba[:, 0], proba[:, 1], proba[:, 2]
    out["confidence"] = proba.max(axis=1)
    out["status"] = [STATUS_NAMES[CLASS_NAMES[i]] for i in proba.argmax(axis=1)]

    under, over = out["status"] == "Understock risk", out["status"] == "Overstock"
    gap = out["target_level"] - out["inventory_level"]
    top_up = np.maximum(gap, out["mu"] * review)
    out["order_qty"] = np.where(under, np.ceil(top_up), 0)
    out["order_value"] = out["order_qty"] * out["avg_price"]
    out["excess_units"] = np.where(over, np.floor(-gap).clip(lower=0), 0)
    out["excess_value"] = out["excess_units"] * out["avg_price"]
    return out
