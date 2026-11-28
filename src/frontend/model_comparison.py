"""Model comparison tab, laid out as in the approved mockup:

  1. One table with every model, default and tuned (test set).
  2. Left card: model tabs (Linear | Random Forest | XGBoost | MLP), Tuned / Default sub-tabs, 3 x 2 grid of graphs.
     Right card: tuning parameters of the selected model (grid searched, best value, CV / scoring, default vs tuned change).

Frontend only. It reads what the model scripts already save in model/<name>/outputs/
(*_base_results.json, *_tuned_results.json, *.png, *_base.joblib); nothing is trained here.
The tuning grids below mirror the param_grid in each model script, so update both if a grid changes.
"""
import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from . import charts
from .config import MODEL_DIR
from .ui import STRETCH, tab_strip

TABLE_METRICS = ["MAE", "RMSE", "R²", "MAPE"]
FORMATS = {"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.4f}", "MAPE": "{:.2f}%"}
VERSIONS = {"base": "Default", "tuned": "Tuned"}
CV_TEXT = "5-fold, neg MSE"            # cv=5, scoring="neg_mean_squared_error" in every model script


@dataclass(frozen=True)
class ModelInfo:
    name: str          # row label in the table
    tab: str           # label on the model tab
    folder: str        # model/<folder>/ (outputs are in model/<folder>/outputs/)
    script: str        # script that produces the outputs
    key: str           # model key inside the results JSON
    prefix: str        # results files: <prefix>_base_results.json / <prefix>_tuned_results.json
    plot_prefix: str   # graph files: <plot_prefix>_<graph>.png (the scripts save graphs for the tuned model only)
    base_file: str     # saved default (untuned) model
    grid: dict = field(default_factory=dict)
    fixed: str = ""
    has_loss_curve: bool = False


MODEL_INFO = [
    ModelInfo("Linear (Ridge)", "Linear", "linear_regression", "linear_regression_model.py", "ridge", "linear_regression",
              "ridge_tuned", "ridge_base.joblib", {"alpha": [0.1, 1.0, 10.0]}),
    ModelInfo("Random Forest", "Random Forest", "random_forest", "random_forest_model.py", "random_forest", "random_forest",
              "random_forest_tuned", "random_forest_base.joblib",
              {"n_estimators": [100, 300], "max_depth": [10, 20, None], "min_samples_leaf": [1, 3, 5]}),
    ModelInfo("XGBoost", "XGBoost", "xgboost", "xgboost_model.py", "xgboost", "xgboost", "xgboost_tuned", "xgboost_base.joblib",
              {"n_estimators": [100, 300], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}, has_loss_curve=True),
    ModelInfo("MLP", "MLP", "mlp", "mlp_model.py", "mlp", "mlp", "mlp_tuned", "mlp_base.joblib",
              {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "alpha": [0.0001, 0.001, 0.01], "learning_rate_init": [0.001, 0.01]},
              "max_iter=500, early stopping, n_iter_no_change=10", has_loss_curve=True),
]
DEFAULT_TAB = 2                        # XGBoost, as in the mockup

# (tile title, file pattern(s) tried in order). Feature importance falls back to permutation importance (the MLP has no built-in one).
TILES = [("Actual vs predicted", ["actual_vs_predicted"]), ("Residuals", ["residuals"]),
         ("Feature importance", ["feature_importance", "permutation_importance"]),
         ("Learning curve", ["learning_curve"]), ("Validation curve", ["validation_curve_*"]), ("Loss curve", ["loss_curve"])]


# ─────────────────────────────────────────────
# Loading what the model scripts saved
# ─────────────────────────────────────────────
def out_dir(info):
    return MODEL_DIR / info.folder / "outputs"


@st.cache_data(show_spinner=False)
def _read_json(path, mtime):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_results(info, version):
    """Saved results of one model ("base" or "tuned"), or None if its script has not been run."""
    path = out_dir(info) / f"{info.prefix}_{version}_results.json"
    if not path.exists():
        return None
    return _read_json(str(path), path.stat().st_mtime).get(info.key)


@st.cache_data(show_spinner="Scoring the default model...")
def _base_predictions(path, mtime, _X):
    model = joblib.load(path)
    try:
        pred = model.predict(_X)
    except Exception:                                   # the default XGBoost was fitted on plain float32 arrays
        pred = model.predict(np.asarray(_X, dtype=np.float32))
    return np.asarray(pred, dtype=float)


def fmt_value(v):
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(str(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    return str(v)


# ─────────────────────────────────────────────
# 1. All models: default vs tuned
# ─────────────────────────────────────────────
def all_models_card():
    rows = []
    for info in MODEL_INFO:
        for version, label in VERSIONS.items():
            res = load_results(info, version)
            if res is not None:
                rows.append({"Model": info.name, "Version": label, **{m: res["test_metrics"].get(m) for m in TABLE_METRICS}})
    with st.container(border=True):
        st.markdown("##### All models: default vs tuned (test set)")
        df = pd.DataFrame(rows)
        st.dataframe(df.style.format(FORMATS) if hasattr(df, "style") else df, hide_index=True, **STRETCH)
        st.caption("Filled from each model's results file. Lower is better for MAE, RMSE and MAPE; higher for R².")


# ─────────────────────────────────────────────
# 2a. Left card: graphs of the selected model
# ─────────────────────────────────────────────
def find_graph(info, patterns):
    for pattern in patterns:
        for path in sorted(out_dir(info).glob(f"{info.plot_prefix}_{pattern}.png")):
            return path
    return None


def graph_grid(info, version, X, y):
    pred = None
    if version == "Default":
        path = out_dir(info) / info.base_file
        if path.exists():
            try:
                pred = _base_predictions(str(path), path.stat().st_mtime, X)
            except Exception as e:
                st.warning(f"Could not score the default model: {e}")
    sample = np.random.default_rng(0).choice(len(y), min(3000, len(y)), replace=False)
    for row in (TILES[:3], TILES[3:]):
        for col, (title, patterns) in zip(st.columns(3), row):
            with col:
                st.markdown(f"**{title}**")
                if version == "Tuned":
                    path = find_graph(info, patterns)
                    if path is not None:
                        st.image(str(path))
                    elif title == "Loss curve" and not info.has_loss_curve:
                        st.caption("Not available for this model.")
                    else:
                        st.caption(f"Not generated yet. Run `python model/{info.folder}/{info.script}`.")
                elif pred is not None and title == "Actual vs predicted":
                    st.altair_chart(charts.actual_vs_predicted(y.to_numpy()[sample], pred[sample]), **STRETCH)
                elif pred is not None and title == "Residuals":
                    st.altair_chart(charts.residual_hist(y.to_numpy(), pred), **STRETCH)
                else:
                    st.caption("Only saved for the tuned model.")


def graphs_card(X, y):
    with st.container(border=True):
        tab = tab_strip([i.tab for i in MODEL_INFO], DEFAULT_TAB, "mc_model")
        info = next(i for i in MODEL_INFO if i.tab == tab)
        version = tab_strip(["Tuned", "Default"], 0, "mc_version")
        graph_grid(info, version, X, y)
    return info


# ─────────────────────────────────────────────
# 2b. Right card: tuning parameters of the selected model
# ─────────────────────────────────────────────
def tuning_card(info):
    tuned, base = load_results(info, "tuned"), load_results(info, "base")
    best = (tuned or {}).get("best_params") or {}
    with st.container(border=True):
        st.markdown(f"##### Tuning parameters ({info.tab})")
        rows = [{"Parameter": p, "Grid searched": ", ".join(fmt_value(v) for v in values),
                 "Best": fmt_value(best[p]) if p in best else "—"} for p, values in info.grid.items()]
        rows.append({"Parameter": "CV / scoring", "Grid searched": CV_TEXT, "Best": ""})
        if info.fixed:
            rows.append({"Parameter": "Fixed settings", "Grid searched": info.fixed, "Best": ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)

        st.markdown("**Default vs tuned change (test set)**")
        if tuned and base:
            change = [{"Metric": m, "Default": base["test_metrics"][m], "Tuned": tuned["test_metrics"][m],
                       "Change": (tuned["test_metrics"][m] - base["test_metrics"][m]) / abs(base["test_metrics"][m])
                       if base["test_metrics"][m] else np.nan} for m in TABLE_METRICS]
            st.dataframe(pd.DataFrame(change).style.format({"Default": "{:.4f}", "Tuned": "{:.4f}", "Change": "{:+.1%}"}),
                         hide_index=True, **STRETCH)
            st.caption("Change = (tuned - default) / default.")
        else:
            st.caption("Needs both the default and the tuned results files.")


# ─────────────────────────────────────────────
# Entry point (called from dashboard.models_tab)
# ─────────────────────────────────────────────
def render(X, y):
    if not any(load_results(i, v) for i in MODEL_INFO for v in VERSIONS):
        st.info("No model results found. Train the models first, for example:")
        st.code("\n".join(f"python model/{i.folder}/{i.script}" for i in MODEL_INFO), language="bash")
        return
    all_models_card()
    left, right = st.columns(2)
    with left:
        info = graphs_card(X, y)
    with right:
        tuning_card(info)
