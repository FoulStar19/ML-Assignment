"""XGBoost demand model: forecasts the units sold in a record (one store-product sale) from the drivers known before it.

The classifiers say WHETHER a record is under- or overstocked. This regressor says HOW MUCH is needed:
the dashboard's recommender uses its forecast to set days of cover, reorder point, target level and order size.

Writes to model/xgboost_demand/outputs/ (demand_metrics.json, demand_model.joblib, gain_importance.*, cv_results.csv).
The file is named demand_metrics.json (not metrics.json) so the classifier tabs never mistake it for a classifier.
"""
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]                    # model/  (evaluation.py)
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR.parent / "src" / "backend"))       # data pipeline + data_modeling.py

import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from data_modeling import RANDOM_STATE, build_preprocessor, prepare_demand_data
from evaluation import _jsonable, parse_args, plot_bar

MODEL_NAME = "XGBoost demand"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
BASELINE_NAME = "Average units sold in the training period"     # what a forecast with no drivers would say


def build_pipeline(ds) -> Pipeline:
    """Same preprocessing as the XGBoost classifier. n_jobs=1 inside the model because the search runs candidates in parallel."""
    return Pipeline([
        ("prep", build_preprocessor(ds.numeric_features, ds.categorical_features, scale=False)),
        ("clf", XGBRegressor(objective="reg:squarederror", tree_method="hist",
                             random_state=RANDOM_STATE, n_jobs=1)),
    ])


SEARCH_SPACE = {
    "clf__n_estimators": randint(150, 600),
    "clf__max_depth": randint(3, 10),
    "clf__learning_rate": loguniform(0.01, 0.3),
    "clf__subsample": uniform(0.6, 0.4),            # 0.6 - 1.0
    "clf__colsample_bytree": uniform(0.5, 0.5),     # 0.5 - 1.0
    "clf__min_child_weight": randint(1, 10),
    "clf__gamma": uniform(0, 0.5),
    "clf__reg_lambda": loguniform(0.1, 10),
    "clf__reg_alpha": loguniform(1e-3, 1),
}


def regression_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "wape": float(np.abs(y - pred).sum() / max(y.sum(), 1e-9)),     # total miss as a share of total demand
        "bias": float((pred - y).mean()),                               # > 0 means the forecast runs high
    }


def subcategory_average_baseline(ds) -> np.ndarray:
    """Predict each test row with the average demand its product subcategory had in the training period."""
    key = lambda X: X["product_subcategory"].astype(str)
    per_group = pd.Series(ds.y_train, index=key(ds.X_train)).groupby(level=0).mean()
    return key(ds.X_test).map(per_group).fillna(ds.y_train.mean()).to_numpy()


def explain(model, out_dir):
    names = model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    imp.to_csv(Path(out_dir) / "gain_importance.csv", header=["importance"])
    plot_bar(imp.head(15), "XGBoost demand - gain importance (top 15)", Path(out_dir) / "gain_importance.png", "importance")


def run(dataset, n_iter=20, cv_splits=3):
    """Tune + evaluate on an already prepared demand dataset (also called by data_modeling.data_model)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv = TimeSeriesSplit(n_splits=cv_splits, gap=dataset.cv_gap)
    X_tr, y_tr, X_te, y_te = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test

    print(f"\n{'=' * 45}\n{MODEL_NAME}\n{'=' * 45}")
    print(f"Train rows : {len(X_tr):,}   Test rows : {len(X_te):,}\n{dataset.split_info}\n")

    baseline = regression_metrics(y_te, np.full(len(y_te), float(np.mean(y_tr))))
    group_avg = regression_metrics(y_te, subcategory_average_baseline(dataset))
    print(f"Baseline ({BASELINE_NAME}): MAE {baseline['mae']:.3f}  WAPE {baseline['wape']:.1%}  R2 {baseline['r2']:.3f}")
    print(f"Baseline (subcategory average in training): MAE {group_avg['mae']:.3f}  WAPE {group_avg['wape']:.1%}  R2 {group_avg['r2']:.3f}")

    # ── Stage 1: default settings
    t0 = time.time()
    pipe = build_pipeline(dataset)
    cv_mae = float(-cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error").mean())
    default_test = regression_metrics(y_te, pipe.fit(X_tr, y_tr).predict(X_te))
    default = {"cv_mae": cv_mae, "seconds": round(time.time() - t0, 1), "test": default_test}
    print(f"Default  CV MAE {cv_mae:.3f} | test MAE {default_test['mae']:.3f}  WAPE {default_test['wape']:.1%}  R2 {default_test['r2']:.3f}")

    # ── Stage 2: tuning
    t0 = time.time()
    search = RandomizedSearchCV(
        build_pipeline(dataset), SEARCH_SPACE, n_iter=n_iter, scoring="neg_mean_absolute_error", cv=cv,
        n_jobs=-1, random_state=RANDOM_STATE, refit=True, error_score="raise")
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    tuned_test = regression_metrics(y_te, best.predict(X_te))
    tuned = {"cv_mae": float(-search.best_score_), "seconds": round(time.time() - t0, 1),
             "best_params": search.best_params_, "test": tuned_test}
    print(f"Tuned    CV MAE {tuned['cv_mae']:.3f} | test MAE {tuned_test['mae']:.3f}  WAPE {tuned_test['wape']:.1%}  R2 {tuned_test['r2']:.3f}")
    print(f"Best parameters: {json.dumps(search.best_params_, default=_jsonable)}")

    results = {
        "model": MODEL_NAME, "target": "units sold in the record",
        "n_iter": n_iter, "cv_splits": cv_splits, "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "split": dataset.split_info,
        "baseline": {"name": BASELINE_NAME, **baseline},
        "subcategory_average": {"name": "Average of the product subcategory in the training period", **group_avg},
        "default": default, "tuned": tuned,
    }
    (OUT_DIR / "demand_metrics.json").write_text(json.dumps(results, indent=2, default=_jsonable))
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(OUT_DIR / "cv_results.csv", index=False)
    joblib.dump(best, OUT_DIR / "demand_model.joblib", compress=3)      # the dashboard loads this file
    explain(best, OUT_DIR)
    print(f"Results saved to {OUT_DIR}\n")
    return results


if __name__ == "__main__":
    args = parse_args("Train and evaluate the XGBoost demand model")
    run(prepare_demand_data(args.data), n_iter=args.n_iter, cv_splits=args.cv_splits)
