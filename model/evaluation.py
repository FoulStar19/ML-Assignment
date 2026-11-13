import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from data_modeling import RANDOM_STATE

LINE = "─" * 45

# ─────────────────────────────────────────────
# 1. COMMAND LINE
# ─────────────────────────────────────────────
def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--n-iter", type=int, default=20,
                   help="random-search candidates to try (default 20)")
    p.add_argument("--cv-splits", type=int, default=3,
                   help="time-series CV folds used for tuning (default 3)")
    p.add_argument("--data", default=None,
                   help="processed CSV to use (default: data/processed/..., "
                        "the data pipeline is run first if it is missing)")
    return p.parse_args()


# ─────────────────────────────────────────────
# 2. METRICS
# ─────────────────────────────────────────────
def evaluate(model, X, y, class_names):
    """All test-set metrics for one fitted model. Returns (metrics dict, predictions)."""
    pred = model.predict(X)
    labels = list(range(len(class_names)))
    prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "f1_macro": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", labels=labels, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "per_class": {
            name: {"precision": prec[i], "recall": rec[i], "f1": f1[i], "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
    }
    if hasattr(model, "predict_proba"):
        try:
            out["roc_auc_ovr_macro"] = roc_auc_score(
                y, model.predict_proba(X), multi_class="ovr", average="macro", labels=labels)
        except ValueError:      # e.g. a class missing from the test set
            pass
    return out, pred


def _cv_f1(pipeline, ds, cv, fit_params):
    """Macro-F1 across the time-series CV folds for a model with fixed settings."""
    scores = []
    for tr, va in cv.split(ds.X_train):
        model = clone(pipeline)
        fp = {k: np.asarray(v)[tr] for k, v in fit_params.items()}
        model.fit(ds.X_train.iloc[tr], ds.y_train[tr], **fp)
        scores.append(f1_score(ds.y_train[va], model.predict(ds.X_train.iloc[va]), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


# ─────────────────────────────────────────────
# 3. PLOTS
# ─────────────────────────────────────────────
def plot_bar(series, title, path, xlabel):
    """Horizontal bar chart of a pandas Series (largest at the top)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    series.iloc[::-1].plot.barh(ax=ax, color="#2B4C7E")
    ax.set(title=title, xlabel=xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(y, pred, class_names, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, labels=list(range(len(class_names))), display_labels=class_names,
        normalize="true", values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_permutation_importance(model, ds, out_dir, n_repeats=5, max_rows=5000):
    """How much does test macro-F1 drop when one ORIGINAL column is shuffled?
    Works on the raw columns, so one-hot columns of the same category are shuffled together."""
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(ds.X_test), size=min(max_rows, len(ds.X_test)), replace=False)
    result = permutation_importance(
        model, ds.X_test.iloc[idx], ds.y_test[idx], scoring="f1_macro",
        n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(result.importances_mean, index=ds.X_test.columns).sort_values(ascending=False)
    imp.to_csv(Path(out_dir) / "permutation_importance.csv", header=["importance"])
    plot_bar(imp.head(15), "Permutation importance (drop in test macro-F1, top 15)",
             Path(out_dir) / "permutation_importance.png", "drop in macro-F1")


# ─────────────────────────────────────────────
# 4. MAIN EXPERIMENT
# ─────────────────────────────────────────────
def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _print_stage(label, cv_mean, cv_std, test):
    print(f"{label:<8} CV macro-F1 {cv_mean:.4f} ± {cv_std:.4f} | test: "
          f"accuracy {test['accuracy']:.4f}  macro-F1 {test['f1_macro']:.4f}  "
          f"Understock recall {test['per_class']['Understock']['recall']:.4f}  "
          f"Overstock recall {test['per_class']['Overstock']['recall']:.4f}")


def run_model_experiment(model_name, pipeline, search_space, ds, out_dir,
                         n_iter=20, cv_splits=3, fit_params=None, explain=None):
    """Default fit -> tuning -> test evaluation -> files in out_dir.

    fit_params : extra fit() arguments, e.g. {"clf__sample_weight": weights}
    explain    : optional function(model, ds, out_dir) for model-specific plots
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_params = fit_params or {}
    # Chronological CV: each fold trains on the past and validates on the next block.
    # `gap` skips the rows whose labels look into the validation block.
    cv = TimeSeriesSplit(n_splits=cv_splits, gap=ds.cv_gap)

    print(f"\n{'=' * 45}\n{model_name}\n{'=' * 45}")
    print(ds.describe())

    # ── Stage 1: default settings
    t0 = time.time()
    cv_mean, cv_std = _cv_f1(pipeline, ds, cv, fit_params)
    default_model = clone(pipeline).fit(ds.X_train, ds.y_train, **fit_params)
    default_test, _ = evaluate(default_model, ds.X_test, ds.y_test, ds.class_names)
    default = {"cv_f1_macro_mean": cv_mean, "cv_f1_macro_std": cv_std,
               "seconds": round(time.time() - t0, 1), "test": default_test}
    _print_stage("Default", cv_mean, cv_std, default_test)

    # ── Stage 2: tuning
    t0 = time.time()
    search = RandomizedSearchCV(
        clone(pipeline), search_space, n_iter=n_iter, scoring="f1_macro", cv=cv,
        n_jobs=-1, random_state=RANDOM_STATE, refit=True, error_score="raise")
    search.fit(ds.X_train, ds.y_train, **fit_params)
    best = search.best_estimator_
    best_idx = search.best_index_
    tuned_test, tuned_pred = evaluate(best, ds.X_test, ds.y_test, ds.class_names)
    tuned = {"cv_f1_macro_mean": float(search.best_score_),
             "cv_f1_macro_std": float(search.cv_results_["std_test_score"][best_idx]),
             "seconds": round(time.time() - t0, 1),
             "best_params": search.best_params_, "test": tuned_test}
    _print_stage("Tuned", tuned["cv_f1_macro_mean"], tuned["cv_f1_macro_std"], tuned_test)
    print(f"Best parameters: {json.dumps(search.best_params_, default=_jsonable)}")

    # ── Files
    results = {
        "model": model_name, "n_iter": n_iter, "cv_splits": cv_splits,
        "n_train": int(len(ds.X_train)), "n_test": int(len(ds.X_test)),
        "split": ds.split_info, "default": default, "tuned": tuned,
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=_jsonable))
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        out_dir / "cv_results.csv", index=False)
    (out_dir / "classification_report.txt").write_text(
        classification_report(ds.y_test, tuned_pred, target_names=ds.class_names, digits=4))
    _plot_confusion(ds.y_test, tuned_pred, ds.class_names,
                    f"{model_name} (tuned) - test set", out_dir / "confusion_matrix.png")
    # The dashboard loads this file to score live store-product lines.
    joblib.dump(best, out_dir / "model.joblib", compress=3)
    if explain is not None:
        explain(best, ds, out_dir)

    print(f"Results saved to {out_dir}\n")
    return results
