"""XGBoost Regressor: predicts Units_Sold (demand) so stock can be planned before it runs out or piles up.

Run on its own (after `python src/backend/pipeline.py --skip-modeling` has made train.csv / test.csv):

    python model/xgboost/xgboost_model.py                 # full run (about 10-20 min on a laptop)
    python model/xgboost/xgboost_model.py --trials 20     # faster run
    python model/xgboost/xgboost_model.py --quick         # 2-minute smoke test, results are NOT final

or through the pipeline:  python src/backend/pipeline.py

Everything is written to model/xgboost/outputs/ (see README.md in this folder for the file list).

Why XGBoost: gradient boosting builds many small trees one after another, and every new tree is fitted to the
mistakes (residuals) of the trees before it. It captures non-linear effects and interactions
(e.g. Promotion x Holiday) without manual feature work, handles one-hot features and missing values natively
and has many regularisation settings, which makes it a strong model for tabular data.

Method, in the order of the assignment's Part A / Part B:
    1. Same data preparation as the other models: the pipeline's time-based train/test split (test = latest 30% of dates).
    2. Reference points: mean predictor and a single decision tree (shows what boosting adds).
    3. Default XGBoost, scored with time-series cross-validation (CV) on the training period only.
    4. Hyper-parameter tuning with Optuna (TPE, a smarter search than grid/random) + median pruning, scored by
       time-series CV, so the test set is never used to choose settings.
    5. Final fit on the whole training period, scored once on the untouched test period (MAE, MSE, RMSE, R2 + WAPE, bias).
    6. Interpretation: gain importance, grouped permutation importance, SHAP, partial dependence, promotion/holiday effect.
    7. Error analysis: residuals, error by segment, worst predictions, under/over-prediction.
    8. Robustness: 5 random seeds, performance over time, walk-forward retraining (does it adapt as new data comes in?).
"""
import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")                     # draw to files only, so it also works on a server without a screen
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# ─────────────────────────────────────────────
# 0. SETTINGS
# ─────────────────────────────────────────────
HERE = Path(__file__).resolve().parent

# The one path defined here: where the shared data-preparation code lives, so it can be imported.
# Everything else (column names, files, seed, metrics, loading) comes from those modules.
BACKEND_DIR = HERE.parents[1] / "src" / "backend"          # model/xgboost/xgboost_model.py -> <project>/src/backend
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from data_modeling import (CAT_COLS, DATE_COL, FLAG_COLS, LINE, NUM_COLS, PREPROCESSOR_FILE, ROOT_DIR, SEED,   # noqa: E402
                           STOCK_COLS, TARGET, get_feature_columns, load_train_test, model_output_dir)
if ROOT_DIR not in sys.path:                               # so `model.evaluation` can be imported when run on its own
    sys.path.insert(0, ROOT_DIR)
from model.evaluation import clipped_predict as predict    # noqa: E402  (predictions clipped at 0)
from model.evaluation import regression_metrics as metrics # noqa: E402  (MAE, MSE, RMSE, R2, WAPE, bias)

OUT_DIR = Path(model_output_dir(HERE.name))                # <project>/model/xgboost/outputs

# Settings that belong to this model only
N_TRIALS = 60            # Optuna trials. TPE needs roughly 10 x (number of tuned settings) to be reliable.
CV_SPLITS = 5            # TimeSeriesSplit folds: always train on the past, validate on the next block of dates
WALK_FORWARD_BLOCKS = 4  # the test period is cut into this many blocks for the "does it adapt over time" check
SHAP_ROWS = 2000         # test rows used for SHAP / partial dependence (keeps it fast)
FIXED = dict(tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="rmse")   # same for every XGBoost model here
SEGMENT_CATS = [c for c in ["Product_Category", "Store_Type", "Weather_Condition"] if c in CAT_COLS]   # error-by-segment charts
PDP_COLS = [c for c in ["Unit_Price", "Marketing_Spend", "Customer_Footfall", "Discount_Percent", "Price_Gap"]
            if c in NUM_COLS]                                         # numeric inputs drawn as partial-dependence curves

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})
BLUE, ORANGE, GREY, GREEN, RED = "#3B5BDB", "#F5A524", "#8A94A6", "#22A27B", "#E5484D"


# ─────────────────────────────────────────────
# 1. DATA (the pipeline's train.csv / test.csv, so every model sees exactly the same rows and features)
# ─────────────────────────────────────────────
def load_data():
    """(train, test, model input columns). The loading and the time-split checks are shared with every model."""
    train, test = load_train_test()
    # Inventory_Level, Returns and Sales_Amount are NOT inputs: they are only known after (or because of) the sale,
    # so using them to predict Units_Sold would leak the answer. The pipeline already leaves them out.
    return train, test, get_feature_columns(train)


def decode(df, prefix):
    """Turn one-hot columns (Product_Category_Toys = 1 ...) back into one readable label per row."""
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    return df[cols].idxmax(axis=1).str[len(prefix) + 1:]


# Category and subcategory describe the same thing (every subcategory belongs to exactly one category), so they are
# reported as ONE feature. Separately, each looks unimportant in permutation importance because the other one still
# carries the information when it is shuffled.
MERGED = {"Product_Category": "Product (category + subcategory)", "Product_Subcategory": "Product (category + subcategory)"}


def feature_groups(columns):
    """{original feature: [model columns]}. One-hot columns are grouped back so importance is per real feature."""
    groups = {}
    for c in columns:
        base = next((cat for cat in CAT_COLS if c.startswith(cat + "_")), c)
        groups.setdefault(MERGED.get(base, base), []).append(c)
    return groups


# ─────────────────────────────────────────────
# 2. METRICS (metrics() and predict() are imported from evaluation.py, see the top of the file)
# ─────────────────────────────────────────────
def cv_scores(params, X, y, n_splits=CV_SPLITS):
    """Time-series cross-validation: fold k trains on everything before block k and is scored on block k."""
    rows = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = XGBRegressor(**params, **FIXED).fit(X.iloc[tr], y.iloc[tr])
        rows.append(metrics(y.iloc[va], predict(m, X.iloc[va])))
    cv = pd.DataFrame(rows)
    return {f"cv_{k}_{s}": float(getattr(cv[k], s)()) for k in ("mae", "mse", "rmse", "r2") for s in ("mean", "std")}


# ─────────────────────────────────────────────
# 3. HYPER-PARAMETER TUNING (Optuna)
# ─────────────────────────────────────────────
def suggest_params(trial):
    """Search space. Ranges follow common XGBoost practice; the comments say what each setting controls."""
    p = {
        # Loss function. Units sold are counts that are right-skewed, so Poisson/Tweedie losses are tried
        # against plain squared error. Optuna decides with the CV score, not us.
        "objective": trial.suggest_categorical("objective", ["reg:squarederror", "count:poisson", "reg:tweedie"]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),     # number of trees
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),  # step size of each tree (small = careful)
        "max_depth": trial.suggest_int("max_depth", 2, 7),                         # tree depth = interaction order (over-fitting)
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 30, log=True),  # min data in a leaf (smooths noisy leaves)
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),                   # share of rows per tree (like bagging)
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),     # share of columns per tree
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),                           # min loss gain needed to split
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),       # L1 regularisation
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),     # L2 regularisation
    }
    if p["objective"] == "reg:tweedie":
        p["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.05, 1.95)
    return p


def tune(X, y, n_trials, n_splits):
    """Find the settings with the lowest time-series CV RMSE. The test set is never touched here."""
    splits = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    def objective(trial):
        params, scores = suggest_params(trial), []
        for step, (tr, va) in enumerate(splits):
            m = XGBRegressor(**params, **FIXED).fit(X.iloc[tr], y.iloc[tr])
            scores.append(metrics(y.iloc[va], predict(m, X.iloc[va]))["rmse"])
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():                      # clearly worse than the median trial -> stop early, save time
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))

    def progress(study, trial):
        if (trial.number + 1) % 10 == 0 or trial.number + 1 == n_trials:
            print(f"   trial {trial.number + 1:>3}/{n_trials}  best CV RMSE so far: {study.best_value:.4f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[progress])
    return study


# ─────────────────────────────────────────────
# 4. INTERPRETATION
# ─────────────────────────────────────────────
def grouped_permutation(model, X, y, groups, n_repeats=5):
    """How much worse does RMSE get when one real feature is shuffled? (all its one-hot columns shuffled together)"""
    rng = np.random.default_rng(SEED)
    base = metrics(y, predict(model, X))["rmse"]
    out = {}
    for g, cols in groups.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            deltas.append(metrics(y, predict(model, Xp))["rmse"] - base)
        out[g] = (float(np.mean(deltas)), float(np.std(deltas)))
    return out


def shap_values(model, X):
    """SHAP values from XGBoost itself (pred_contribs): how many units each column added/removed per row.
    Last column is the base value. For Poisson/Tweedie the contributions are on the log scale."""
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return contrib[:, :-1]


def effect_curve(model, X, col, grid):
    """Partial dependence: average prediction when `col` is forced to each value in `grid`, all else unchanged."""
    vals = []
    for v in grid:
        Xv = X.copy()
        Xv[col] = v
        vals.append(float(predict(model, Xv).mean()))
    return np.array(vals)


def scaler_info():
    """Mean/std the pipeline used to standardise numeric columns, so charts can show real units (RM, visitors ...)."""
    try:
        sc = joblib.load(PREPROCESSOR_FILE).named_transformers_["num"]
        return dict(zip(NUM_COLS, zip(sc.mean_, sc.scale_)))
    except Exception:
        return {}


# ─────────────────────────────────────────────
# 5. ERROR ANALYSIS
# ─────────────────────────────────────────────
def segment_table(test, y, pred):
    """MAE / RMSE / bias per group: where does the model do well, and where does it struggle?"""
    seg = {c: decode(test, c) for c in SEGMENT_CATS}
    for flag in FLAG_COLS:
        seg[flag] = test[flag].map({0: "No", 1: "Yes"})
    seg["Test month"] = test[DATE_COL].dt.strftime("%Y-%m")
    # Actual demand band: reveals the typical regression-to-the-mean error (big sales days are under-predicted).
    seg["Actual demand band"] = pd.qcut(pd.Series(y).rank(method="first"), 4,
                                        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]).astype(str)
    rows = []
    for name, labels in seg.items():
        labels = pd.Series(np.asarray(labels), index=range(len(y)))
        for g in sorted(labels.unique()):
            idx = labels[labels == g].index.to_numpy()
            m = metrics(y[idx], pred[idx])
            rows.append({"segment": name, "group": g, "n": len(idx), "mean_actual": float(y[idx].mean()), **m})
    return pd.DataFrame(rows)


def walk_forward(params, train, test, features):
    """Does the model adapt over time? Cut the test period into blocks; for each block retrain on everything
    that happened before it (train + earlier blocks) and predict the block. Compared with the one-off model
    that was trained once and never updated."""
    dates = np.sort(test[DATE_COL].unique())
    rows = []
    for k, block in enumerate(np.array_split(dates, WALK_FORWARD_BLOCKS), start=1):
        in_block = test[DATE_COL].isin(block)
        history = pd.concat([train, test[test[DATE_COL] < block.min()]])
        m = XGBRegressor(**params, **FIXED).fit(history[features], history[TARGET])
        rows.append({"block": k, "from": str(pd.Timestamp(block.min()).date()), "to": str(pd.Timestamp(block.max()).date()),
                     "n": int(in_block.sum()), "train_rows": len(history), "mask": in_block.to_numpy(),
                     "retrained": metrics(test.loc[in_block, TARGET], predict(m, test.loc[in_block, features]))})
    return rows


# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────
def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name)
    plt.close(fig)


def plot_all(ctx):
    """Draw every figure from the results collected in train_xgboost (ctx = dict of results)."""
    y, p, test = ctx["y"], ctx["pred"], ctx["test"]

    # Default vs tuned, the four required metrics side by side
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, k in zip(axes, ["mae", "mse", "rmse", "r2"]):
        vals = [ctx["default_test"][k], ctx["tuned_test"][k]]
        ax.bar(["Default", "Tuned"], vals, color=[GREY, BLUE])
        ax.set_title(k.upper() if k != "r2" else "R²")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if k == "r2" else f"{v:.1f}", ha="center", va="bottom")
        ax.set_ylim(0, max(vals) * 1.15)
    fig.suptitle("XGBoost on the test period: default vs tuned")
    save(fig, "default_vs_tuned.png")

    # Predicted vs actual + weekly totals over time (what a planner sees)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y, p, s=4, alpha=0.25, color=BLUE)
    lim = max(y.max(), p.max())
    axes[0].plot([0, lim], [0, lim], color=RED, lw=1, label="perfect prediction")
    axes[0].set(xlabel="Actual units sold", ylabel="Predicted units sold", title="Predicted vs actual (test)")
    axes[0].legend()
    wk = pd.DataFrame({"week": test[DATE_COL].dt.to_period("W").dt.start_time, "actual": y, "pred": p}).groupby("week").sum()
    axes[1].plot(wk.index, wk["actual"], color=GREY, label="Actual")
    axes[1].plot(wk.index, wk["pred"], color=BLUE, label="Predicted")
    axes[1].set(title="Weekly total units sold (test)", ylabel="Units")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)
    save(fig, "predicted_vs_actual.png")

    # Residuals
    res = p - y
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(res, bins=60, color=BLUE, alpha=0.85)
    axes[0].axvline(0, color=RED, lw=1)
    axes[0].set(title=f"Residuals (predicted − actual), mean {res.mean():+.2f}", xlabel="Units")
    axes[1].scatter(p, res, s=4, alpha=0.25, color=BLUE)
    axes[1].axhline(0, color=RED, lw=1)
    axes[1].set(title="Residuals vs predicted", xlabel="Predicted units", ylabel="Error")
    save(fig, "residuals.png")

    # Error by segment
    seg = ctx["segments"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, name in zip(axes.ravel(), SEGMENT_CATS + FLAG_COLS + ["Actual demand band"]):
        s = seg[seg["segment"] == name]
        ax.bar(s["group"], s["mae"], color=BLUE)
        ax.set_title(f"MAE by {name.replace('_', ' ')}")
        ax.tick_params(axis="x", rotation=40)
    save(fig, "error_by_segment.png")

    # Importance: gain, permutation, SHAP (all per real feature)
    imp = ctx["importance"].sort_values("shap_mean_abs")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title, colour in zip(axes, ["gain", "permutation_rmse_increase", "shap_mean_abs"],
                                      ["Gain (built-in)", "Permutation (RMSE increase)", "SHAP (mean |impact|)"],
                                      [BLUE, GREEN, ORANGE]):
        d = imp.sort_values(col)
        ax.barh(d.index, d[col], color=colour)
        ax.set_title(title)
    save(fig, "feature_importance.png")

    # SHAP beeswarm per model column (top 15)
    try:
        import shap
        plt.figure(figsize=(8, 6))
        shap.summary_plot(ctx["shap"], ctx["X_shap"], max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_summary.png")
        plt.close()
    except Exception as err:                                # the numbers are saved anyway; only the picture is skipped
        print(f"   (SHAP picture skipped: {err})")

    # Partial dependence of the numeric inputs, shown in real units
    cols = list(ctx["pdp"])
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3))
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.plot(ctx["pdp"][c]["x"], ctx["pdp"][c]["y"], color=BLUE, marker="o", ms=3)
        ax.set(title=c.replace("_", " "), ylabel="Avg predicted units")
    save(fig, "partial_dependence.png")

    # Training curve: error per boosting round on the training data and on the test period
    ev = ctx["evals"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ev["validation_0"]["rmse"], label="Train", color=BLUE)
    ax.plot(ev["validation_1"]["rmse"], label="Test", color=ORANGE)
    ax.set(xlabel="Boosting round (tree)", ylabel="RMSE", title="Training curve (tuned model)")
    ax.legend()
    save(fig, "training_curves.png")

    # Tuning history and which settings mattered
    study = ctx["study"]
    done = [t for t in study.trials if t.value is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].scatter([t.number for t in done], [t.value for t in done], s=14, color=GREY, label="trial")
    axes[0].plot([t.number for t in done], np.minimum.accumulate([t.value for t in done]), color=BLUE, label="best so far")
    axes[0].set(xlabel="Trial", ylabel="CV RMSE", title=f"Optuna search ({len(done)} finished, {len(study.trials) - len(done)} pruned)")
    axes[0].legend()
    try:
        pi = optuna.importance.get_param_importances(study)
        axes[1].barh(list(pi)[::-1], list(pi.values())[::-1], color=BLUE)
        axes[1].set_title("Which settings mattered most")
    except Exception:
        axes[1].axis("off")
    save(fig, "tuning_history.png")

    # Walk-forward: static model vs retrained on all data before each block
    wf = ctx["walk_forward"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(wf))
    ax.bar(x - 0.2, [w["static"]["rmse"] for w in wf], 0.4, label="Trained once (train period)", color=GREY)
    ax.bar(x + 0.2, [w["retrained"]["rmse"] for w in wf], 0.4, label="Retrained before each block", color=BLUE)
    ax.set_xticks(x, [f"{w['from'][:7]}\n→ {w['to'][:7]}" for w in wf])
    ax.set(ylabel="RMSE", title="Does it adapt over time? (walk-forward)")
    ax.set_ylim(0, max(w["static"]["rmse"] for w in wf) * 1.35)
    ax.legend()
    save(fig, "walk_forward.png")


# ─────────────────────────────────────────────
# 7. MAIN: train, tune, evaluate, explain, save
# ─────────────────────────────────────────────
def train_xgboost(n_trials=N_TRIALS, cv_splits=CV_SPLITS, quick=False):
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if quick:                                  # smoke test only (checks the code runs); never use these numbers
        n_trials, cv_splits = 8, 3

    train, test, features = load_data()
    Xtr, ytr, Xte, yte = train[features], train[TARGET], test[features], test[TARGET]
    groups = feature_groups(features)
    print(f"Data: {len(train):,} train rows ({train[DATE_COL].min().date()} to {train[DATE_COL].max().date()}), "
          f"{len(test):,} test rows ({test[DATE_COL].min().date()} to {test[DATE_COL].max().date()}), "
          f"{len(features)} model inputs from {len(groups)} features.")

    # 1. Reference points. If XGBoost cannot beat these, it is not adding value.
    print(LINE + "\n1. Reference points")
    ref = {"mean_predictor": metrics(yte, np.full(len(yte), ytr.mean())),
           "single_decision_tree": metrics(yte, DecisionTreeRegressor(max_depth=6, random_state=SEED).fit(Xtr, ytr).predict(Xte))}
    for k, v in ref.items():
        print(f"   {k:<22} MAE {v['mae']:.3f}  RMSE {v['rmse']:.3f}  R2 {v['r2']:.4f}")

    # 2. Default XGBoost
    print(LINE + "\n2. Default XGBoost")
    t = time.time()
    default_cv = cv_scores({}, Xtr, ytr, cv_splits)
    default = XGBRegressor(**FIXED).fit(Xtr, ytr)
    default_res = {"train": metrics(ytr, predict(default, Xtr)), "test": metrics(yte, predict(default, Xte)), **default_cv,
                   "params": {}, "seconds": round(time.time() - t, 1)}
    print(f"   CV RMSE {default_cv['cv_rmse_mean']:.3f} ± {default_cv['cv_rmse_std']:.3f} | "
          f"test MAE {default_res['test']['mae']:.3f}  RMSE {default_res['test']['rmse']:.3f}  R2 {default_res['test']['r2']:.4f}")

    # 3. Tuning
    print(LINE + f"\n3. Tuning with Optuna ({n_trials} trials, {cv_splits}-fold time-series CV)")
    t = time.time()
    study = tune(Xtr, ytr, n_trials, cv_splits)
    best = study.best_params
    tuned_cv = cv_scores(best, Xtr, ytr, cv_splits)
    tune_seconds = round(time.time() - t, 1)
    print(f"   Best settings: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()})}")

    # 4. Final model: fit on the whole training period, score once on the test period.
    print(LINE + "\n4. Final tuned model")
    t = time.time()
    model = XGBRegressor(**best, **FIXED).fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], verbose=False)
    evals = model.evals_result()          # only logged for the training-curve plot; NOT used for early stopping or tuning
    pred = predict(model, Xte)
    tuned_res = {"train": metrics(ytr, predict(model, Xtr)), "test": metrics(yte, pred), **tuned_cv, "best_params": best,
                 "seconds": round(time.time() - t + tune_seconds, 1)}
    tt, dt = tuned_res["test"], default_res["test"]
    print(f"   CV RMSE {tuned_cv['cv_rmse_mean']:.3f} ± {tuned_cv['cv_rmse_std']:.3f} | "
          f"test MAE {tt['mae']:.3f}  MSE {tt['mse']:.2f}  RMSE {tt['rmse']:.3f}  R2 {tt['r2']:.4f}  WAPE {tt['wape']:.1%}")
    print(f"   vs default: RMSE {dt['rmse']:.3f} -> {tt['rmse']:.3f} ({(tt['rmse'] / dt['rmse'] - 1):+.1%}), "
          f"R2 {dt['r2']:.4f} -> {tt['r2']:.4f}")
    y_arr = yte.to_numpy()

    # 5. Feature importance, three ways (they should agree; if not, be careful with the story)
    print(LINE + "\n5. Feature importance")
    gain = model.get_booster().get_score(importance_type="gain")
    gain = pd.Series({c: gain.get(c, 0.0) for c in features})
    sample = Xte.sample(min(SHAP_ROWS, len(Xte)), random_state=SEED)
    sv = shap_values(model, sample)
    shap_col = pd.Series(np.abs(sv).mean(axis=0), index=features)
    perm = grouped_permutation(model, Xte, yte, groups)
    importance = pd.DataFrame({
        "gain": {g: gain[c].sum() / gain.sum() for g, c in groups.items()},
        "permutation_rmse_increase": {g: perm[g][0] for g in groups},
        "permutation_std": {g: perm[g][1] for g in groups},
        "shap_mean_abs": {g: shap_col[c].sum() for g, c in groups.items()},
    }).sort_values("permutation_rmse_increase", ascending=False)
    importance.index.name = "feature"
    importance.to_csv(OUT_DIR / "feature_importance.csv")
    pd.DataFrame({"gain": gain, "shap_mean_abs": shap_col}).sort_values("shap_mean_abs", ascending=False) \
        .to_csv(OUT_DIR / "feature_importance_by_column.csv", index_label="column")
    print(importance.round(4).to_string())

    # Business view: what does a promotion / holiday add on average, and how do the numeric inputs act?
    effects, pdp, sc = {}, {}, scaler_info()
    for flag in FLAG_COLS:
        on, off = Xte.copy(), Xte.copy()
        on[flag], off[flag] = 1, 0
        effects[flag] = float(predict(model, on).mean() - predict(model, off).mean())
    print("   Average effect on predicted units sold: " + ", ".join(f"{k} {v:+.1f}" for k, v in effects.items()))
    for c in PDP_COLS:
        grid = np.quantile(sample[c], np.linspace(0.02, 0.98, 15))
        mean, scale = sc.get(c, (0.0, 1.0))
        pdp[c] = {"x": grid * scale + mean, "y": effect_curve(model, sample, c, grid)}

    # 6. Error analysis
    print(LINE + "\n6. Error analysis")
    seg = segment_table(test, y_arr, pred)
    seg.to_csv(OUT_DIR / "segment_errors.csv", index=False)
    err = pred - y_arr
    preds = test[[DATE_COL] + STOCK_COLS].copy()
    preds[DATE_COL] = preds[DATE_COL].dt.date
    for c in SEGMENT_CATS:
        preds[c] = decode(test, c)
    preds = preds.assign(**{TARGET: y_arr, "pred_default": predict(default, Xte), "pred_tuned": pred,
                            "error": err, "abs_error": np.abs(err)})
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    preds.sort_values("abs_error", ascending=False).head(50).to_csv(OUT_DIR / "worst_predictions.csv", index=False)
    top_band = seg[(seg["segment"] == "Actual demand band")].set_index("group")
    error_analysis = {
        "share_under_predicted": float((err < 0).mean()),
        "within_5_units": float((np.abs(err) <= 5).mean()),
        "within_20_percent": float((np.abs(err) <= 0.2 * y_arr).mean()),
        "bias_lowest_demand_quarter": float(top_band.loc["Q1 lowest", "bias"]),
        "bias_highest_demand_quarter": float(top_band.loc["Q4 highest", "bias"]),
        "hardest_category": seg[seg["segment"] == "Product_Category"].sort_values("mae").iloc[-1]["group"],
        "easiest_category": seg[seg["segment"] == "Product_Category"].sort_values("mae").iloc[0]["group"],
    }
    print(f"   {error_analysis['share_under_predicted']:.0%} of rows under-predicted | "
          f"within 5 units: {error_analysis['within_5_units']:.0%} | "
          f"bias on the lowest / highest demand quarter: {error_analysis['bias_lowest_demand_quarter']:+.1f} / "
          f"{error_analysis['bias_highest_demand_quarter']:+.1f} units")
    print(f"   Hardest category (MAE): {error_analysis['hardest_category']}, easiest: {error_analysis['easiest_category']}")

    # 7. Robustness
    print(LINE + "\n7. Robustness")
    seeds = [metrics(yte, predict(XGBRegressor(**{**FIXED, "random_state": s}, **best).fit(Xtr, ytr), Xte)) for s in range(5)]
    seeds = pd.DataFrame(seeds)
    seed_stability = {f"{k}_{s}": float(getattr(seeds[k], s)()) for k in ("mae", "rmse", "r2") for s in ("mean", "std")}
    print(f"   5 random seeds: RMSE {seed_stability['rmse_mean']:.3f} ± {seed_stability['rmse_std']:.3f}, "
          f"R2 {seed_stability['r2_mean']:.4f} ± {seed_stability['r2_std']:.4f}")
    wf = walk_forward(best, train, test, features)
    for w in wf:
        w["static"] = metrics(y_arr[w["mask"]], pred[w["mask"]])
        print(f"   {w['from']} → {w['to']}: static RMSE {w['static']['rmse']:.3f} | retrained RMSE {w['retrained']['rmse']:.3f}")
    wf_json = [{k: v for k, v in w.items() if k != "mask"} for w in wf]

    # 8. Save everything
    print(LINE + "\n8. Saving")
    plot_all({"y": y_arr, "pred": pred, "test": test, "default_test": dt, "tuned_test": tt, "segments": seg,
              "importance": importance, "shap": sv, "X_shap": sample, "pdp": pdp, "evals": evals, "study": study,
              "walk_forward": wf})
    trials = study.trials_dataframe()
    trials.to_csv(OUT_DIR / "tuning_trials.csv", index=False)
    joblib.dump(model, OUT_DIR / "model.joblib", compress=3)
    joblib.dump(default, OUT_DIR / "model_default.joblib", compress=3)
    (OUT_DIR / "best_params.json").write_text(json.dumps(best, indent=2))
    result = {
        "model": "XGBoost", "task": "regression", "target": TARGET, "quick_run": quick,
        "n_train": len(train), "n_test": len(test), "features": features, "n_model_inputs": len(features),
        "split": f"Train {train[DATE_COL].min().date()} to {train[DATE_COL].max().date()}; "
                 f"test {test[DATE_COL].min().date()} to {test[DATE_COL].max().date()} (latest dates held out)",
        "validation": f"TimeSeriesSplit, {cv_splits} folds on the training period",
        "tuning": {"method": "Optuna TPE + median pruner", "n_trials": n_trials,
                   "n_pruned": int(sum(t.state.name == "PRUNED" for t in study.trials)), "best_cv_rmse": study.best_value},
        "reference": ref, "default": default_res, "tuned": tuned_res,
        "feature_importance": importance.round(6).reset_index().to_dict("records"),
        "effect_on_units": effects, "error_analysis": error_analysis,
        "robustness": {"seed_stability": seed_stability, "walk_forward": wf_json},
        "run": {"seed": SEED, "seconds": round(time.time() - t0, 1), "xgboost": xgb.__version__,
                "python": platform.python_version(), "finished": time.strftime("%Y-%m-%d %H:%M")},
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"   Saved to {OUT_DIR}  ({result['run']['seconds']:.0f}s in total)")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train, tune and evaluate the XGBoost demand regressor")
    ap.add_argument("--trials", type=int, default=N_TRIALS, help=f"Optuna trials (default {N_TRIALS})")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (8 trials, 3 CV folds); not for final results")
    a = ap.parse_args()
    train_xgboost(n_trials=a.trials, quick=a.quick)