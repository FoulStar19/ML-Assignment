import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")                     
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[1]                             
OUT_DIR = HERE / "outputs"

if str(ROOT_DIR / "src" / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src" / "backend"))
from data_modeling import (CAT_COLS, DATE_COL, NUM_COLS, PREPROCESSOR_FILE, STOCK_COLS,   # noqa: E402
                           TARGET, TEST_FILE, TRAIN_FILE)

LINE = "─" * 45
SEED = 42
N_TRIALS = 80            # Optuna trials 
CV_SPLITS = 5            # time-series CV folds: always train on the past, validate on the next block of dates
ENSEMBLE_SIZE = 5        # number of best tuned settings averaged in the ensemble check
WALK_FORWARD_BLOCKS = 4  # the test period is cut into this many blocks for the "does it adapt over time" check
SHAP_ROWS = 2000         # test rows used for SHAP / partial dependence (keeps it fast)
FIXED = dict(tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="rmse")   # same for every model here

MERGED = {"Product_Category": "Product (category + subcategory)",
          "Product_Subcategory": "Product (category + subcategory)"}

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})
BLUE, ORANGE, GREY, GREEN, RED = "#3B5BDB", "#F5A524", "#8A94A6", "#22A27B", "#E5484D"


# ─────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────
def load_data():
    for f in (TRAIN_FILE, TEST_FILE):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found. Run `python src/backend/pipeline.py --skip-modeling` first.")
    train = pd.read_csv(TRAIN_FILE, parse_dates=[DATE_COL])
    test = pd.read_csv(TEST_FILE, parse_dates=[DATE_COL])

    # Model inputs = everything except the target, the date and the stock/ID columns.
    # (Inventory_Level, Returns and Sales_Amount are not inputs: they would leak the answer.)
    non_features = [TARGET, DATE_COL] + STOCK_COLS
    features = [c for c in train.columns if c not in non_features]
    if features != [c for c in test.columns if c not in non_features]:
        raise ValueError("train.csv and test.csv have different feature columns; re-run the pipeline.")

    # Safety checks: the split must be by time with no overlap, otherwise the test score is too optimistic.
    if not train[DATE_COL].is_monotonic_increasing:
        raise ValueError("train.csv is not sorted by date; CV folds would not respect time order.")
    if train[DATE_COL].max() >= test[DATE_COL].min():
        raise ValueError("train and test dates overlap; the split is not a clean time-based split.")
    return train, test, features


def decode(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    return df[cols].idxmax(axis=1).str[len(prefix) + 1:]


def feature_groups(columns):
    groups = {}
    for c in columns:
        base = next((cat for cat in CAT_COLS if c.startswith(cat + "_")), c)
        groups.setdefault(MERGED.get(base, base), []).append(c)
    return groups


# ─────────────────────────────────────────────
# 2. METRICS AND CROSS-VALIDATION
# ─────────────────────────────────────────────
def metrics(y, pred):
   
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    mse = float(np.mean(err ** 2))
    return {"mae": float(np.mean(np.abs(err))), "mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y, pred)), "wape": float(np.abs(err).sum() / y.sum()), "bias": float(err.mean())}


def predict(model, X):
    return np.clip(model.predict(X), 0, None)       # units sold cannot be negative


def new_model(params=None, **extra):
    return XGBRegressor(**(params or {}), **{**FIXED, **extra})


def cv_scores(params, X, y, n_splits=CV_SPLITS):
    rows = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = new_model(params).fit(X.iloc[tr], y.iloc[tr])
        rows.append(metrics(y.iloc[va], predict(m, X.iloc[va])))
    cv = pd.DataFrame(rows)
    return {f"cv_{k}_{s}": float(getattr(cv[k], s)()) for k in ("mae", "mse", "rmse", "r2") for s in ("mean", "std")}


# ─────────────────────────────────────────────
# 3. HYPER-PARAMETER TUNING (Optuna)
# ─────────────────────────────────────────────
def suggest_params(trial):
    p = {
        # Loss function: units sold are right-skewed counts, so Poisson / Tweedie are tried against squared error.
        "objective": trial.suggest_categorical("objective", ["reg:squarederror", "count:poisson", "reg:tweedie"]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=50),         # number of trees
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),    # step size of each tree
        "max_depth": trial.suggest_int("max_depth", 1, 6),                             # interaction order / over-fitting
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),  # min data in a leaf
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),                       # share of rows per tree
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),         # share of columns per tree
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),                              # min loss gain needed to split
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),           # L1 regularisation
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),         # L2 regularisation
    }
    if p["objective"] == "reg:tweedie":
        p["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.05, 1.95)
    return p


def tune(X, y, n_trials, n_splits):
    splits = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    def objective(trial):
        params, scores = suggest_params(trial), []
        for step, (tr, va) in enumerate(splits):
            m = new_model(params).fit(X.iloc[tr], y.iloc[tr])
            scores.append(metrics(y.iloc[va], predict(m, X.iloc[va]))["rmse"])
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():                      # clearly worse than the median trial -> stop early
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    def progress(study, trial):
        if (trial.number + 1) % 10 == 0 or trial.number + 1 == n_trials:
            print(f"   trial {trial.number + 1:>3}/{n_trials}  best CV RMSE so far: {study.best_value:.4f}")

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True, n_startup_trials=20),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials, callbacks=[progress])
    return study


def ensemble_of_best_trials(study, Xtr, ytr, Xte, yte, k=ENSEMBLE_SIZE):
    finished = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)[:k]
    preds = [predict(new_model(t.params).fit(Xtr, ytr), Xte) for t in finished]
    return {"members": len(finished), "test": metrics(yte, np.mean(preds, axis=0))}


# ─────────────────────────────────────────────
# 4. INTERPRETATION
# ─────────────────────────────────────────────
def grouped_permutation(model, X, y, groups, n_repeats=5):
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
    return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


def effect_curve(model, X, col, grid):
    vals = []
    for v in grid:
        Xv = X.copy()
        Xv[col] = v
        vals.append(float(predict(model, Xv).mean()))
    return np.array(vals)


def scaler_info():
    try:
        sc = joblib.load(PREPROCESSOR_FILE).named_transformers_["num"]
        return dict(zip(NUM_COLS, zip(sc.mean_, sc.scale_)))
    except Exception:
        return {}


def importance_table(model, Xte, yte, features, groups):
    gain = model.get_booster().get_score(importance_type="gain")
    gain = pd.Series({c: gain.get(c, 0.0) for c in features})
    sample = Xte.sample(min(SHAP_ROWS, len(Xte)), random_state=SEED)
    sv = shap_values(model, sample)
    shap_col = pd.Series(np.abs(sv).mean(axis=0), index=features)
    perm = grouped_permutation(model, Xte, yte, groups)
    table = pd.DataFrame({
        "gain": {g: gain[c].sum() / gain.sum() for g, c in groups.items()},
        "permutation_rmse_increase": {g: perm[g][0] for g in groups},
        "permutation_std": {g: perm[g][1] for g in groups},
        "shap_mean_abs": {g: shap_col[c].sum() for g, c in groups.items()},
    }).sort_values("permutation_rmse_increase", ascending=False)
    table.index.name = "feature"
    by_column = pd.DataFrame({"gain": gain, "shap_mean_abs": shap_col}).sort_values("shap_mean_abs", ascending=False)
    return table, by_column, sample, sv


def business_effects(model, Xte, sample):
    effects, pdp, sc = {}, {}, scaler_info()
    for flag in ("Promotion", "Holiday"):
        on, off = Xte.copy(), Xte.copy()
        on[flag], off[flag] = 1, 0
        effects[flag] = float(predict(model, on).mean() - predict(model, off).mean())
    for c in ["Unit_Price", "Marketing_Spend", "Customer_Footfall", "Discount_Percent"]:
        grid = np.quantile(sample[c], np.linspace(0.02, 0.98, 15))
        mean, scale = sc.get(c, (0.0, 1.0))
        pdp[c] = {"x": grid * scale + mean, "y": effect_curve(model, sample, c, grid)}
    return effects, pdp


# ─────────────────────────────────────────────
# 5. ERROR ANALYSIS
# ─────────────────────────────────────────────
def segment_table(test, y, pred):
    seg = {c: decode(test, c) for c in ["Product_Category", "Store_Type", "Weather_Condition"]}
    seg["Promotion"] = test["Promotion"].map({0: "No", 1: "Yes"})
    seg["Holiday"] = test["Holiday"].map({0: "No", 1: "Yes"})
    seg["Test month"] = test[DATE_COL].dt.strftime("%Y-%m")
    # Actual demand band: reveals the usual regression-to-the-mean error (big sales days are under-predicted).
    seg["Actual demand band"] = pd.qcut(pd.Series(y).rank(method="first"), 4,
                                        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]).astype(str)
    rows = []
    for name, labels in seg.items():
        labels = pd.Series(np.asarray(labels), index=range(len(y)))
        for g in sorted(labels.unique()):
            idx = labels[labels == g].index.to_numpy()
            rows.append({"segment": name, "group": g, "n": len(idx), "mean_actual": float(y[idx].mean()),
                         **metrics(y[idx], pred[idx])})
    return pd.DataFrame(rows)


def error_summary(seg, y, pred):
    err = pred - y
    band = seg[seg["segment"] == "Actual demand band"].set_index("group")
    by_cat = seg[seg["segment"] == "Product_Category"].sort_values("mae")
    return {
        "share_under_predicted": float((err < 0).mean()),
        "within_5_units": float((np.abs(err) <= 5).mean()),
        "within_20_percent": float((np.abs(err) <= 0.2 * y).mean()),
        "bias_lowest_demand_quarter": float(band.loc["Q1 lowest", "bias"]),
        "bias_highest_demand_quarter": float(band.loc["Q4 highest", "bias"]),
        "hardest_category": by_cat.iloc[-1]["group"],
        "easiest_category": by_cat.iloc[0]["group"],
    }


def save_predictions(test, y, default_pred, pred):
    err = pred - y
    preds = pd.DataFrame({"Date": test[DATE_COL].dt.date, "Transaction_ID": test["Transaction_ID"],
                          "Product_ID": test["Product_ID"], "Product_Category": decode(test, "Product_Category"),
                          "Store_Type": decode(test, "Store_Type"), "Units_Sold": y, "pred_default": default_pred,
                          "pred_tuned": pred, "error": err, "abs_error": np.abs(err)})
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    preds.sort_values("abs_error", ascending=False).head(50).to_csv(OUT_DIR / "worst_predictions.csv", index=False)


# ─────────────────────────────────────────────
# 6. ROBUSTNESS
# ─────────────────────────────────────────────
def seed_stability(params, Xtr, ytr, Xte, yte, n_seeds=5):
    rows = [metrics(yte, predict(new_model(params, random_state=s).fit(Xtr, ytr), Xte)) for s in range(n_seeds)]
    df = pd.DataFrame(rows)
    return {f"{k}_{s}": float(getattr(df[k], s)()) for k in ("mae", "rmse", "r2") for s in ("mean", "std")}


def walk_forward(params, train, test, features, static_pred):
    y = test[TARGET].to_numpy()
    rows = []
    for k, block in enumerate(np.array_split(np.sort(test[DATE_COL].unique()), WALK_FORWARD_BLOCKS), start=1):
        in_block = test[DATE_COL].isin(block).to_numpy()
        history = pd.concat([train, test[test[DATE_COL] < block.min()]])
        m = new_model(params).fit(history[features], history[TARGET])
        rows.append({"block": k, "from": str(pd.Timestamp(block.min()).date()), "to": str(pd.Timestamp(block.max()).date()),
                     "n": int(in_block.sum()), "train_rows": len(history),
                     "retrained": metrics(y[in_block], predict(m, test.loc[in_block, features])),
                     "static": metrics(y[in_block], static_pred[in_block])})
    return rows


# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name)
    plt.close(fig)


def plot_default_vs_tuned(c):
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, k in zip(axes, ["mae", "mse", "rmse", "r2"]):
        vals = [c["default_test"][k], c["tuned_test"][k]]
        ax.bar(["Default", "Tuned"], vals, color=[GREY, BLUE])
        ax.set_title("R²" if k == "r2" else k.upper())
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if k == "r2" else f"{v:.1f}", ha="center", va="bottom")
        ax.set_ylim(0, max(vals) * 1.15)
    fig.suptitle("XGBoost on the test period: default vs tuned")
    save(fig, "default_vs_tuned.png")


def plot_predicted_vs_actual(c):
    y, p, test = c["y"], c["pred"], c["test"]
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


def plot_residuals(c):
    res = c["pred"] - c["y"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(res, bins=60, color=BLUE, alpha=0.85)
    axes[0].axvline(0, color=RED, lw=1)
    axes[0].set(title=f"Residuals (predicted − actual), mean {res.mean():+.2f}", xlabel="Units")
    axes[1].scatter(c["pred"], res, s=4, alpha=0.25, color=BLUE)
    axes[1].axhline(0, color=RED, lw=1)
    axes[1].set(title="Residuals vs predicted", xlabel="Predicted units", ylabel="Error")
    save(fig, "residuals.png")


def plot_error_by_segment(c):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, name in zip(axes.ravel(), ["Product_Category", "Store_Type", "Weather_Condition",
                                       "Promotion", "Holiday", "Actual demand band"]):
        s = c["segments"][c["segments"]["segment"] == name]
        ax.bar(s["group"], s["mae"], color=BLUE)
        ax.set_title(f"MAE by {name.replace('_', ' ')}")
        ax.tick_params(axis="x", rotation=40)
    save(fig, "error_by_segment.png")


def plot_feature_importance(c):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title, colour in zip(axes, ["gain", "permutation_rmse_increase", "shap_mean_abs"],
                                      ["Gain (built-in)", "Permutation (RMSE increase)", "SHAP (mean |impact|)"],
                                      [BLUE, GREEN, ORANGE]):
        d = c["importance"].sort_values(col)
        ax.barh(d.index, d[col], color=colour)
        ax.set_title(title)
    save(fig, "feature_importance.png")


def plot_shap_summary(c):
    import shap                                           # optional library: skipped by plot_all if missing
    plt.figure(figsize=(8, 6))
    shap.summary_plot(c["shap"], c["X_shap"], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png")
    plt.close()


def plot_partial_dependence(c):
    cols = list(c["pdp"])
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3))
    for ax, col in zip(np.atleast_1d(axes), cols):
        ax.plot(c["pdp"][col]["x"], c["pdp"][col]["y"], color=BLUE, marker="o", ms=3)
        ax.set(title=col.replace("_", " "), ylabel="Avg predicted units")
    save(fig, "partial_dependence.png")


def plot_training_curves(c):
    ev = c["evals"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ev["validation_0"]["rmse"], label="Train", color=BLUE)
    ax.plot(ev["validation_1"]["rmse"], label="Test", color=ORANGE)
    ax.set(xlabel="Boosting round (tree)", ylabel="RMSE", title="Training curve (tuned model)")
    ax.legend()
    save(fig, "training_curves.png")


def plot_tuning_history(c):
    study = c["study"]
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


def plot_walk_forward(c):
    wf = c["walk_forward"]
    x = np.arange(len(wf))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(x - 0.2, [w["static"]["rmse"] for w in wf], 0.4, label="Trained once (train period)", color=GREY)
    ax.bar(x + 0.2, [w["retrained"]["rmse"] for w in wf], 0.4, label="Retrained before each block", color=BLUE)
    ax.set_xticks(x, [f"{w['from'][:7]}\n→ {w['to'][:7]}" for w in wf])
    ax.set(ylabel="RMSE", title="Does it adapt over time? (walk-forward)")
    ax.set_ylim(0, max(w["static"]["rmse"] for w in wf) * 1.35)
    ax.legend()
    save(fig, "walk_forward.png")


PLOTS = [plot_default_vs_tuned, plot_predicted_vs_actual, plot_residuals, plot_error_by_segment,
         plot_feature_importance, plot_shap_summary, plot_partial_dependence, plot_training_curves,
         plot_tuning_history, plot_walk_forward]


def plot_all(ctx):
    for fn in PLOTS:
        try:
            fn(ctx)
        except Exception as err:
            plt.close("all")
            print(f"   ({fn.__name__} skipped: {err})")


# ─────────────────────────────────────────────
# 8. MAIN: train, tune, evaluate, explain, save
# ─────────────────────────────────────────────
def train_xgboost(n_trials=N_TRIALS, cv_splits=CV_SPLITS, quick=False):
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if quick:                                  # smoke test only (checks the code runs); never use these numbers
        n_trials, cv_splits = 8, 3

    train, test, features = load_data()
    Xtr, ytr, Xte, yte = train[features], train[TARGET], test[features], test[TARGET]
    y_arr, groups = yte.to_numpy(), feature_groups(features)
    print(f"Data: {len(train):,} train rows ({train[DATE_COL].min().date()} to {train[DATE_COL].max().date()}), "
          f"{len(test):,} test rows ({test[DATE_COL].min().date()} to {test[DATE_COL].max().date()}), "
          f"{len(features)} model inputs from {len(groups)} features.")

    # 1. Reference points. If XGBoost cannot beat these, it is not adding value.
    print(LINE + "\n1. Reference points")
    tree = DecisionTreeRegressor(max_depth=6, random_state=SEED).fit(Xtr, ytr)
    ref = {"mean_predictor": metrics(yte, np.full(len(yte), ytr.mean())),
           "single_decision_tree": metrics(yte, tree.predict(Xte))}
    for k, v in ref.items():
        print(f"   {k:<22} MAE {v['mae']:.3f}  RMSE {v['rmse']:.3f}  R2 {v['r2']:.4f}")

    # 2. Default XGBoost
    print(LINE + "\n2. Default XGBoost")
    t = time.time()
    default_cv = cv_scores({}, Xtr, ytr, cv_splits)
    default = new_model().fit(Xtr, ytr)
    default_res = {"train": metrics(ytr, predict(default, Xtr)), "test": metrics(yte, predict(default, Xte)),
                   **default_cv, "params": {}, "seconds": round(time.time() - t, 1)}
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
    model = new_model(best).fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], verbose=False)
    evals = model.evals_result()          # only logged for the training-curve plot; NOT used for early stopping or tuning
    pred = predict(model, Xte)
    tuned_res = {"train": metrics(ytr, predict(model, Xtr)), "test": metrics(yte, pred), **tuned_cv,
                 "best_params": best, "seconds": round(time.time() - t + tune_seconds, 1)}
    tt, dt = tuned_res["test"], default_res["test"]
    print(f"   CV RMSE {tuned_cv['cv_rmse_mean']:.3f} ± {tuned_cv['cv_rmse_std']:.3f} | "
          f"test MAE {tt['mae']:.3f}  MSE {tt['mse']:.2f}  RMSE {tt['rmse']:.3f}  R2 {tt['r2']:.4f}  WAPE {tt['wape']:.1%}")
    print(f"   vs default: RMSE {dt['rmse']:.3f} -> {tt['rmse']:.3f} ({(tt['rmse'] / dt['rmse'] - 1):+.1%}), "
          f"R2 {dt['r2']:.4f} -> {tt['r2']:.4f}")
    ensemble = ensemble_of_best_trials(study, Xtr, ytr, Xte, yte)
    et = ensemble["test"]
    print(f"   Ensemble of the best {ensemble['members']} settings: RMSE {et['rmse']:.3f}  R2 {et['r2']:.4f} "
          f"(single tuned model RMSE {tt['rmse']:.3f})")

    # 5. Feature importance (three ways) and business view
    print(LINE + "\n5. Feature importance")
    importance, by_column, sample, sv = importance_table(model, Xte, yte, features, groups)
    importance.to_csv(OUT_DIR / "feature_importance.csv")
    by_column.to_csv(OUT_DIR / "feature_importance_by_column.csv", index_label="column")
    print(importance.round(4).to_string())
    effects, pdp = business_effects(model, Xte, sample)
    print("   Average effect on predicted units sold: " + ", ".join(f"{k} {v:+.1f}" for k, v in effects.items()))

    # 6. Error analysis
    print(LINE + "\n6. Error analysis")
    seg = segment_table(test, y_arr, pred)
    seg.to_csv(OUT_DIR / "segment_errors.csv", index=False)
    save_predictions(test, y_arr, predict(default, Xte), pred)
    error_analysis = error_summary(seg, y_arr, pred)
    ea = error_analysis
    print(f"   {ea['share_under_predicted']:.0%} of rows under-predicted | within 5 units: {ea['within_5_units']:.0%} | "
          f"bias on the lowest / highest demand quarter: {ea['bias_lowest_demand_quarter']:+.1f} / "
          f"{ea['bias_highest_demand_quarter']:+.1f} units")
    print(f"   Hardest category (MAE): {ea['hardest_category']}, easiest: {ea['easiest_category']}")

    # 7. Robustness
    print(LINE + "\n7. Robustness")
    stability = seed_stability(best, Xtr, ytr, Xte, yte)
    print(f"   5 random seeds: RMSE {stability['rmse_mean']:.3f} ± {stability['rmse_std']:.3f}, "
          f"R2 {stability['r2_mean']:.4f} ± {stability['r2_std']:.4f}")
    wf = walk_forward(best, train, test, features, pred)
    for w in wf:
        print(f"   {w['from']} → {w['to']}: static RMSE {w['static']['rmse']:.3f} | retrained RMSE {w['retrained']['rmse']:.3f}")

    # 8. Save everything (file names are read by xgboost_results.py, so keep them as they are)
    print(LINE + "\n8. Saving")
    plot_all({"y": y_arr, "pred": pred, "test": test, "default_test": dt, "tuned_test": tt, "segments": seg,
              "importance": importance, "shap": sv, "X_shap": sample, "pdp": pdp, "evals": evals, "study": study,
              "walk_forward": wf})
    study.trials_dataframe().to_csv(OUT_DIR / "tuning_trials.csv", index=False)
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
        "reference": ref, "default": default_res, "tuned": tuned_res, "ensemble": ensemble,
        "feature_importance": importance.round(6).reset_index().to_dict("records"),
        "effect_on_units": effects, "error_analysis": error_analysis,
        "robustness": {"seed_stability": stability, "walk_forward": wf},
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
