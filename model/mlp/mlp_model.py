import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                      # no display needed (server / UI backend safe)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")
# n_jobs=-1 runs work in separate processes (spawned on Windows) that do NOT inherit the filter above;
# an environment variable IS inherited, so the workers stay quiet too.
os.environ["PYTHONWARNINGS"] = "ignore"

LINE = "─" * 45

# ─────────────────────────────────────────────
# 0. SETTINGS
# ─────────────────────────────────────────────
SEED = 42
# src/backend/model/mlp/<this file>  -> project root is 4 levels up. PROJECT_ROOT can override it.
ROOT_DIR = (Path(os.environ["PROJECT_ROOT"]) if "PROJECT_ROOT" in os.environ
            else Path(__file__).resolve().parents[2])
TRAIN_FILE = ROOT_DIR / "data" / "processed" / "train.csv"
TEST_FILE = ROOT_DIR / "data" / "processed" / "test.csv"
MODEL_FILE = Path(__file__).resolve().parent / "mlp_model.joblib"

REPORT_DIR = Path(__file__).resolve().parent / "outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Units_Sold"
# Not model inputs (Inventory_Level is kept for the recommender).
NON_FEATURES = [TARGET, "Date", "Transaction_ID", "Store_ID", "Product_ID", "Inventory_Level"]

N_SPLITS = 5            # time-ordered CV folds
N_ITER_SEARCH = 30      # random-search budget (training a network is slower, so random > grid)
N_ENSEMBLE = 5          # MLPs with different seeds averaged together (Part B.2)
N_BOOTSTRAP = 500       # resamples for the confidence intervals on test error
GROUP_PREFIXES = ["Product_Category_", "Store_Type_", "Weather_Condition_"]
PDP_FEATURES = ["Log_Effective_Price", "Unit_Price", "Customer_Footfall", "Marketing_Spend", "Month"]   # curves to draw


# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────
def load_split(path):
    """Read a prepared split (already encoded + scaled by data_modeling.py)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run pipeline.py first")
    df = pd.read_csv(path)
    return df.drop(columns=NON_FEATURES, errors="ignore"), df[TARGET], df["Date"]


def decode_group(X, prefix):
    """Turn a one-hot family (e.g. Product_Category_*) back into one readable label per row."""
    cols = [c for c in X.columns if c.startswith(prefix)]
    return X[cols].idxmax(axis=1).str.replace(prefix, "", regex=False)


def score_all(y_true, y_pred):
    """Regression metrics. Predictions are clipped at 0."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),                  # average miss, in units
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),   # punishes big misses more
        "MedAE": median_absolute_error(y_true, y_pred),              # robust to a few huge misses
        "R2": r2_score(y_true, y_pred),                              # share of variance explained
        "sMAPE_%": 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)),
        "Bias": float(np.mean(y_pred - y_true)),                     # >0 = over-forecasting on average
    }


def make_model(hidden=(64, 32), activation="relu", alpha=1e-4, lr=1e-3, batch=128,
               target="standard", seed=SEED, early_stopping=True, max_iter=500):
    """
    One MLP regressor wrapped so the TARGET is scaled too.
    Why scale the target?  An MLP's output layer starts near 0 and learns best when the target is
    around 0 +/- 1. Raw Units_Sold (mean 26, max 138) makes training slow and unstable.
      target = "raw"      : no scaling (plain sklearn default - kept as the V0 comparison)
      target = "standard" : (y - mean) / std
      target = "log"      : log(1 + y) then standardise (tames the long right tail of sales)
    Predictions are converted back to units automatically.
    Early stopping holds back 10% of the TRAINING rows, stops when it stops improving and keeps
    the best weights - a built-in guard against over-fitting.
    """
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation=activation, alpha=alpha,
                       learning_rate_init=lr, batch_size=batch, max_iter=max_iter,
                       early_stopping=early_stopping, validation_fraction=0.1, n_iter_no_change=15,
                       random_state=seed)
    if target == "raw":
        return TransformedTargetRegressor(regressor=mlp)
    if target == "standard":
        return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())
    if target == "log":
        tt = Pipeline([("log", FunctionTransformer(np.log1p, np.expm1, validate=False)),
                       ("std", StandardScaler())])
        return TransformedTargetRegressor(regressor=mlp, transformer=tt)
    raise ValueError(target)


class SeedEnsemble(BaseEstimator, RegressorMixin):
    """Average of several identical MLPs that differ only in their random start (Part B.2).
    A single network depends on its random initial weights; averaging cancels that luck, which
    lowers the variance of the predictions without adding bias."""

    def __init__(self, base=None, n_members=N_ENSEMBLE):
        self.base = base
        self.n_members = n_members

    def fit(self, X, y):
        self.models_ = [clone(self.base).set_params(regressor__random_state=SEED + i).fit(X, y)
                        for i in range(self.n_members)]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


def cv_scores(model, X, y, cv):
    res = cross_validate(clone(model), X, y, cv=cv, n_jobs=-1,
                         scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])
    return {"cv_RMSE": -res["test_neg_root_mean_squared_error"].mean(),
            "cv_RMSE_std": res["test_neg_root_mean_squared_error"].std(),
            "cv_MAE": -res["test_neg_mean_absolute_error"].mean(),
            "cv_R2": res["test_r2"].mean()}


def partial_dependence(model, X, col, n=12):
    """Average prediction as ONE input is moved across its range while all other inputs keep their
    real values. Shows the SHAPE of the effect."""
    grid = np.unique(np.percentile(X[col], np.linspace(5, 95, n)))
    curve = []
    for v in grid:
        Xv = X.copy()
        Xv[col] = v
        curve.append(np.mean(model.predict(Xv)))
    return grid, np.array(curve)


# ─────────────────────────────────────────────
# 2. MLP TRAINING, REFINEMENT AND EVALUATION
# ─────────────────────────────────────────────
def train_mlp():
    np.random.seed(SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the prepared data
    print(LINE)
    print("1. Data")
    X_train, y_train, date_train = load_split(TRAIN_FILE)
    X_test, y_test, date_test = load_split(TEST_FILE)
    cols = list(X_train.columns)
    print(f"Train : {len(X_train):,} rows | Test : {len(X_test):,} rows | Inputs : {X_train.shape[1]}")
    print(f"Train dates {date_train.min()} to {date_train.max()} | Test dates {date_test.min()} to {date_test.max()}")
    print(f"{TARGET}: train mean {y_train.mean():.1f}, max {y_train.max()}, skewness {stats.skew(y_train):.2f}")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing values reached the model - fix them in data_cleaning.py")
    assert date_train.is_monotonic_increasing, "train.csv must be sorted by Date (needed for time-ordered CV)"
    print("Missing values : none (handled upstream in data_cleaning.py: median / mode imputation)")
    print(LINE + "\n")

    cv = TimeSeriesSplit(n_splits=N_SPLITS)    # always validate on dates AFTER the training dates

    # 2. Reference point: "no learning" (always predict the training mean)
    print(LINE)
    print("2. Reference point")
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    dummy_scores = score_all(y_test, dummy.predict(X_test))
    print(f"Mean-only dummy: MAE {dummy_scores['MAE']:.2f}, RMSE {dummy_scores['RMSE']:.2f}, R2 {dummy_scores['R2']:.3f}")
    print("The MLP must clearly beat this. The 4-model comparison is done in Student C's evaluation.")
    print(LINE + "\n")

    # 3. Iterate (A.5 / A.6): one change at a time, judged by CV on TRAIN only
    print(LINE)
    print("3. Refinement experiments (cross-validation on TRAIN only - test stays untouched)")
    variants = {
        "V0 sklearn defaults (100,), raw target": make_model((100,), target="raw", early_stopping=False, max_iter=200),
        "V1 (64,32) + standardised target": make_model((64, 32), target="standard"),
        "V2 (64,32) + log + standardised target": make_model((64, 32), target="log"),
        "V3 (128,64) + standardised target": make_model((128, 64), target="standard"),
    }
    rows = [{"variant": n, **cv_scores(m, X_train, y_train, cv)} for n, m in variants.items()]
    exp = pd.DataFrame(rows).round(4)
    print(exp.to_string(index=False))
    exp.to_csv(REPORT_DIR / "experiments_cv.csv", index=False)
    # choose the target representation automatically from the experiments
    target_mode = "log" if exp.loc[exp.variant.str.contains("log"), "cv_RMSE"].iloc[0] < \
        exp.loc[exp.variant.str.contains("V1"), "cv_RMSE"].iloc[0] else "standard"
    print(f"Target representation chosen by CV: {target_mode}")
    print(LINE + "\n")

    # 4. Part B.1 - tuning, shown BEFORE vs AFTER
    print(LINE)
    print("4. Hyper-parameter tuning (RandomizedSearchCV, TimeSeriesSplit)")
    # BEFORE = scikit-learn's own default settings (same target scaling, so only the settings differ)
    before_p = {"hidden_layer_sizes": (100,), "activation": "relu", "alpha": 1e-4,
                "learning_rate_init": 1e-3, "batch_size": 200, "n_iter_no_change": 15}
    mk_before = lambda: make_model((100,), "relu", 1e-4, 1e-3, 200, target=target_mode)
    before = mk_before().fit(X_train, y_train)
    before_cv = cv_scores(mk_before(), X_train, y_train, cv)
    # AFTER = random search over a wider space: more architectures, a much wider L2 range (alpha),
    # wider learning-rate range, more batch sizes and the early-stopping patience.
    space = {
        "regressor__hidden_layer_sizes": [(32,), (64,), (128,), (256,), (64, 32), (128, 64), (256, 128),
                                          (128, 64, 32), (256, 128, 64)],
        "regressor__activation": ["relu", "tanh"],
        "regressor__alpha": loguniform(1e-5, 1e1),            # L2 penalty: bigger = simpler network
        "regressor__learning_rate_init": loguniform(1e-4, 3e-2),
        "regressor__batch_size": [32, 64, 128, 256],
        "regressor__n_iter_no_change": [10, 15, 25],
    }
    search = RandomizedSearchCV(make_model(target=target_mode), space, n_iter=N_ITER_SEARCH, cv=cv,
                                scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=SEED, refit=True)
    search.fit(X_train, y_train)
    best_params = {k.replace("regressor__", ""): v for k, v in search.best_params_.items()}
    trials = pd.DataFrame(search.cv_results_)
    trials["cv_RMSE"] = -trials["mean_test_score"]
    trials = trials.rename(columns=lambda c: c.replace("param_regressor__", ""))
    trials = trials[list(best_params) + ["cv_RMSE", "std_test_score", "rank_test_score"]].sort_values("rank_test_score")
    trials.to_csv(REPORT_DIR / "tuning_trials.csv", index=False)
    hp = pd.DataFrame({"before_tuning": {k: str(v) for k, v in before_p.items()},
                       "after_tuning": {k: (f"{v:.6g}" if isinstance(v, float) else str(v)) for k, v in best_params.items()}})
    hp.to_csv(REPORT_DIR / "hyperparameters_before_after.csv")
    print(f"Candidates tried : {N_ITER_SEARCH} (x {N_SPLITS} folds)")
    print("\nSettings BEFORE vs AFTER tuning:")
    print(hp.to_string())
    print(f"\nCV RMSE before tuning: {before_cv['cv_RMSE']:.3f} (+/- {before_cv['cv_RMSE_std']:.3f}) -> "
          f"after tuning: {-search.best_score_:.3f}")
    print("Top-5 CV RMSE    :", trials["cv_RMSE"].head(5).round(3).tolist())
    print(LINE + "\n")

    # 5. Part B.2 - ensemble of MLPs with different random seeds
    print(LINE)
    print(f"5. Ensemble of {N_ENSEMBLE} MLPs (different random seeds)")
    best_single = search.best_estimator_                         # already refit on all training rows
    ens = SeedEnsemble(base=clone(search.best_estimator_), n_members=N_ENSEMBLE).fit(X_train, y_train)
    single_pred = np.clip(best_single.predict(X_test), 0, None)
    ens_pred = np.clip(ens.predict(X_test), 0, None)
    member_rmse = [score_all(y_test, np.clip(m.predict(X_test), 0, None))["RMSE"] for m in ens.models_]
    print(f"RMSE of the {N_ENSEMBLE} individual networks: {np.round(member_rmse, 3).tolist()}")
    print(f"Spread across seeds (std): {np.std(member_rmse):.3f}  -> how much a single network depends on luck")
    print(f"Ensemble RMSE: {score_all(y_test, ens_pred)['RMSE']:.3f}")
    print(LINE + "\n")

    # 6. Final evaluation on the held-out LATEST dates (used once)
    print(LINE)
    print("6. Evaluation (test set = most recent dates)")
    before_pred = np.clip(before.predict(X_test), 0, None)
    results = {"dummy_mean": dummy_scores, "mlp_before_tuning": score_all(y_test, before_pred),
               "mlp_after_tuning": score_all(y_test, single_pred), "mlp_ensemble": score_all(y_test, ens_pred)}
    cmp = pd.DataFrame(results).T.round(4)
    print(cmp.to_string())
    cmp.to_csv(REPORT_DIR / "test_metrics_comparison.csv")
    ba = pd.DataFrame({"before tuning": results["mlp_before_tuning"], "after tuning (single)": results["mlp_after_tuning"],
                       "after tuning (ensemble)": results["mlp_ensemble"]}).T[["MAE", "RMSE", "R2"]].round(4)
    ba.to_csv(REPORT_DIR / "before_vs_after_tuning.csv")
    b, a = results["mlp_before_tuning"], results["mlp_ensemble"]
    print("\nBEFORE vs AFTER tuning (test set):")
    print(ba.to_string())
    print(f"RMSE {b['RMSE']:.3f} -> {a['RMSE']:.3f} ({100 * (b['RMSE'] - a['RMSE']) / b['RMSE']:.1f}% lower) | "
          f"MAE {b['MAE']:.3f} -> {a['MAE']:.3f} ({100 * (b['MAE'] - a['MAE']) / b['MAE']:.1f}% lower) | "
          f"R2 {b['R2']:.3f} -> {a['R2']:.3f}")
    # final model = whichever of single / ensemble has the lower CV-independent test RMSE is NOT used to
    # choose (that would peek at the test set): the ensemble is the default because averaging never
    # increases the expected error and makes the result less seed-dependent.
    final_pred, final_name = ens_pred, "mlp_ensemble"
    yt = y_test.to_numpy(float)
    f = results[final_name]
    print(f"\nFinal model ({final_name}): off by {f['MAE']:.1f} units on average (mean sales {yt.mean():.1f}), "
          f"R2 {f['R2']:.3f}")

    # Bootstrap CI for the MLP's RMSE, and a PAIRED bootstrap of before vs after tuning on the same resampled rows
    rng = np.random.default_rng(SEED)
    boot_rmse, boot_diff = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(yt), len(yt))
        r_m = np.sqrt(np.mean((yt[idx] - final_pred[idx]) ** 2))
        r_l = np.sqrt(np.mean((yt[idx] - before_pred[idx]) ** 2))
        boot_rmse.append(r_m)
        boot_diff.append(r_l - r_m)                      # positive = tuned MLP better
    lo, hi = np.percentile(boot_rmse, [2.5, 97.5])
    dlo, dhi = np.percentile(boot_diff, [2.5, 97.5])
    print(f"Bootstrap 95% CI for MLP RMSE: [{lo:.3f}, {hi:.3f}]")
    print(f"Paired bootstrap, RMSE(before) - RMSE(after): mean {np.mean(boot_diff):+.3f}, 95% CI [{dlo:.3f}, {dhi:.3f}]")
    verdict = ("tuning gives a significant improvement" if dlo > 0 else
               "tuning made the MLP significantly WORSE" if dhi < 0 else
               "the tuning gain is NOT statistically significant")
    print(f"-> {verdict} (interval {'excludes' if dlo > 0 or dhi < 0 else 'includes'} 0).")
    print(LINE + "\n")

    # 7. Generalisation: over-fitting check and learning curve (A.4)
    print(LINE)
    print("7. Generalisation")
    train_pred = np.clip(ens.predict(X_train), 0, None)
    tr = score_all(y_train, train_pred)
    print(f"Train RMSE {tr['RMSE']:.3f} vs test RMSE {f['RMSE']:.3f} (gap {f['RMSE'] - tr['RMSE']:+.3f}); "
          f"train R2 {tr['R2']:.3f} vs test R2 {f['R2']:.3f}")
    print("A large gap means over-fitting; a small gap with a low R2 means the limit is the data, not the model.")
    lc = []                                  # learning curve: train on the EARLIEST fraction of the data
    for frac in (0.25, 0.5, 0.75, 1.0):
        n = int(len(X_train) * frac)
        m = clone(best_single).fit(X_train.iloc[:n], y_train.iloc[:n])
        lc.append({"train_fraction": frac, "train_rows": n,
                   "train_RMSE": score_all(y_train.iloc[:n], np.clip(m.predict(X_train.iloc[:n]), 0, None))["RMSE"],
                   "test_RMSE": score_all(y_test, np.clip(m.predict(X_test), 0, None))["RMSE"]})
    lc = pd.DataFrame(lc).round(3)
    lc.to_csv(REPORT_DIR / "learning_curve.csv", index=False)
    print(lc.to_string(index=False))
    print("If test RMSE is still falling at 100%, more data would help; if it is flat, the model has saturated.")
    print(LINE + "\n")

    # 8. Interpretability (A.3): an MLP has no coefficients, so use model-agnostic tools
    print(LINE)
    print("8. Key drivers of sales")
    perm = permutation_importance(ens, X_test, y_test, scoring="neg_root_mean_squared_error",
                                  n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False)
    imp.to_csv(REPORT_DIR / "permutation_importance.csv", header=["rmse_increase"])
    print("Permutation importance (RMSE increase when the input is shuffled):")
    print(imp.head(8).round(3).to_string())
    pdps = {}
    for col in [c for c in PDP_FEATURES if c in cols]:
        pdps[col] = partial_dependence(ens, X_test, col)
    print("Partial-dependence curves saved: they show the SHAPE of each effect (straight = linear, bend = non-linear).")
    print(LINE + "\n")

    # 9. Error analysis (A.4)
    print(LINE)
    print("9. Error analysis")
    resid = yt - final_pred                          # actual - predicted
    print(f"Mean residual (bias): {-resid.mean():+.2f} units  (0 = no systematic over/under-forecast)")
    r_ph, p_ph = stats.spearmanr(final_pred, np.abs(resid))
    print(f"Error grows with predicted size? Spearman(|residual|, prediction) = {r_ph:.2f} (p={p_ph:.1e})")
    print(f"Residual skewness {stats.skew(resid):.2f}, excess kurtosis {stats.kurtosis(resid):.2f}")
    print(f"Over-forecast: {np.mean(resid < 0):.1%} of rows | under-forecast: {np.mean(resid > 0):.1%}")
    big = np.abs(resid) > 2 * f["MAE"]
    print(f"Large misses (> 2 x MAE = {2 * f['MAE']:.1f} units): {big.mean():.1%} of rows; "
          f"their average actual sales {yt[big].mean():.1f} vs {yt[~big].mean():.1f} for the rest")

    slices = []
    for prefix in GROUP_PREFIXES:
        label = decode_group(X_test, prefix)
        for val in sorted(label.unique()):
            m = (label == val).to_numpy()
            if m.sum() >= 30:                                   # tiny groups are too noisy
                s = score_all(yt[m], final_pred[m])
                slices.append({"slice": prefix.rstrip("_"), "value": val, "rows": int(m.sum()),
                               "MAE": s["MAE"], "RMSE": s["RMSE"], "R2": s["R2"], "Bias": s["Bias"]})
    dts = pd.to_datetime(date_test).reset_index(drop=True)
    for i, chunk in enumerate(np.array_split(np.arange(len(dts)), 3), 1):   # 3 chronological periods
        s = score_all(yt[chunk], final_pred[chunk])
        slices.append({"slice": "Time period", "value": f"T{i}: {dts[chunk[0]].date()} to {dts[chunk[-1]].date()}",
                       "rows": len(chunk), "MAE": s["MAE"], "RMSE": s["RMSE"], "R2": s["R2"], "Bias": s["Bias"]})
    rob = pd.DataFrame(slices).round(3)
    rob.to_csv(REPORT_DIR / "robustness_by_slice.csv", index=False)
    print("\nRobustness by slice (large gaps = weak spots):")
    print(rob.to_string(index=False))

    worst = pd.DataFrame({"Date": date_test.values, "Actual": yt, "Predicted": final_pred.round(1),
                          "AbsError": np.abs(resid).round(1)}).sort_values("AbsError", ascending=False)
    worst.head(20).to_csv(REPORT_DIR / "worst_predictions.csv", index=False)

    # Row-level predictions for Student C (evaluation) and Student D (UI / recommender)
    pd.DataFrame({"Date": date_test.values, "Actual_Units": yt, "Predicted_Units": final_pred.round(2),
                  "Error": (final_pred - yt).round(2),
                  "Predicted_Units_Before_Tuning": before_pred.round(2)}).to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    print(LINE + "\n")

    # 10. Plots
    print(LINE)
    print("10. Plots")
    _save_plots(ens, yt, final_pred, resid, imp, results, pdps, lc, trials, before_cv)
    print(f"Saved to {REPORT_DIR}")
    print(LINE + "\n")

    # 11. Shortlist summary + save (A.7)
    print(LINE)
    print("11. Summary and saving")
    print(f"Before tuning : RMSE {b['RMSE']:.3f} | MAE {b['MAE']:.3f} | R2 {b['R2']:.3f}")
    print(f"After tuning  : RMSE {a['RMSE']:.3f} | MAE {a['MAE']:.3f} | R2 {a['R2']:.3f}  (RMSE change {a['RMSE'] - b['RMSE']:+.3f})")
    if dlo <= 0:
        print("Tuning gave no statistically significant gain: the limit is the data, not the settings - a valid finding.")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Save the list of fitted networks (average their predictions) - avoids pickling a custom class.
    joblib.dump({"models": ens.models_, "features": cols, "best_params": best_params,
                 "target_mode": target_mode, "metrics": results}, MODEL_FILE)
    with open(REPORT_DIR / "metrics.json", "w") as fh:
        json.dump({"best_params": {k: str(v) for k, v in best_params.items()}, "target_mode": target_mode,
                   "bootstrap_rmse_ci95": [lo, hi], "paired_bootstrap_before_minus_after_ci95": [dlo, dhi],
                   "cv_rmse_before_tuning": before_cv["cv_RMSE"], "cv_rmse_after_tuning": -search.best_score_,
                   "params_before_tuning": {k: str(v) for k, v in before_p.items()},
                   "results": results}, fh, indent=2, default=float)
    print(f"Model saved   : {MODEL_FILE}")
    print(f"Reports saved : {REPORT_DIR}")
    print(LINE + "\n")
    return ens


def predict_units(bundle, X):
    """Use the saved model (for the UI / recommender):  bundle = joblib.load(MODEL_FILE)."""
    return np.clip(np.mean([m.predict(X[bundle["features"]]) for m in bundle["models"]], axis=0), 0, None)


# ─────────────────────────────────────────────
# 3. PLOTS
# ─────────────────────────────────────────────
def _save_plots(ens, y, pred, resid, imp, results, pdps, lc, trials, before_cv):
    # Training curves of the first network: loss should fall and validation R2 should level off
    reg = ens.models_[0].regressor_
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(reg.loss_curve_); axs[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
    axs[1].plot(reg.validation_scores_); axs[1].set(title="Validation R2 (early stopping)", xlabel="epoch", ylabel="R2")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "training_curves.png", dpi=150); plt.close(fig)

    # Actual vs predicted
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, pred, s=5, alpha=0.3)
    lim = [0, max(y.max(), pred.max())]
    ax.plot(lim, lim, "r--", lw=1, label="perfect prediction")
    ax.set(xlabel="Actual units sold", ylabel="Predicted units sold", title="MLP: actual vs predicted (test set)")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORT_DIR / "actual_vs_predicted.png", dpi=150); plt.close(fig)

    # Residuals vs prediction
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pred, resid, s=5, alpha=0.3); ax.axhline(0, color="r", lw=1)
    ax.set(xlabel="Predicted units", ylabel="Residual (actual - predicted)", title="Residuals vs predicted")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "residuals_vs_predicted.png", dpi=150); plt.close(fig)

    # Residual distribution + Q-Q plot
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].hist(resid, bins=50, color="#2a7ab0"); axs[0].set(title="Residual distribution", xlabel="Residual (units)")
    stats.probplot(resid, dist="norm", plot=axs[1]); axs[1].set_title("Q-Q plot of residuals")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "residual_distribution.png", dpi=150); plt.close(fig)

    # Permutation importance
    t = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(t.index, t.values, color="#2a7ab0")
    ax.set(title="MLP permutation importance", xlabel="RMSE increase when shuffled")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "permutation_importance.png", dpi=150); plt.close(fig)

    # Partial dependence: shape of each effect
    if pdps:
        fig, axs = plt.subplots(1, len(pdps), figsize=(4 * len(pdps), 3.6), squeeze=False)
        for ax, (col, (g, c_mlp)) in zip(axs[0], pdps.items()):
            ax.plot(g, c_mlp, "o-")
            ax.set(title=col, xlabel="scaled value", ylabel="avg predicted units")
        fig.suptitle("Partial dependence: shape of each effect", y=1.03)
        fig.tight_layout(); fig.savefig(REPORT_DIR / "partial_dependence.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Learning curve
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(lc.train_rows, lc.train_RMSE, "o-", label="train"); ax.plot(lc.train_rows, lc.test_RMSE, "s-", label="test")
    ax.set(xlabel="training rows (earliest dates)", ylabel="RMSE", title="Learning curve"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "learning_curve.png", dpi=150); plt.close(fig)

    # Before vs after tuning (test set, lower = better)
    m = pd.DataFrame({k: results[k] for k in ("mlp_before_tuning", "mlp_after_tuning", "mlp_ensemble")}).T[["MAE", "RMSE"]]
    ax = m.plot.bar(figsize=(7, 4), rot=0)
    ax.set_title("MLP before vs after tuning (test set, lower = better)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    plt.tight_layout(); plt.savefig(REPORT_DIR / "tuning_before_after.png", dpi=150); plt.close()

    # Every tuning trial (ranked) against the default settings
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(range(1, len(trials) + 1), trials["cv_RMSE"].values, s=25, label="tuning trials (ranked)")
    ax.axhline(before_cv["cv_RMSE"], color="r", ls="--", label="before tuning (defaults)")
    ax.axhline(trials["cv_RMSE"].min(), color="g", ls="--", label="best after tuning")
    ax.set(xlabel="trial rank (1 = best)", ylabel="CV RMSE", title="Tuning trials vs default settings"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "tuning_trials.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    train_mlp()