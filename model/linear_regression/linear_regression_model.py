import json
import os
import sys
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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")
# n_jobs=-1 runs the search in separate worker processes (spawned on Windows), and those do NOT
# inherit the filter above. This environment variable is inherited, so the workers stay quiet too.
os.environ["PYTHONWARNINGS"] = "ignore"

LINE = "─" * 45

# ─────────────────────────────────────────────
# 0. SETTINGS
# ─────────────────────────────────────────────
SEED = 42
# src/backend/model/linear_regression/<this file>  -> project root is 4 levels up.
# PROJECT_ROOT can override it (tests / other machines).
ROOT_DIR = (Path(os.environ["PROJECT_ROOT"]) if "PROJECT_ROOT" in os.environ
            else Path(__file__).resolve().parents[2])
TRAIN_FILE = ROOT_DIR / "data" / "processed" / "train.csv"
TEST_FILE = ROOT_DIR / "data" / "processed" / "test.csv"
MODEL_FILE = Path(__file__).resolve().parent / "linear_regression_model.joblib"

REPORT_DIR = Path(__file__).resolve().parent / "outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Units_Sold"

# Column lists come from the data-preparation code, so they are defined in one place only.
if str(ROOT_DIR / "src" / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src" / "backend"))
from data_modeling import NON_FEATURE_COLS   # noqa: E402  (target, date, ID and stock columns)
# Not model inputs: the label, bookkeeping columns, and Inventory_Level (kept for the recommender;
# it is strongly tied to sales, so excluding it keeps this a pure "demand drivers" model).
NON_FEATURES = NON_FEATURE_COLS   # Units_Sold, Date, Transaction_ID, Store_ID, Product_ID, Inventory_Level

N_SPLITS = 5            # time-ordered CV folds
N_ITER_SEARCH = 40      # random-search budget (cheaper than a full grid, Part B.1)
SOLVER_ITER, SOLVER_TOL = 20000, 1e-3   # Lasso / ElasticNet solver: enough iterations, sensible tolerance
N_BOOTSTRAP = 500       # resamples for the confidence interval on test RMSE
N_BOOT_COEF = 200       # resamples for coefficient confidence intervals
GROUP_PREFIXES = ["Product_Category_", "Store_Type_", "Weather_Condition_"]
INTERACTION_COLS = ["Unit_Price", "Discount_Percent", "Marketing_Spend",
                    "Customer_Footfall", "Month", "Promotion", "Holiday"]


# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────
def load_split(path):
    """Read a prepared split (already encoded + scaled by data_modeling.py)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run pipeline.py first")
    df = pd.read_csv(path)
    return df.drop(columns=NON_FEATURES), df[TARGET], df["Date"]


def decode_group(X, prefix):
    """Turn a one-hot family (e.g. Product_Category_*) back into one readable label per row."""
    cols = [c for c in X.columns if c.startswith(prefix)]
    return X[cols].idxmax(axis=1).str.replace(prefix, "", regex=False)


def score_all(y_true, y_pred):
    """Regression metrics. Predictions are clipped at 0: a negative unit count is impossible."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),                 # average miss, in units
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),  # punishes big misses more
        "MedAE": median_absolute_error(y_true, y_pred),             # robust to a few huge misses
        "R2": r2_score(y_true, y_pred),                             # share of variance explained
        "sMAPE_%": 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)),
        "Bias": float(np.mean(y_pred - y_true)),                    # >0 = over-forecasting on average
    }


def make_pipeline(feature_cols, reg, use_interactions=False, use_l1_selection=False, log_target=False):
    """
    One candidate model. Switches map to Part A.5 (representations / feature selection):
      use_interactions : add pairwise products of the main drivers (price x promotion, ...) so a LINEAR
                         model can capture simple non-linear effects
      use_l1_selection : keep only inputs an L1-penalised model finds useful
      log_target       : model log(1 + Units_Sold) - stabilises the spread of a skewed count and keeps
                         predictions positive. Predictions are converted back to units automatically.
    """
    steps = []
    if use_interactions:
        cols = [c for c in INTERACTION_COLS if c in feature_cols]
        steps.append(("interact", ColumnTransformer(
            [("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), cols)],
            remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")))
    if use_l1_selection:
        steps.append(("select", SelectFromModel(Lasso(alpha=0.01, max_iter=SOLVER_ITER, tol=SOLVER_TOL, random_state=SEED), threshold=1e-6)))
    steps.append(("reg", reg))
    pipe = Pipeline(steps)
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def inner(model):
    """The Pipeline inside a (possibly target-transformed) model."""
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def feature_names_of(pipe, input_cols):
    """Follow the column names through the optional steps so coefficients stay readable."""
    names = np.array(input_cols)
    if not isinstance(pipe.named_steps.get("interact", "passthrough"), str):
        names = np.array(pipe.named_steps["interact"].get_feature_names_out())
    if not isinstance(pipe.named_steps.get("select", "passthrough"), str):
        names = names[pipe.named_steps["select"].get_support()]
    return names


def cv_rmse_mae(model, X, y, cv):
    res = cross_validate(clone(model), X, y, cv=cv, n_jobs=-1,
                         scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])
    return {"cv_RMSE": -res["test_neg_root_mean_squared_error"].mean(),
            "cv_RMSE_std": res["test_neg_root_mean_squared_error"].std(),
            "cv_MAE": -res["test_neg_mean_absolute_error"].mean(),
            "cv_R2": res["test_r2"].mean()}


# ─────────────────────────────────────────────
# 2. LINEAR REGRESSION BASELINE + REFINEMENT
# ─────────────────────────────────────────────
def train_linear_regression():
    np.random.seed(SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the prepared data (already encoded and scaled by data_modeling)
    print(LINE)
    print("1. Data")
    X_train, y_train, date_train = load_split(TRAIN_FILE)
    X_test, y_test, date_test = load_split(TEST_FILE)
    cols = list(X_train.columns)
    print(f"Train : {len(X_train):,} rows | Test : {len(X_test):,} rows | Inputs : {X_train.shape[1]}")
    print(f"Train dates {date_train.min()} to {date_train.max()} | Test dates {date_test.min()} to {date_test.max()}")
    print(f"{TARGET}: train mean {y_train.mean():.1f}, median {y_train.median():.0f}, "
          f"min {y_train.min()}, max {y_train.max()}  | skewness {stats.skew(y_train):.2f}")
    print("Skewness > 0.5 means a long right tail -> a log-transformed target is worth testing (V1).")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing values reached the model - fix them in data_cleaning.py")
    assert date_train.is_monotonic_increasing, "train.csv must be sorted by Date (needed for time-ordered CV)"
    print("Missing values : none (handled upstream in data_cleaning.py: median / mode imputation)")
    print(LINE + "\n")

    cv = TimeSeriesSplit(n_splits=N_SPLITS)    # always validate on dates AFTER the training dates

    # 2. Reference point: what does "no learning" score?
    print(LINE)
    print("2. Reference point (is the baseline learning anything?)")
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    dummy_scores = score_all(y_test, dummy.predict(X_test))
    print(f"Always predict the training mean ({y_train.mean():.1f} units): "
          f"MAE {dummy_scores['MAE']:.2f}, RMSE {dummy_scores['RMSE']:.2f}, R2 {dummy_scores['R2']:.3f}")
    print("Any real model must clearly beat this.")
    print(LINE + "\n")

    # 3. V0 - plain Ordinary Least Squares. This is THE number other models must beat.
    print(LINE)
    print("3. V0 baseline (plain Linear Regression / OLS, no tuning)")
    v0 = make_pipeline(cols, LinearRegression())
    v0.fit(X_train, y_train)
    v0_cv = cv_rmse_mae(v0, X_train, y_train, cv)
    print(f"Time-series CV ({N_SPLITS} folds): RMSE {v0_cv['cv_RMSE']:.2f} +/- {v0_cv['cv_RMSE_std']:.2f} | "
          f"MAE {v0_cv['cv_MAE']:.2f} | R2 {v0_cv['cv_R2']:.3f}")
    print("A small +/- means the score is stable from fold to fold (stability / reliability).")
    print(LINE + "\n")

    # 4. Iterate (A.5 / A.6): one improvement at a time, judged by CV on TRAIN only
    print(LINE)
    print("4. Refinement experiments (cross-validation on TRAIN only - test stays untouched)")
    variants = {
        "V0 OLS baseline": v0,
        "V1 + log target": make_pipeline(cols, LinearRegression(), log_target=True),
        "V2 + interactions": make_pipeline(cols, LinearRegression(), use_interactions=True),
        "V3 + log + interactions": make_pipeline(cols, LinearRegression(), use_interactions=True, log_target=True),
        "V4 Ridge (a=10)": make_pipeline(cols, Ridge(alpha=10), use_interactions=True, log_target=True),
    }
    rows = [{"variant": n, **cv_rmse_mae(m, X_train, y_train, cv)} for n, m in variants.items()]
    exp = pd.DataFrame(rows).round(4)
    print(exp.to_string(index=False))
    exp.to_csv(REPORT_DIR / "experiments_cv.csv", index=False)
    print(LINE + "\n")

    # 5. Part B.1 - automatic search over regularisation family, strength, representation and target.
    #    Random search (not a full grid) is cheaper and just as good for a handful of settings.
    print(LINE)
    print("5. Hyper-parameter tuning (RandomizedSearchCV, TimeSeriesSplit)")
    best_cv, best_search, best_log = np.inf, None, None
    for log_target in (False, True):
        base = Pipeline([("interact", "passthrough"), ("select", "passthrough"), ("reg", Ridge())])
        inter = ColumnTransformer(
            [("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
              [c for c in INTERACTION_COLS if c in cols])],
            remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
        select = SelectFromModel(Lasso(alpha=0.01, max_iter=SOLVER_ITER, tol=SOLVER_TOL, random_state=SEED), threshold=1e-6)
        space = [   # one dict per model family so each only gets parameters it understands
            {"interact": ["passthrough", inter], "select": ["passthrough", select],
             "reg": [Ridge()], "reg__alpha": loguniform(1e-2, 1e3)},
            {"interact": ["passthrough", inter], "select": ["passthrough", select],
             "reg": [Lasso(max_iter=SOLVER_ITER, tol=SOLVER_TOL)], "reg__alpha": loguniform(1e-3, 1)},
            {"interact": ["passthrough", inter], "select": ["passthrough", select],
             "reg": [ElasticNet(max_iter=SOLVER_ITER, tol=SOLVER_TOL)], "reg__alpha": loguniform(1e-3, 1),
             "reg__l1_ratio": [0.2, 0.5, 0.8]},
        ]
        est = (TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)
               if log_target else base)
        if log_target:     # parameters of a wrapped pipeline need the "regressor__" prefix
            space = [{f"regressor__{k}": v for k, v in d.items()} for d in space]
        s = RandomizedSearchCV(est, space, n_iter=N_ITER_SEARCH // 2, cv=cv, n_jobs=-1, random_state=SEED,
                               scoring="neg_root_mean_squared_error", refit=True)
        s.fit(X_train, y_train)
        print(f"  target = {'log1p(units)' if log_target else 'units       '} -> best CV RMSE {-s.best_score_:.3f}")
        if -s.best_score_ < best_cv:
            best_cv, best_search, best_log = -s.best_score_, s, log_target
    best = best_search.best_estimator_
    bp = {k.replace("regressor__", ""): v for k, v in best_search.best_params_.items()}
    clean = {k: (type(v).__name__ if not isinstance(v, (str, float, int)) else
                 (round(float(v), 5) if isinstance(v, float) else v)) for k, v in bp.items()}
    clean["log_target"] = best_log
    print(f"Candidates tried : {N_ITER_SEARCH} (x {N_SPLITS} folds)")
    print(f"Best CV RMSE     : {best_cv:.3f}  (V0 plain OLS: {v0_cv['cv_RMSE']:.3f})")
    print(f"Best settings    : {clean}")
    print(LINE + "\n")

    # 6. Final evaluation on the held-out LATEST dates (used once)
    print(LINE)
    print("6. Evaluation (test set = most recent dates)")
    pred_v0 = np.clip(v0.predict(X_test), 0, None)
    pred = np.clip(best.predict(X_test), 0, None)
    results = {"dummy_mean": dummy_scores, "baseline_v0_OLS": score_all(y_test, pred_v0),
               "tuned": score_all(y_test, pred)}
    cmp = pd.DataFrame(results).T.round(4)
    print(cmp.to_string())
    cmp.to_csv(REPORT_DIR / "test_metrics_comparison.csv")
    print(f"\nOn average the tuned model is off by {results['tuned']['MAE']:.1f} units "
          f"(mean sales {y_test.mean():.1f}) and explains {results['tuned']['R2']:.1%} of the variance.")

    # Bootstrap: how much could RMSE move just from which rows ended up in the test set?
    rng = np.random.default_rng(SEED)
    yt = y_test.to_numpy(float)
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(yt), len(yt))
        boot.append(np.sqrt(np.mean((yt[idx] - pred[idx]) ** 2)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"Bootstrap 95% CI for RMSE ({N_BOOTSTRAP} resamples): [{lo:.3f}, {hi:.3f}]")
    print("If another model's RMSE lies inside this band, the difference may just be noise.")
    print(LINE + "\n")

    # 7. Interpretability (A.3)
    print(LINE)
    print("7. Key drivers of sales")
    inn = inner(best)
    names = feature_names_of(inn, cols)
    coef = pd.Series(inn.named_steps["reg"].coef_, index=names)
    # ---- Statistical drivers: plain OLS + bootstrap confidence intervals -------------------------
    # IMPORTANT (dummy-variable trap): every one-hot family sums to 1 (a row has exactly ONE store type,
    # ONE category ...) and the sub-categories add up to their category. With an intercept these columns
    # are perfectly collinear, so OLS cannot tell them apart and returns arbitrary huge numbers (1e13).
    # Fix: (1) leave the sub-category family out of THIS table (it only repeats the category) and
    # (2) drop the most common level of each remaining family as the REFERENCE level. Each remaining
    # dummy coefficient then means "units sold compared with the reference level".
    ols_cols = [c for c in cols if not c.startswith("Product_Subcategory_")]
    reference = []
    for prefix in GROUP_PREFIXES:
        fam = [c for c in ols_cols if c.startswith(prefix)]
        reference.append(X_train[fam].sum().idxmax())          # most frequent level = reference
    ols_cols = [c for c in ols_cols if c not in reference]
    rank = np.linalg.matrix_rank(np.column_stack([np.ones(len(X_train)), X_train[ols_cols].to_numpy()]))
    print(f"OLS design check: {len(ols_cols) + 1} columns incl. intercept, rank {rank} "
          f"-> {'full rank, coefficients are identifiable' if rank == len(ols_cols) + 1 else 'RANK DEFICIENT'}")
    print(f"Reference levels (coefficients are relative to these): {', '.join(reference)}")
    assert rank == len(ols_cols) + 1, "OLS design is collinear - coefficients would be meaningless"
    ols = LinearRegression().fit(X_train[ols_cols], y_train)
    ols_coef = pd.Series(ols.coef_, index=ols_cols)
    bc = []
    for _ in range(N_BOOT_COEF):
        idx = rng.integers(0, len(X_train), len(X_train))
        bc.append(LinearRegression().fit(X_train[ols_cols].iloc[idx], y_train.iloc[idx]).coef_)
    bc = np.array(bc)
    drivers = pd.DataFrame({"coef_units": ols_coef, "ci_low": np.percentile(bc, 2.5, axis=0),
                            "ci_high": np.percentile(bc, 97.5, axis=0)})
    drivers["reliable"] = (drivers.ci_low > 0) | (drivers.ci_high < 0)   # CI excludes 0
    drivers = drivers.reindex(drivers.coef_units.abs().sort_values(ascending=False).index)
    drivers.to_csv(REPORT_DIR / "coefficients_ols_bootstrap.csv")
    coef.sort_values(key=abs, ascending=False).to_csv(REPORT_DIR / "coefficients_tuned.csv", header=["coef"])
    print("OLS coefficients. Numeric inputs: units per +1 std. Promotion/Holiday: effect of 0 -> 1. "
          "Category / store / weather: units vs the reference level.")
    print(drivers.head(10).round(3).to_string())
    print(f"Inputs with a reliable (CI excludes 0) effect: {int(drivers.reliable.sum())} of {len(drivers)}")

    # Permutation importance: shuffle one input, see how much the error grows
    perm = permutation_importance(best, X_test, y_test, scoring="neg_root_mean_squared_error",
                                  n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False)
    imp.to_csv(REPORT_DIR / "permutation_importance.csv", header=["rmse_increase"])
    print("\nPermutation importance (RMSE increase when the input is shuffled):")
    print(imp.head(8).round(3).to_string())

    # One merged table so the three views of "what matters" can be read side by side.
    # Interaction terms are excluded here (they are in coefficients_tuned.csv); a blank means the
    # input is not part of that view (e.g. sub-categories are not in the OLS table, see above).
    summary = pd.DataFrame(index=cols)
    summary["ols_coef"] = drivers.coef_units
    summary["ols_ci_low"], summary["ols_ci_high"] = drivers.ci_low, drivers.ci_high
    summary["ols_reliable"] = drivers.reliable
    summary["tuned_coef"] = coef.reindex(cols)        # 0 = removed by the L1 penalty; main effect only
    summary["permutation_rmse_increase"] = imp
    summary = summary.sort_values("permutation_rmse_increase", ascending=False)
    summary.to_csv(REPORT_DIR / "feature_summary.csv")
    print(LINE + "\n")

    # 8. Error analysis (A.4)
    print(LINE)
    print("8. Error analysis")
    resid = yt - pred                       # actual - predicted
    print(f"Mean residual (bias): {-resid.mean():+.2f} units  (0 = no systematic over/under-forecast)")
    # Heteroscedasticity: do errors grow with the predicted level? (a linear-model weakness)
    r_ph, p_ph = stats.spearmanr(pred, np.abs(resid))
    print(f"Error grows with predicted size? Spearman(|residual|, prediction) = {r_ph:.2f} (p={p_ph:.1e})"
          f"{'  -> yes, big sellers are harder to predict' if r_ph > 0.1 else ''}")
    # Normality of residuals (matters for confidence statements, not for the predictions themselves)
    print(f"Residual skewness {stats.skew(resid):.2f}, excess kurtosis {stats.kurtosis(resid):.2f} "
          "(0 / 0 would be perfectly normal)")
    # Over- vs under-forecasting: the business cost differs (over = excess stock, under = stock-out)
    print(f"Over-forecast (pred > actual): {np.mean(resid < 0):.1%} of rows | "
          f"under-forecast: {np.mean(resid > 0):.1%}")
    big = np.abs(resid) > 2 * results["tuned"]["MAE"]
    print(f"Large misses (> 2 x MAE = {2 * results['tuned']['MAE']:.1f} units): {big.mean():.1%} of rows")
    print("Their average actual sales vs the rest: "
          f"{yt[big].mean():.1f} vs {yt[~big].mean():.1f} units")

    slices = []
    for prefix in GROUP_PREFIXES:
        label = decode_group(X_test, prefix)
        for val in sorted(label.unique()):
            m = (label == val).to_numpy()
            if m.sum() >= 30:                                   # tiny groups are too noisy
                s = score_all(yt[m], pred[m])
                slices.append({"slice": prefix.rstrip("_"), "value": val, "rows": int(m.sum()),
                               "MAE": s["MAE"], "RMSE": s["RMSE"], "R2": s["R2"], "Bias": s["Bias"]})
    dts = pd.to_datetime(date_test).reset_index(drop=True)
    for i, chunk in enumerate(np.array_split(np.arange(len(dts)), 3), 1):   # 3 chronological periods
        s = score_all(yt[chunk], pred[chunk])
        slices.append({"slice": "Time period", "value": f"T{i}: {dts[chunk[0]].date()} to {dts[chunk[-1]].date()}",
                       "rows": len(chunk), "MAE": s["MAE"], "RMSE": s["RMSE"], "R2": s["R2"], "Bias": s["Bias"]})
    rob = pd.DataFrame(slices).round(3)
    rob.to_csv(REPORT_DIR / "robustness_by_slice.csv", index=False)
    print("\nRobustness by slice (large gaps = bias / weak spots):")
    print(rob.to_string(index=False))
    print("Time periods show whether error grows as the data moves away from the training dates (drift).")

    # Worst predictions, for manual inspection
    worst = pd.DataFrame({"Date": date_test.values, "Actual": yt, "Predicted": pred.round(1),
                          "AbsError": np.abs(resid).round(1)}).sort_values("AbsError", ascending=False)
    worst.head(20).to_csv(REPORT_DIR / "worst_predictions.csv", index=False)

    # Row-level predictions for Student C (evaluation) and Student D (UI / recommender)
    out = pd.DataFrame({"Date": date_test.values, "Actual_Units": yt, "Predicted_Units": pred.round(2),
                        "Error": (pred - yt).round(2)})
    out.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    print(LINE + "\n")

    # 9. Plots
    print(LINE)
    print("9. Plots")
    _save_plots(yt, pred, resid, drivers, imp, results)
    print(f"Saved to {REPORT_DIR}")
    print(LINE + "\n")

    # 10. Shortlist summary + save (A.7)
    print(LINE)
    print("10. Summary and saving")
    gain = results["baseline_v0_OLS"]["RMSE"] - results["tuned"]["RMSE"]
    print(f"Plain OLS RMSE : {results['baseline_v0_OLS']['RMSE']:.3f} | R2 {results['baseline_v0_OLS']['R2']:.3f}")
    print(f"Tuned RMSE     : {results['tuned']['RMSE']:.3f} | R2 {results['tuned']['R2']:.3f}  (RMSE change {-gain:+.3f})")
    if results["tuned"]["R2"] < 0.6:
        print("A linear model leaves much variance unexplained -> motivates non-linear models (MLP / XGBoost / RF).")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "baseline": v0, "features": cols, "metrics": results}, MODEL_FILE)
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump({"best_params": {k: str(v) for k, v in clean.items()},
                   "bootstrap_rmse_ci95": [lo, hi], "results": results},
                  f, indent=2, default=float)
    print(f"Model saved   : {MODEL_FILE}")
    print(f"Reports saved : {REPORT_DIR}")
    print(LINE + "\n")
    return best


# ─────────────────────────────────────────────
# 3. PLOTS
# ─────────────────────────────────────────────
def _save_plots(y, pred, resid, drivers, imp, results):
    # Actual vs predicted
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, pred, s=5, alpha=0.3)
    lim = [0, max(y.max(), pred.max())]
    ax.plot(lim, lim, "r--", lw=1, label="perfect prediction")
    ax.set_xlabel("Actual units sold"); ax.set_ylabel("Predicted units sold")
    ax.set_title("Actual vs predicted (test set)"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "actual_vs_predicted.png", dpi=150); plt.close(fig)

    # Residuals vs prediction (a funnel shape = errors grow with size = heteroscedasticity)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pred, resid, s=5, alpha=0.3); ax.axhline(0, color="r", lw=1)
    ax.set_xlabel("Predicted units"); ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals vs predicted")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "residuals_vs_predicted.png", dpi=150); plt.close(fig)

    # Residual distribution + Q-Q plot (normality check)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].hist(resid, bins=50, color="#2a7ab0"); axs[0].set_title("Residual distribution")
    axs[0].set_xlabel("Residual (units)")
    stats.probplot(resid, dist="norm", plot=axs[1]); axs[1].set_title("Q-Q plot of residuals")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "residual_distribution.png", dpi=150); plt.close(fig)

    # Top coefficients with bootstrap 95% CI
    t = drivers.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(t.index, t.coef_units,
            xerr=[t.coef_units - t.ci_low, t.ci_high - t.coef_units],
            color=np.where(t.coef_units > 0, "#2a7ab0", "#c0504d"), capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Top drivers of units sold (OLS, 95% bootstrap CI)"); ax.set_xlabel("units sold (see notes on units per input)")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "top_coefficients.png", dpi=150); plt.close(fig)

    # Permutation importance
    t = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(t.index, t.values, color="#2a7ab0")
    ax.set_title("Permutation importance"); ax.set_xlabel("RMSE increase when shuffled")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "permutation_importance.png", dpi=150); plt.close(fig)

    # Dummy vs baseline vs tuned
    m = pd.DataFrame(results).T[["MAE", "RMSE"]]
    ax = m.plot.bar(figsize=(6, 4), rot=0)
    ax.set_title("Error comparison (test set, lower = better)")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "model_comparison.png", dpi=150); plt.close()


if __name__ == "__main__":
    train_linear_regression()