import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

LINE = "─" * 45

# ─────────────────────────────────────────────
# 0. SETTINGS
# ─────────────────────────────────────────────
SEED = 42
ROOT_DIR = (Path(os.environ["PROJECT_ROOT"]) if "PROJECT_ROOT" in os.environ
            else Path(__file__).resolve().parents[2])
TRAIN_FILE = ROOT_DIR / "data" / "processed" / "train.csv"
TEST_FILE = ROOT_DIR / "data" / "processed" / "test.csv"
MODEL_FILE = Path(__file__).resolve().parent / "mlp_model.joblib"

REPORT_DIR = Path(__file__).resolve().parent / "outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Units_Sold"

if str(ROOT_DIR / "src" / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src" / "backend"))
from data_modeling import NON_FEATURE_COLS  
NON_FEATURES = NON_FEATURE_COLS  

N_SPLITS = 5          
N_ITER_SEARCH = 25   
N_ENSEMBLE = 5         
N_BOOTSTRAP = 500 
GROUP_PREFIXES = ["Product_Category_", "Store_Type_", "Weather_Condition_"]
PDP_FEATURES = ["Unit_Price", "Customer_Footfall", "Marketing_Spend", "Month"] 


# ─────────────────────────────────────────────
# 1. HELPERS
# ─────────────────────────────────────────────
def load_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run pipeline.py first")
    df = pd.read_csv(path)
    return df.drop(columns=NON_FEATURES), df[TARGET], df["Date"]


def decode_group(X, prefix):
    cols = [c for c in X.columns if c.startswith(prefix)]
    return X[cols].idxmax(axis=1).str.replace(prefix, "", regex=False)


def score_all(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "sMAPE_%": 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def make_model(hidden=(64, 32), activation="relu", alpha=1e-4, lr=1e-3, batch=128, target="standard", seed=SEED, early_stopping=True, max_iter=500):

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

    cv = TimeSeriesSplit(n_splits=N_SPLITS)   

    # 2. Reference points: "no learning" and the linear baseline the MLP must beat
    print(LINE)
    print("2. Reference points")
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    ols = LinearRegression().fit(X_train, y_train)
    ols_pred = np.clip(ols.predict(X_test), 0, None)
    ref = {"dummy_mean": score_all(y_test, dummy.predict(X_test)),
           "linear_regression_baseline": score_all(y_test, ols_pred)}
    ols_cv = cv_scores(LinearRegression(), X_train, y_train, cv)
    print(f"Mean-only dummy       : RMSE {ref['dummy_mean']['RMSE']:.2f}, R2 {ref['dummy_mean']['R2']:.3f}")
    print(f"Linear regression (V0): RMSE {ref['linear_regression_baseline']['RMSE']:.2f}, "
          f"R2 {ref['linear_regression_baseline']['R2']:.3f}  (CV RMSE {ols_cv['cv_RMSE']:.2f})")
    print("The MLP must clearly beat the linear model to justify its extra complexity.")
    print(LINE + "\n")

    # 3. Iterate: one change at a time, judged by CV on TRAIN only
    print(LINE)
    print("3. Refinement experiments (cross-validation on TRAIN only - test stays untouched)")
    variants = {
        "V0 sklearn defaults (100,), raw target": make_model((100,), target="raw", early_stopping=False, max_iter=200),
        "V1 (64,32) + standardised target": make_model((64, 32), target="standard"),
        "V2 (64,32) + log + standardised target": make_model((64, 32), target="log"),
        "V3 (128,64) + standardised target": make_model((128, 64), target="standard"),
    }
    rows = [{"variant": "Linear regression (reference)", **ols_cv}]
    rows += [{"variant": n, **cv_scores(m, X_train, y_train, cv)} for n, m in variants.items()]
    exp = pd.DataFrame(rows).round(4)
    print(exp.to_string(index=False))
    exp.to_csv(REPORT_DIR / "experiments_cv.csv", index=False)
    target_mode = "log" if exp.loc[exp.variant.str.contains("log"), "cv_RMSE"].iloc[0] < \
        exp.loc[exp.variant.str.contains("V1"), "cv_RMSE"].iloc[0] else "standard"
    print(f"Target representation chosen by CV: {target_mode}")
    print(LINE + "\n")

    # 4. automatic random search over architecture, regularisation and learning settings
    print(LINE)
    print("4. Hyper-parameter tuning (RandomizedSearchCV, TimeSeriesSplit)")
    space = {
        "regressor__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (128, 64, 32), (256, 128)],
        "regressor__activation": ["relu", "tanh"],
        "regressor__alpha": loguniform(1e-5, 1e0),   
        "regressor__learning_rate_init": loguniform(1e-4, 1e-2),
        "regressor__batch_size": [64, 128, 256],
    }
    search = RandomizedSearchCV(make_model(target=target_mode), space, n_iter=N_ITER_SEARCH, cv=cv,
                                scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=SEED, refit=True)
    search.fit(X_train, y_train)
    best_params = {k.replace("regressor__", ""): v for k, v in search.best_params_.items()}
    print(f"Candidates tried : {N_ITER_SEARCH} (x {N_SPLITS} folds)")
    print(f"Best CV RMSE     : {-search.best_score_:.3f}  (linear baseline CV RMSE: {ols_cv['cv_RMSE']:.3f})")
    print(f"Best settings    : { {k: (round(float(v), 6) if isinstance(v, float) else v) for k, v in best_params.items()} }")
    top = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(5)
    print("Top-5 CV RMSE    :", (-top["mean_test_score"]).round(3).tolist())
    print(LINE + "\n")

    # 5. ensemble of MLPs with different random seeds
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
    results = {**ref, "mlp_single": score_all(y_test, single_pred), "mlp_ensemble": score_all(y_test, ens_pred)}
    cmp = pd.DataFrame(results).T.round(4)
    print(cmp.to_string())
    cmp.to_csv(REPORT_DIR / "test_metrics_comparison.csv")
    final_pred, final_name = ens_pred, "mlp_ensemble"
    yt = y_test.to_numpy(float)
    f = results[final_name]
    print(f"\nFinal model ({final_name}): off by {f['MAE']:.1f} units on average (mean sales {yt.mean():.1f}), "
          f"R2 {f['R2']:.3f}")

    rng = np.random.default_rng(SEED)
    boot_rmse, boot_diff = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(yt), len(yt))
        r_m = np.sqrt(np.mean((yt[idx] - final_pred[idx]) ** 2))
        r_l = np.sqrt(np.mean((yt[idx] - ols_pred[idx]) ** 2))
        boot_rmse.append(r_m)
        boot_diff.append(r_l - r_m)            
    lo, hi = np.percentile(boot_rmse, [2.5, 97.5])
    dlo, dhi = np.percentile(boot_diff, [2.5, 97.5])
    print(f"Bootstrap 95% CI for MLP RMSE: [{lo:.3f}, {hi:.3f}]")
    print(f"Paired bootstrap, RMSE(linear) - RMSE(MLP): mean {np.mean(boot_diff):+.3f}, 95% CI [{dlo:.3f}, {dhi:.3f}]")
    verdict = ("the MLP is significantly better than the linear baseline" if dlo > 0 else
               "the MLP is significantly WORSE than the linear baseline" if dhi < 0 else
               "the difference is NOT statistically significant")
    print(f"-> {verdict} (interval {'excludes' if dlo > 0 or dhi < 0 else 'includes'} 0).")
    print(LINE + "\n")

    # 7. Generalisation: over-fitting check and learning curve 
    print(LINE)
    print("7. Generalisation")
    train_pred = np.clip(ens.predict(X_train), 0, None)
    tr = score_all(y_train, train_pred)
    print(f"Train RMSE {tr['RMSE']:.3f} vs test RMSE {f['RMSE']:.3f} (gap {f['RMSE'] - tr['RMSE']:+.3f}); "
          f"train R2 {tr['R2']:.3f} vs test R2 {f['R2']:.3f}")
    print("A large gap means over-fitting; a small gap with a low R2 means the limit is the data, not the model.")
    lc = []                            
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

    # 8. Interpretability: an MLP has no coefficients, so use model-agnostic tools
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
        g, c_mlp = partial_dependence(ens, X_test, col)
        _, c_lin = partial_dependence(ols, X_test, col)
        pdps[col] = (g, c_mlp, c_lin)
    print("Partial-dependence curves saved: a straight MLP curve = the effect is linear; a bend = the MLP found")
    print("non-linear behaviour that the linear baseline cannot represent.")
    print(LINE + "\n")

    # 9. Error analysis 
    print(LINE)
    print("9. Error analysis")
    resid = yt - final_pred
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
            if m.sum() >= 30:
                s, sl = score_all(yt[m], final_pred[m]), score_all(yt[m], ols_pred[m])
                slices.append({"slice": prefix.rstrip("_"), "value": val, "rows": int(m.sum()),
                               "MAE": s["MAE"], "RMSE": s["RMSE"], "R2": s["R2"], "Bias": s["Bias"],
                               "linear_MAE": sl["MAE"], "MAE_gain_vs_linear": sl["MAE"] - s["MAE"]})
    dts = pd.to_datetime(date_test).reset_index(drop=True)
    for i, chunk in enumerate(np.array_split(np.arange(len(dts)), 3), 1):
        s, sl = score_all(yt[chunk], final_pred[chunk]), score_all(yt[chunk], ols_pred[chunk])
        slices.append({"slice": "Time period", "value": f"T{i}: {dts[chunk[0]].date()} to {dts[chunk[-1]].date()}",
                       "rows": len(chunk), "MAE": s["MAE"], "RMSE": s["RMSE"], "R2": s["R2"], "Bias": s["Bias"],
                       "linear_MAE": sl["MAE"], "MAE_gain_vs_linear": sl["MAE"] - s["MAE"]})
    rob = pd.DataFrame(slices).round(3)
    rob.to_csv(REPORT_DIR / "robustness_by_slice.csv", index=False)
    print("\nRobustness by slice (MAE_gain_vs_linear > 0 = the MLP is better there):")
    print(rob.to_string(index=False))
    print(f"The MLP beats the linear model in {int((rob.MAE_gain_vs_linear > 0).sum())} of {len(rob)} slices.")

    worst = pd.DataFrame({"Date": date_test.values, "Actual": yt, "Predicted": final_pred.round(1),
                          "AbsError": np.abs(resid).round(1)}).sort_values("AbsError", ascending=False)
    worst.head(20).to_csv(REPORT_DIR / "worst_predictions.csv", index=False)

    pd.DataFrame({"Date": date_test.values, "Actual_Units": yt, "Predicted_Units": final_pred.round(2),
                  "Error": (final_pred - yt).round(2),
                  "Predicted_Units_Linear": ols_pred.round(2)}).to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    print(LINE + "\n")

    # 10. Plots
    print(LINE)
    print("10. Plots")
    _save_plots(ens, yt, final_pred, resid, imp, results, pdps, lc)
    print(f"Saved to {REPORT_DIR}")
    print(LINE + "\n")

    # 11. Shortlist summary + save
    print(LINE)
    print("11. Summary and saving")
    base_r = results["linear_regression_baseline"]
    print(f"Linear baseline : RMSE {base_r['RMSE']:.3f} | MAE {base_r['MAE']:.3f} | R2 {base_r['R2']:.3f}")
    print(f"MLP ensemble    : RMSE {f['RMSE']:.3f} | MAE {f['MAE']:.3f} | R2 {f['R2']:.3f}  "
          f"(RMSE change {f['RMSE'] - base_r['RMSE']:+.3f})")
    if dlo <= 0:
        print("The MLP does not clearly beat the linear model: most of the remaining error is noise the inputs")
        print("cannot explain, so extra model complexity brings little - a valid and reportable finding.")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": ens.models_, "features": cols, "best_params": best_params,
                 "target_mode": target_mode, "metrics": results}, MODEL_FILE)
    with open(REPORT_DIR / "metrics.json", "w") as fh:
        json.dump({"best_params": {k: str(v) for k, v in best_params.items()}, "target_mode": target_mode,
                   "bootstrap_rmse_ci95": [lo, hi], "paired_bootstrap_linear_minus_mlp_ci95": [dlo, dhi],
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
def _save_plots(ens, y, pred, resid, imp, results, pdps, lc):
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

    # Partial dependence: MLP curve vs linear line
    if pdps:
        fig, axs = plt.subplots(1, len(pdps), figsize=(4 * len(pdps), 3.6), squeeze=False)
        for ax, (col, (g, c_mlp, c_lin)) in zip(axs[0], pdps.items()):
            ax.plot(g, c_mlp, "o-", label="MLP"); ax.plot(g, c_lin, "s--", label="Linear")
            ax.set(title=col, xlabel="scaled value", ylabel="avg predicted units")
        axs[0][0].legend(); fig.suptitle("Partial dependence: shape of each effect", y=1.03)
        fig.tight_layout(); fig.savefig(REPORT_DIR / "partial_dependence.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Learning curve
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(lc.train_rows, lc.train_RMSE, "o-", label="train"); ax.plot(lc.train_rows, lc.test_RMSE, "s-", label="test")
    ax.set(xlabel="training rows (earliest dates)", ylabel="RMSE", title="Learning curve"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "learning_curve.png", dpi=150); plt.close(fig)

    # Model comparison
    m = pd.DataFrame(results).T[["MAE", "RMSE"]]
    ax = m.plot.bar(figsize=(7, 4), rot=15)
    ax.set_title("Error comparison (test set, lower = better)")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "model_comparison.png", dpi=150); plt.close()


if __name__ == "__main__":
    train_mlp()