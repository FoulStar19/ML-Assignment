import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                      # no display needed (works on servers / inside the UI backend)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, balanced_accuracy_score,
                             cohen_kappa_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, label_binarize

warnings.filterwarnings("ignore", category=FutureWarning)

LINE = "─" * 45

# ─────────────────────────────────────────────
# 0. SETTINGS
# ─────────────────────────────────────────────
SEED = 42
# src/backend/model/logistic_regression/<this file>  -> project root is 4 levels up.
# PROJECT_ROOT can override it (handy for tests / other machines).
ROOT_DIR = (Path(os.environ["PROJECT_ROOT"]) if "PROJECT_ROOT" in os.environ
            else Path(__file__).resolve().parents[4])
TRAIN_FILE = ROOT_DIR / "data" / "processed" / "train.csv"
TEST_FILE = ROOT_DIR / "data" / "processed" / "test.csv"
MODEL_FILE = ROOT_DIR / "models" / "logistic_regression_model.joblib"
REPORT_DIR = ROOT_DIR / "reports" / "logistic_regression"      # plots + csv/json for Student C and D

TARGET = "Units_Sold"
# Not used as inputs: label, bookkeeping columns, and Inventory_Level (kept for the recommender;
# it is also strongly tied to sales, so leaving it out keeps this a pure "demand drivers" model).
NON_FEATURES = [TARGET, "Date", "Transaction_ID", "Product_ID", "Inventory_Level"]
CLASSES = ["Low", "Medium", "High"]          # bottom / middle / top third of Units_Sold (train cut-offs)

N_SPLITS = 5            # time-ordered CV folds
N_ITER_SEARCH = 20      # random-search budget (random search is cheaper than a full grid, Part B.1)
SAGA_ITER, SAGA_TOL = 1000, 1e-3   # looser tolerance: same accuracy, much faster
N_BOOTSTRAP = 500       # resamples for the confidence interval on the test score
GROUP_PREFIXES = ["Product_Category_", "Store_Type_", "Weather_Condition_"]   # one-hot families for slices
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


def demand_cutoffs(units_train):
    # Cut points come from the TRAINING data only, so the test set does not influence the labels
    return units_train.quantile([1 / 3, 2 / 3]).tolist()


def to_demand_level(units, cutoffs):
    return pd.cut(units, bins=[-float("inf"), cutoffs[0], cutoffs[1], float("inf")],
                  labels=CLASSES).astype(str)


def decode_group(X, prefix):
    """Turn a one-hot family (e.g. Product_Category_*) back into one readable label per row."""
    cols = [c for c in X.columns if c.startswith(prefix)]
    return X[cols].idxmax(axis=1).str.replace(prefix, "", regex=False)


def score_all(model, X, y):
    """All headline metrics in one dict (macro = every class counts equally)."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=CLASSES, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_precision": p, "macro_recall": r, "macro_f1": f,
        "weighted_f1": f1_score(y, pred, average="weighted"),
        # quadratic kappa penalises Low->High far more than Low->Medium (the classes are ordered)
        "quadratic_kappa": cohen_kappa_score(y, pred, labels=CLASSES, weights="quadratic"),
        "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", labels=model.classes_),
        "log_loss": log_loss(y, proba, labels=model.classes_),
    }


def make_pipeline(feature_cols, use_interactions=False, use_l1_selection=False, **lr_params):
    """
    Build one candidate model. Switches map to Part A.5 (representations / feature selection):
      use_interactions  : add pairwise products of the main drivers (price x promotion, ...) so a
                          LINEAR model can capture simple non-linear effects
      use_l1_selection  : keep only features an L1-penalised model finds useful
    """
    steps = []
    if use_interactions:
        cols = [c for c in INTERACTION_COLS if c in feature_cols]
        steps.append(("interact", ColumnTransformer(
            [("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False), cols)],
            remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")))
    if use_l1_selection:
        steps.append(("select", SelectFromModel(
            LogisticRegression(l1_ratio=1.0, C=0.05, solver="saga", max_iter=SAGA_ITER, tol=SAGA_TOL,
                               random_state=SEED),
            threshold=1e-6)))  # keep every feature L1 did not zero out
    # saga supports l1 / l2 / elastic-net through l1_ratio (0 = ridge, 1 = lasso)
    steps.append(("clf", LogisticRegression(solver="saga", max_iter=SAGA_ITER, tol=SAGA_TOL, random_state=SEED, **lr_params)))
    return Pipeline(steps)


def feature_names_of(pipe, input_cols):
    """Follow the column names through the optional steps so coefficients stay readable."""
    names = np.array(input_cols)
    if not isinstance(pipe.named_steps.get("interact", "passthrough"), str):
        names = np.array(pipe.named_steps["interact"].get_feature_names_out())
    if not isinstance(pipe.named_steps.get("select", "passthrough"), str):
        names = names[pipe.named_steps["select"].get_support()]
    return names


# ─────────────────────────────────────────────
# 2. LOGISTIC REGRESSION BASELINE + REFINEMENT
# ─────────────────────────────────────────────
def train_logistic_regression():
    np.random.seed(SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the prepared data (already encoded and scaled by data_modeling)
    print(LINE)
    print("1. Data")
    X_train, units_train, date_train = load_split(TRAIN_FILE)
    X_test, units_test, date_test = load_split(TEST_FILE)
    cutoffs = demand_cutoffs(units_train)
    y_train = to_demand_level(units_train, cutoffs)
    y_test = to_demand_level(units_test, cutoffs)
    print(f"Train : {len(X_train):,} rows | Test : {len(X_test):,} rows | Inputs : {X_train.shape[1]}")
    print(f"Train dates {date_train.min()} to {date_train.max()} | Test dates {date_test.min()} to {date_test.max()}")
    print(f"Demand level from {TARGET}: Low <= {cutoffs[0]:.0f}, Medium <= {cutoffs[1]:.0f}, High above")
    print("Train classes:", y_train.value_counts(normalize=True).round(3).reindex(CLASSES).to_dict())
    print("Test classes :", y_test.value_counts(normalize=True).round(3).reindex(CLASSES).to_dict())
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing values reached the model - fix them in data_cleaning.py")
    # Time-ordered CV needs the rows in date order (data_modeling guarantees this)
    assert date_train.is_monotonic_increasing, "train.csv must be sorted by Date"
    print("Missing values : none (handled upstream in data_cleaning.py)")
    print(LINE + "\n")

    cols = list(X_train.columns)
    cv = TimeSeriesSplit(n_splits=N_SPLITS)       # always validate on dates AFTER the training dates

    # 2. Sanity floor: what does "no learning" score?
    print(LINE)
    print("2. Reference points (is the baseline learning anything?)")
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_scores = score_all(dummy, X_test, y_test)
    print(f"Majority-class dummy -> accuracy {dummy_scores['accuracy']:.3f}, macro-F1 {dummy_scores['macro_f1']:.3f}")
    print("Any real model must clearly beat this, otherwise the features carry no signal.")
    print(LINE + "\n")

    # 3. V0 - the plain baseline (defaults, nothing tuned). This is the number others must beat.
    print(LINE)
    print("3. V0 baseline (plain defaults)")
    v0 = make_pipeline(cols, C=1.0)               # L2 penalty by default
    v0.fit(X_train, y_train)
    v0_cv = cross_validate(clone(v0), X_train, y_train, cv=cv, n_jobs=-1,
                           scoring=["accuracy", "f1_macro", "neg_log_loss"])
    print(f"Time-series CV ({N_SPLITS} folds): accuracy {v0_cv['test_accuracy'].mean():.3f} "
          f"+/- {v0_cv['test_accuracy'].std():.3f} | macro-F1 {v0_cv['test_f1_macro'].mean():.3f} "
          f"+/- {v0_cv['test_f1_macro'].std():.3f}")
    print("A small +/- means the score is stable from fold to fold (stability / reliability).")
    print(LINE + "\n")

    # 4. Iterate (Part A.5 / A.6): try one improvement at a time and keep what really helps
    print(LINE)
    print("4. Refinement experiments (cross-validation on TRAIN only - test stays untouched)")
    variants = {
        "V0 baseline": v0,
        "V1 + class_weight": make_pipeline(cols, C=1.0, class_weight="balanced"),
        "V2 + interactions": make_pipeline(cols, use_interactions=True, C=1.0),
        "V3 + L1 feature selection": make_pipeline(cols, use_l1_selection=True, C=1.0),
    }
    rows = []
    for name, pipe in variants.items():
        res = cross_validate(clone(pipe), X_train, y_train, cv=cv, n_jobs=-1,
                             scoring=["accuracy", "f1_macro", "neg_log_loss"])
        rows.append({"variant": name, "cv_accuracy": res["test_accuracy"].mean(),
                     "cv_macro_f1": res["test_f1_macro"].mean(),
                     "cv_macro_f1_std": res["test_f1_macro"].std(),
                     "cv_log_loss": -res["test_neg_log_loss"].mean()})
    exp = pd.DataFrame(rows).round(4)
    print(exp.to_string(index=False))
    exp.to_csv(REPORT_DIR / "experiments_cv.csv", index=False)
    print(LINE + "\n")

    # 5. Part B.1 - automatic hyper-parameter search (random search, time-aware CV)
    print(LINE)
    print("5. Hyper-parameter tuning (RandomizedSearchCV, TimeSeriesSplit)")
    # The search also decides the representation (interactions / selection), so feature choices
    # are tuned automatically rather than by hand.
    base = Pipeline([("interact", "passthrough"), ("select", "passthrough"),
                     ("clf", LogisticRegression(solver="saga", max_iter=SAGA_ITER, tol=SAGA_TOL, random_state=SEED))])
    inter_cols = [c for c in INTERACTION_COLS if c in cols]
    inter_step = ColumnTransformer(
        [("poly", PolynomialFeatures(degree=2, include_bias=False), inter_cols)],
        remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
    select_step = SelectFromModel(
        LogisticRegression(l1_ratio=1.0, C=0.05, solver="saga", max_iter=SAGA_ITER, tol=SAGA_TOL, random_state=SEED),
        threshold=1e-6)  # keep every feature L1 did not zero out
    space = {
        "interact": ["passthrough", inter_step],
        "select": ["passthrough", select_step],
        "clf__C": loguniform(1e-3, 1e2),                 # regularisation strength (log scale)
        "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],    # 0 = L2, 1 = L1, between = elastic-net
        "clf__class_weight": [None, "balanced"],
    }
    search = RandomizedSearchCV(base, space, n_iter=N_ITER_SEARCH, cv=cv, scoring="f1_macro",
                                n_jobs=-1, random_state=SEED, refit=True)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"Candidates tried : {N_ITER_SEARCH} (x {N_SPLITS} folds)")
    print(f"Best CV macro-F1 : {search.best_score_:.3f}")
    clean = {k: (v if isinstance(v, str) else type(v).__name__ if k in ("interact", "select") else float(v) if k == "clf__C" else v)
             for k, v in search.best_params_.items()}
    print(f"Best settings    : {clean}")
    top = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(5)
    print("Top-5 mean CV macro-F1:", top["mean_test_score"].round(4).tolist())
    print(LINE + "\n")

    # 6. Final evaluation on the held-out LATEST dates (used once)
    print(LINE)
    print("6. Evaluation (test set = most recent dates)")
    results = {"dummy": dummy_scores,
               "baseline_v0": score_all(v0, X_test, y_test),
               "tuned": score_all(best, X_test, y_test)}
    cmp = pd.DataFrame(results).T.round(4)
    print(cmp.to_string())
    cmp.to_csv(REPORT_DIR / "test_metrics_comparison.csv")

    y_pred = best.predict(X_test)
    majority = y_test.value_counts(normalize=True).max()
    print(f"\nTuned accuracy {results['tuned']['accuracy']:.3f} vs always guessing the biggest class {majority:.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    print("\n" + classification_report(y_test, y_pred, labels=CLASSES, digits=3))

    # Bootstrap: how much could the score move just by luck of which rows are in the test set?
    rng = np.random.default_rng(SEED)
    boot = []
    yt, yp = y_test.to_numpy(), np.asarray(y_pred)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(yt), len(yt))
        boot.append(f1_score(yt[idx], yp[idx], average="macro"))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"Bootstrap 95% CI for macro-F1 ({N_BOOTSTRAP} resamples): [{lo:.3f}, {hi:.3f}]")
    print("If another model's score lies inside this band, the difference may just be noise.")
    print(LINE + "\n")

    # 7. Interpretability (Part A.3)
    print(LINE)
    print("7. Key drivers of demand")
    clf = best.named_steps["clf"]
    names = feature_names_of(best, cols)
    coefs = pd.DataFrame(clf.coef_.T, index=names, columns=clf.classes_)
    coefs["abs_max"] = coefs[list(clf.classes_)].abs().max(axis=1)
    coefs = coefs.sort_values("abs_max", ascending=False)
    coefs.to_csv(REPORT_DIR / "coefficients.csv")
    print("Largest coefficients (scaled inputs, so sizes are comparable; positive = pushes toward that class):")
    print(coefs.drop(columns="abs_max").head(10).round(3).to_string())
    print("Odds ratio for 'High': exp(coef) -> e.g. 1.5 means +1 std of the input makes High 50% more likely.")

    # Permutation importance on the ORIGINAL inputs: shuffle one column, see how much macro-F1 drops.
    # Works on the whole pipeline, so it is also valid when interactions / selection are used.
    perm = permutation_importance(best, X_test, y_test, scoring="f1_macro", n_repeats=5,
                                  random_state=SEED, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False)
    imp.to_csv(REPORT_DIR / "permutation_importance.csv", header=["f1_drop"])
    print("\nPermutation importance (drop in macro-F1 when the input is shuffled):")
    print(imp.head(8).round(4).to_string())
    print(LINE + "\n")

    # 8. Error analysis (Part A.4)
    print(LINE)
    print("8. Error analysis")
    order = {c: i for i, c in enumerate(CLASSES)}
    gap = np.abs(y_test.map(order).to_numpy() - pd.Series(y_pred).map(order).to_numpy())
    wrong = gap > 0
    print(f"Wrong predictions : {wrong.mean():.1%}")
    print(f"  off by one level (Low<->Medium, Medium<->High) : {(gap == 1).mean():.1%}")
    print(f"  severe (Low<->High)                            : {(gap == 2).mean():.1%}")
    print("Severe errors matter most in retail: they cause real stock-outs or over-stocking.")

    # Are the mistakes near the class boundaries? (i.e. 'almost right')
    near = ((units_test - cutoffs[0]).abs() <= 3) | ((units_test - cutoffs[1]).abs() <= 3)
    print(f"Error rate when Units_Sold is within 3 units of a cut-off : {wrong[near.to_numpy()].mean():.1%}")
    print(f"Error rate elsewhere                                       : {wrong[~near.to_numpy()].mean():.1%}")

    # Robustness across data slices: does performance hold for every store type / category / period?
    slices = []
    for prefix in GROUP_PREFIXES:
        label = decode_group(X_test, prefix)
        for val in sorted(label.unique()):
            m = (label == val).to_numpy()
            if m.sum() >= 30:                                   # ignore tiny groups (too noisy)
                slices.append({"slice": prefix.rstrip("_"), "value": val, "rows": int(m.sum()),
                               "accuracy": accuracy_score(yt[m], yp[m]),
                               "macro_f1": f1_score(yt[m], yp[m], average="macro")})
    dates = pd.to_datetime(date_test).reset_index(drop=True)
    for i, chunk in enumerate(np.array_split(np.arange(len(dates)), 3), 1):   # 3 chronological periods
        slices.append({"slice": "Time period", "value": f"T{i}: {dates[chunk[0]].date()} to {dates[chunk[-1]].date()}",
                       "rows": len(chunk), "accuracy": accuracy_score(yt[chunk], yp[chunk]),
                       "macro_f1": f1_score(yt[chunk], yp[chunk], average="macro")})
    rob = pd.DataFrame(slices).round(3)
    rob.to_csv(REPORT_DIR / "robustness_by_slice.csv", index=False)
    print("\nRobustness by slice (a big gap between slices = a bias / weak spot):")
    print(rob.to_string(index=False))
    print(f"Spread of macro-F1 across slices: {rob['macro_f1'].min():.3f} - {rob['macro_f1'].max():.3f}")
    print("Time periods show whether accuracy decays as the data moves away from the training dates (drift).")

    # Save row-level predictions for Student C (evaluation) and Student D (UI)
    proba = best.predict_proba(X_test)
    out = pd.DataFrame(proba, columns=[f"P_{c}" for c in best.classes_])
    out.insert(0, "Date", date_test.values)
    out.insert(1, "Actual_Units", units_test.values)
    out.insert(2, "Actual_Level", y_test.values)
    out.insert(3, "Predicted_Level", y_pred)
    out.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    print(LINE + "\n")

    # 9. Plots (Part A.3 visualisation)
    print(LINE)
    print("9. Plots")
    _save_plots(best, X_test, y_test, cm, coefs, imp, results)
    print(f"Saved to {REPORT_DIR}")
    print(LINE + "\n")

    # 10. Shortlisting summary + Save (Part A.7)
    print(LINE)
    print("10. Summary and saving")
    gain = results["tuned"]["macro_f1"] - results["baseline_v0"]["macro_f1"]
    print(f"Plain baseline macro-F1 : {results['baseline_v0']['macro_f1']:.3f}")
    print(f"Tuned model macro-F1    : {results['tuned']['macro_f1']:.3f}  ({gain:+.3f})")
    if gain < 0.01:
        print("Tuning adds little: the limit is the LINEAR form, not the settings -> motivates non-linear models (MLP/XGBoost/RF).")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "baseline": v0, "cutoffs": cutoffs, "classes": CLASSES,
                 "features": cols, "metrics": results}, MODEL_FILE)
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump({"cutoffs": cutoffs, "best_params": {k: str(v) for k, v in clean.items()},
                   "bootstrap_f1_ci95": [lo, hi], "results": results}, f, indent=2, default=float)
    print(f"Model saved   : {MODEL_FILE}")
    print(f"Reports saved : {REPORT_DIR}")
    print(LINE + "\n")

    return best


# ─────────────────────────────────────────────
# 3. PLOTS
# ─────────────────────────────────────────────
def _save_plots(model, X_test, y_test, cm, coefs, imp, results):
    # Confusion matrix (counts + row-normalised %)
    fig, ax = plt.subplots(figsize=(5, 4))
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Logistic Regression - confusion matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center",
                    color="white" if pct[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    # One-vs-rest ROC curves
    proba = model.predict_proba(X_test)
    Y = label_binarize(y_test, classes=list(model.classes_))
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, c in enumerate(model.classes_):
        fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC {roc_auc_score(Y[:, i], proba[:, i]):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest)"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "roc_curves.png", dpi=150); plt.close(fig)

    # Top coefficients for the 'High' class
    top = coefs["High"].reindex(coefs["High"].abs().sort_values(ascending=False).index)[:12][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top.index, top.values, color=np.where(top.values > 0, "#2a7ab0", "#c0504d"))
    ax.set_title("Top drivers of HIGH demand (coefficients)"); ax.set_xlabel("coefficient (scaled input)")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "top_coefficients_high.png", dpi=150); plt.close(fig)

    # Permutation importance
    t = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(t.index, t.values, color="#2a7ab0")
    ax.set_title("Permutation importance (macro-F1 drop)"); ax.set_xlabel("drop in macro-F1")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "permutation_importance.png", dpi=150); plt.close(fig)

    # Baseline vs tuned vs dummy
    m = pd.DataFrame(results).T[["accuracy", "macro_f1", "roc_auc_ovr"]]
    ax = m.plot.bar(figsize=(6, 4), rot=0)
    ax.set_title("Dummy vs baseline vs tuned (test set)"); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(REPORT_DIR / "model_comparison.png", dpi=150); plt.close()


if __name__ == "__main__":
    train_logistic_regression()
