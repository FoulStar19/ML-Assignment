"""
Model 2 - Random Forest (bagging ensemble)

Why this model
--------------
Averages many decorrelated decision trees. It captures non-linear effects and
feature interactions (e.g. "high stock AND low recent sales AND promotion")
that Logistic Regression cannot, is robust to noise/outliers, and needs no
feature scaling. Feature importance supports the "key influencing variables"
analysis required in Part A.

Run:  python random_forest_model.py [--n-iter 20] [--data path/to/csv]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from common import config as cfg
from common.data_prep import build_preprocessor, prepare_data
from common.evaluation import (parse_args, plot_bar, run_model_experiment,
                               save_permutation_importance)

MODEL_NAME = "Random Forest"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_pipeline(ds) -> Pipeline:
    """Trees are scale-invariant, so no StandardScaler.
    n_jobs=1 inside the model because the search already runs candidates in parallel."""
    return Pipeline([
        ("prep", build_preprocessor(ds.numeric_features, ds.categorical_features, scale=False)),
        ("clf", RandomForestClassifier(random_state=cfg.RANDOM_STATE, n_jobs=1)),
    ])


# Part B tuning. Depth / leaf size control over-fitting; max_features controls how
# decorrelated the trees are; balanced class weights help the minority classes.
SEARCH_SPACE = {
    "clf__n_estimators": randint(100, 500),
    "clf__max_depth": [None, 8, 12, 16, 24],
    "clf__min_samples_split": randint(2, 20),
    "clf__min_samples_leaf": randint(1, 10),
    "clf__max_features": ["sqrt", "log2", 0.5],
    "clf__class_weight": [None, "balanced", "balanced_subsample"],
}


def explain(model, ds, out_dir):
    """Two views of importance: impurity-based (internal) and permutation-based (on test data)."""
    names = model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names)
    imp = imp.sort_values(ascending=False)
    imp.to_csv(Path(out_dir) / "impurity_importance.csv", header=["importance"])
    plot_bar(imp.head(15), "Random Forest - impurity importance (top 15)",
             Path(out_dir) / "impurity_importance.png", "importance")
    save_permutation_importance(model, ds, out_dir)


if __name__ == "__main__":
    args = parse_args("Train and evaluate the Random Forest model")
    dataset = prepare_data(args.data)
    run_model_experiment(MODEL_NAME, build_pipeline(dataset), SEARCH_SPACE, dataset, OUT_DIR,
                         n_iter=args.n_iter, cv_splits=args.cv_splits, explain=explain)
