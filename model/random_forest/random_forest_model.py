import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]                    # model/  (evaluation.py)
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR.parent / "src" / "backend"))       # data pipeline + data_modeling.py

import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from data_modeling import RANDOM_STATE, build_preprocessor, prepare_data
from evaluation import (parse_args, plot_bar, run_model_experiment,
                        save_permutation_importance)

MODEL_NAME = "Random Forest"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_pipeline(ds) -> Pipeline:
    """Trees are scale-invariant, so no StandardScaler.
    n_jobs=1 inside the model because the search already runs candidates in parallel."""
    return Pipeline([
        ("prep", build_preprocessor(ds.numeric_features, ds.categorical_features, scale=False)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
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


def run(dataset, n_iter=20, cv_splits=3):
    """Tune + evaluate this model on an already prepared dataset (also called by pipeline.py)."""
    return run_model_experiment(MODEL_NAME, build_pipeline(dataset), SEARCH_SPACE, dataset, OUT_DIR,
                                n_iter=n_iter, cv_splits=cv_splits, explain=explain)


if __name__ == "__main__":
    args = parse_args("Train and evaluate the Random Forest model")
    run(prepare_data(args.data), n_iter=args.n_iter, cv_splits=args.cv_splits)