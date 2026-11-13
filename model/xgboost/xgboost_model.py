"""
Model 3 - XGBoost (gradient boosting)

Why this model
--------------
Builds trees sequentially, each one correcting the errors of the previous ones.
Usually the strongest model on structured/tabular data, with built-in
regularisation. It is also the model chosen for the recommender system (as a
demand-forecast regressor), so tuning it here pays off twice.

Run:  python xgboost_model.py [--n-iter 20] [--cv-splits 3] [--data path/to/csv]
"""
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]                    # model/  (evaluation.py)
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR.parent / "src" / "backend"))       # data pipeline + data_modeling.py

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from data_modeling import RANDOM_STATE, build_preprocessor, prepare_data
from evaluation import (parse_args, plot_bar, run_model_experiment,
                        save_permutation_importance)

MODEL_NAME = "XGBoost"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_pipeline(ds) -> Pipeline:
    """XGBoost needs integer class labels (0,1,2) - prepare_data already provides them.
    n_jobs=1 inside the model because the search runs candidates in parallel."""
    return Pipeline([
        ("prep", build_preprocessor(ds.numeric_features, ds.categorical_features, scale=False)),
        ("clf", XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                              tree_method="hist", random_state=RANDOM_STATE, n_jobs=1)),
    ])


# Part B tuning. learning_rate x n_estimators trade off against each other;
# subsample / colsample_bytree / gamma / reg_* regularise to limit over-fitting.
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


def explain(model, ds, out_dir):
    names = model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names)
    imp = imp.sort_values(ascending=False)
    imp.to_csv(Path(out_dir) / "gain_importance.csv", header=["importance"])
    plot_bar(imp.head(15), "XGBoost - gain importance (top 15)",
             Path(out_dir) / "gain_importance.png", "importance")
    save_permutation_importance(model, ds, out_dir)


if __name__ == "__main__":
    args = parse_args("Train and evaluate the XGBoost model")
    dataset = prepare_data(args.data)
    # "balanced" sample weights give the rarer Understock/Overstock rows more influence
    # (XGBClassifier has no class_weight argument, so weights are passed to fit()).
    weights = compute_sample_weight("balanced", dataset.y_train)
    run_model_experiment(MODEL_NAME, build_pipeline(dataset), SEARCH_SPACE, dataset, OUT_DIR,
                         n_iter=args.n_iter, cv_splits=args.cv_splits,
                         fit_params={"clf__sample_weight": weights}, explain=explain)
