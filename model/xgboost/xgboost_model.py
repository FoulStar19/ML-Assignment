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


def fit_params(dataset):
    """Balanced sample weights (XGBoost has no class_weight option)."""
    return {"clf__sample_weight": compute_sample_weight("balanced", dataset.y_train)}


def run(dataset, n_iter=20, cv_splits=3):
    """Tune + evaluate this model on an already prepared dataset (also called by pipeline.py)."""
    return run_model_experiment(MODEL_NAME, build_pipeline(dataset), SEARCH_SPACE, dataset, OUT_DIR,
                                n_iter=n_iter, cv_splits=cv_splits,
                                fit_params=fit_params(dataset), explain=explain)


if __name__ == "__main__":
    args = parse_args("Train and evaluate the XGBoost model")
    run(prepare_data(args.data), n_iter=args.n_iter, cv_splits=args.cv_splits)