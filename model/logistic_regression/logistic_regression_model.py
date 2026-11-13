import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]                    # model/  (evaluation.py)
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR.parent / "src" / "backend"))       # data pipeline + data_modeling.py

import pandas as pd
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_modeling import RANDOM_STATE, build_preprocessor, prepare_data
from evaluation import parse_args, plot_bar, run_model_experiment

MODEL_NAME = "Logistic Regression (Baseline)"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_pipeline(ds) -> Pipeline:
    """Scaling matters for LR (regularisation penalises large coefficients), so scale=True."""
    return Pipeline([
        ("prep", build_preprocessor(ds.numeric_features, ds.categorical_features, scale=True)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


# Part B tuning: C = inverse regularisation strength (small C = stronger regularisation);
# class_weight="balanced" up-weights the rarer Understock/Overstock classes.
SEARCH_SPACE = {
    "clf__C": loguniform(1e-3, 1e2),
    "clf__class_weight": [None, "balanced"],
}


def explain(model, ds, out_dir):
    """Coefficients per class: positive = pushes the prediction towards that class."""
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_                      # shape (n_classes, n_features)
    table = pd.DataFrame(coefs.T, index=names, columns=ds.class_names)
    table.to_csv(Path(out_dir) / "coefficients.csv")
    for cls in ("Understock", "Overstock"):
        top = table[cls].reindex(table[cls].abs().sort_values(ascending=False).index).head(15)
        plot_bar(top, f"Logistic Regression - top drivers of {cls}",
                 Path(out_dir) / f"coefficients_{cls.lower()}.png", "coefficient (standardised)")


def run(dataset, n_iter=20, cv_splits=3):
    """Tune + evaluate this model on an already prepared dataset (also called by pipeline.py)."""
    return run_model_experiment(MODEL_NAME, build_pipeline(dataset), SEARCH_SPACE, dataset, OUT_DIR,
                                n_iter=n_iter, cv_splits=cv_splits, explain=explain)


if __name__ == "__main__":
    args = parse_args("Train and evaluate the Logistic Regression baseline")
    run(prepare_data(args.data), n_iter=args.n_iter, cv_splits=args.cv_splits)