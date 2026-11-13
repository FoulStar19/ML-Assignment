import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]                    # model/  (evaluation.py)
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR.parent / "src" / "backend"))       # data pipeline + data_modeling.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from data_modeling import RANDOM_STATE, build_preprocessor, prepare_data
from evaluation import parse_args, run_model_experiment, save_permutation_importance

MODEL_NAME = "MLP Neural Network"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_pipeline(ds) -> Pipeline:
    """Neural networks need scaled inputs, so scale=True."""
    return Pipeline([
        ("prep", build_preprocessor(ds.numeric_features, ds.categorical_features, scale=True)),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                              n_iter_no_change=10, random_state=RANDOM_STATE)),
    ])


# Part B tuning. alpha = L2 penalty (regularisation); architecture and learning rate
# control capacity and training stability.
SEARCH_SPACE = {
    "clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
    "clf__activation": ["relu", "tanh"],
    "clf__alpha": loguniform(1e-5, 1e-1),
    "clf__learning_rate_init": loguniform(1e-4, 1e-2),
    "clf__batch_size": [64, 128, 256],
}


def explain(model, ds, out_dir):
    """Training curves (to check convergence / over-fitting) + permutation importance."""
    clf = model.named_steps["clf"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(clf.loss_curve_)
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="log-loss")
    axes[1].plot(clf.validation_scores_)
    axes[1].set(title="Validation accuracy (early stopping)", xlabel="epoch", ylabel="accuracy")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "training_curves.png", dpi=150)
    plt.close(fig)
    save_permutation_importance(model, ds, out_dir)


def run(dataset, n_iter=20, cv_splits=3):
    """Tune + evaluate this model on an already prepared dataset (also called by pipeline.py)."""
    return run_model_experiment(MODEL_NAME, build_pipeline(dataset), SEARCH_SPACE, dataset, OUT_DIR,
                                n_iter=n_iter, cv_splits=cv_splits, explain=explain)


if __name__ == "__main__":
    args = parse_args("Train and evaluate the MLP neural network")
    run(prepare_data(args.data), n_iter=args.n_iter, cv_splits=args.cv_splits)