"""
Model 4 - Multi-Layer Perceptron (deep learning / neural network)

Why this model
--------------
A neural network learns its own non-linear feature combinations, so it
represents a completely different methodology from the linear model and the
two tree ensembles. Including it lets the report test whether deep learning
actually helps on structured retail data.

Note: scikit-learn's MLPClassifier has no class_weight option, so class
imbalance is judged with macro-F1 / per-class recall rather than accuracy.
early_stopping holds out 10% of the TRAINING rows (random) only to decide when
to stop; the real test set is never touched.

Run:  python mlp_model.py [--n-iter 20] [--cv-splits 3] [--data path/to/csv]
"""
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


if __name__ == "__main__":
    args = parse_args("Train and evaluate the MLP neural network")
    dataset = prepare_data(args.data)
    run_model_experiment(MODEL_NAME, build_pipeline(dataset), SEARCH_SPACE, dataset, OUT_DIR,
                         n_iter=args.n_iter, cv_splits=args.cv_splits, explain=explain)
