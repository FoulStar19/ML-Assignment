import os
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_modeling import load_model_data, SEED
from model.evaluation import (
    evaluate_model, compare_model_results, save_results, save_model, generate_model_plots
)

# Models, results and plots are saved in an "outputs" folder inside this model's folder
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
# The dashboard (src/frontend/config.py) loads its model from this folder / OUTPUT_DIR
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1. BASE MODEL (no hyperparameter tuning)
# ─────────────────────────────────────────────
def train_mlp_base():
    X_train, y_train, X_test, y_test, meta_test = load_model_data()

    # Train a base MLP model (sklearn defaults: 1 hidden layer of 100 units, ReLU, Adam)
    mlp_model = MLPRegressor(random_state=SEED)
    mlp_model.fit(X_train, y_train)

    # Evaluate on the held-out (most recent) test dates
    train_pred_mlp = mlp_model.predict(X_train)
    test_pred_mlp = mlp_model.predict(X_test)

    results = {
        "mlp": {
            "train_metrics": evaluate_model(y_train, train_pred_mlp),
            "test_metrics": evaluate_model(y_test, test_pred_mlp)
        }
    }
    return results, {"mlp": mlp_model}

# ─────────────────────────────────────────────
# 2. TUNED MODEL (with hyperparameter tuning)
# ─────────────────────────────────────────────
def train_mlp_tuned():
    X_train, y_train, X_test, y_test, meta_test = load_model_data()

    # Define hyperparameter grid for MLP
    param_grid_mlp = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    }

    # Tune MLP Regressor
    # early_stopping holds out 10% of each training fold to stop when validation score stops improving
    mlp_model = MLPRegressor(
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=SEED,
    )
    mlp_search = GridSearchCV(
        mlp_model,
        param_grid_mlp,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
    )
    mlp_search.fit(X_train, y_train)

    best_mlp = mlp_search.best_estimator_

    # Evaluate on the held-out (most recent) test dates
    train_pred_mlp = best_mlp.predict(X_train)
    test_pred_mlp = best_mlp.predict(X_test)

    results = {
        "mlp": {
            "best_params": mlp_search.best_params_,
            "train_metrics": evaluate_model(y_train, train_pred_mlp),
            "test_metrics": evaluate_model(y_test, test_pred_mlp)
        }
    }
    return results, {"mlp": best_mlp}

def main():
    # Train base and tuned models
    base_results, base_models = train_mlp_base()
    tuned_results, tuned_models = train_mlp_tuned()

    # Compare default vs tuned models
    compare_model_results(base_results, tuned_results)

    # Save results and models
    save_results(base_results, "mlp_base", OUTPUT_DIR)
    save_model(base_models["mlp"], "mlp_base", OUTPUT_DIR)
    save_results(tuned_results, "mlp_tuned", OUTPUT_DIR)
    save_model(tuned_models["mlp"], "mlp_tuned", OUTPUT_DIR)

    # Plots for the tuned model
    X_train, y_train, X_test, y_test, _ = load_model_data()
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
    best_mlp = tuned_models["mlp"]

    # Model file used by the dashboard (a list of networks that it averages)
    save_model({"models": [best_mlp], "features": feature_names}, "mlp_model", MODEL_DIR)

    generate_model_plots(
        best_mlp, "mlp_tuned", X_train, y_train, X_test, y_test, OUTPUT_DIR,
        feature_names=feature_names,
        importances=None,                              # MLP has no built-in importances (permutation is used)
        loss_curves=(best_mlp.loss_curve_, None),      # training loss per epoch
        validation_param=("alpha", [0.0001, 0.001, 0.01, 0.1, 1.0]),
    )

if __name__ == "__main__":
    main()