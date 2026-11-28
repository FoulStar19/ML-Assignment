import json
import os
import sys

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_modeling import load_model_data, model_output_dir, SEED
from model.evaluation import evaluate_model, default_vs_tuned_model_comparison

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
    return results

# ─────────────────────────────────────────────
# 2. TUNED MODEL (with hyperparameter tuning)
# ─────────────────────────────────────────────
def train_mlp_tuned():
    X_train, y_train, X_test, y_test, meta_test = load_model_data()

    # Define hyperparameter grid for MLP
    param_grid_mlp = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
        "batch_size": [64, 128, 256]
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
    return results

def save_model(model, model_name):
    model_dir = os.path.join(model_output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def save_results(results, model_name):
    results_dir = os.path.join(model_output_dir, model_name)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

def main():
    # Train and evaluate base model
    base_results = train_mlp_base()
    save_results(base_results, "mlp_base")

    # Train and evaluate tuned model
    tuned_results = train_mlp_tuned()
    save_results(tuned_results, "mlp_tuned")

    # Compare default vs tuned models
    default_vs_tuned_model_comparison(base_results, tuned_results)

if __name__ == "__main__":
    main()