import os
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
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
def train_linear_regression_base():
    X_train, y_train, X_test, y_test, meta_test = load_model_data()

    # Train a base Ridge model
    ridge_model = Ridge(random_state=SEED)
    ridge_model.fit(X_train, y_train)

    # Evaluate on the held-out (most recent) test dates
    train_pred_ridge = ridge_model.predict(X_train)
    test_pred_ridge = ridge_model.predict(X_test)

    results = {
        "ridge": {
            "train_metrics": evaluate_model(y_train, train_pred_ridge),
            "test_metrics": evaluate_model(y_test, test_pred_ridge)
        }
    }
    return results, {"ridge": ridge_model}

# ─────────────────────────────────────────────
# 2. TUNED MODEL (with hyperparameter tuning)
# ─────────────────────────────────────────────
def train_linear_regression_tuned():
    X_train, y_train, X_test, y_test, meta_test = load_model_data()

    # Define hyperparameter grid for Ridge
    param_grid_ridge = {"alpha": [0.1, 1.0, 10.0]}

    # Tune Ridge Regression
    ridge_model = Ridge(random_state=SEED)
    ridge_search = GridSearchCV(ridge_model, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')
    ridge_search.fit(X_train, y_train)

    best_ridge = ridge_search.best_estimator_

    # Evaluate on the held-out (most recent) test dates
    train_pred_ridge = best_ridge.predict(X_train)
    test_pred_ridge = best_ridge.predict(X_test)

    results = {
        "ridge": {
            "best_params": ridge_search.best_params_,
            "train_metrics": evaluate_model(y_train, train_pred_ridge),
            "test_metrics": evaluate_model(y_test, test_pred_ridge)
        }
    }
    return results, {"ridge": best_ridge}

def main():
    # Train base and tuned models
    base_results, base_models = train_linear_regression_base()
    tuned_results, tuned_models = train_linear_regression_tuned()

    # Compare default vs tuned models
    compare_model_results(base_results, tuned_results)

    # Save results and models
    save_results(base_results, "linear_regression_base", OUTPUT_DIR)
    save_model(base_models["ridge"], "ridge_base", OUTPUT_DIR)
    save_results(tuned_results, "linear_regression_tuned", OUTPUT_DIR)
    save_model(tuned_models["ridge"], "ridge_tuned", OUTPUT_DIR)

    # Plots for the tuned model
    X_train, y_train, X_test, y_test, _ = load_model_data()
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
    best_ridge = tuned_models["ridge"]

    # Model file used by the dashboard
    save_model({"model": best_ridge, "features": feature_names}, "linear_regression_model", MODEL_DIR)

    generate_model_plots(
        best_ridge, "ridge_tuned", X_train, y_train, X_test, y_test, OUTPUT_DIR,
        feature_names=feature_names,
        importances=np.abs(best_ridge.coef_),          # |coefficient| as importance
        validation_param=("alpha", [0.01, 0.1, 1.0, 10.0, 100.0]),
    )

if __name__ == "__main__":
    main()