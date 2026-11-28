import json
import os
import sys

import joblib
import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_modeling import load_model_data, model_output_dir, SEED
from model.evaluation import evaluate_model, default_vs_tuned_model_comparison

# ─────────────────────────────────────────────
# 1. BASE MODEL (no hyperparameter tuning)
# ─────────────────────────────────────────────
def train_linear_regression_base():
    X_train, y_train, X_test, y_test = load_model_data()

    # Train a base Ridge model
    ridge_model = Ridge(random_state=SEED)
    ridge_model.fit(X_train, y_train)

    # Train a base Lasso model
    lasso_model = Lasso(random_state=SEED)
    lasso_model.fit(X_train, y_train)

    # Evaluate on the held-out (most recent) test dates
    train_pred_ridge = ridge_model.predict(X_train)
    test_pred_ridge = ridge_model.predict(X_test)
    train_pred_lasso = lasso_model.predict(X_train)
    test_pred_lasso = lasso_model.predict(X_test)
    
    results = {
        "ridge": {
            "train_metrics": evaluate_model(y_train, train_pred_ridge),
            "test_metrics": evaluate_model(y_test, test_pred_ridge)
        },
        "lasso": {
            "train_metrics": evaluate_model(y_train, train_pred_lasso),
            "test_metrics": evaluate_model(y_test, test_pred_lasso)
        }
    }
    return results

# ─────────────────────────────────────────────
# 2. TUNED MODEL (with hyperparameter tuning)
# ─────────────────────────────────────────────
def train_linear_regression_tuned():
    X_train, y_train, X_test, y_test = load_model_data()

    # Define hyperparameter grid for Ridge and Lasso
    param_grid_ridge = {"alpha": [0.1, 1.0, 10.0]}
    param_grid_lasso = {"alpha": [0.001, 0.01, 0.1]}

    # Tune Ridge Regression
    ridge_model = Ridge(random_state=SEED)
    ridge_search = GridSearchCV(ridge_model, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')
    ridge_search.fit(X_train, y_train)

    best_ridge = ridge_search.best_estimator_

    # Tune Lasso Regression
    lasso_model = Lasso(random_state=SEED)
    lasso_search = GridSearchCV(lasso_model, param_grid_lasso, cv=5, scoring='neg_mean_squared_error')
    lasso_search.fit(X_train, y_train)
    best_lasso = lasso_search.best_estimator_

    # Evaluate on the held-out (most recent) test dates
    train_pred_ridge = best_ridge.predict(X_train)
    test_pred_ridge = best_ridge.predict(X_test)
    train_pred_lasso = best_lasso.predict(X_train)
    test_pred_lasso = best_lasso.predict(X_test)
    results = {
        "ridge": {
            "best_params": ridge_search.best_params_,
            "train_metrics": evaluate_model(y_train, train_pred_ridge),
            "test_metrics": evaluate_model(y_test, test_pred_ridge)
        },
        "lasso": {
            "best_params": lasso_search.best_params_,
            "train_metrics": evaluate_model(y_train, train_pred_lasso),
            "test_metrics": evaluate_model(y_test, test_pred_lasso)
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
    # Train and evaluate base models
    base_results = train_linear_regression_base()
    save_results(base_results, "linear_regression_base")

    # Train and evaluate tuned models
    tuned_results = train_linear_regression_tuned()
    save_results(tuned_results, "linear_regression_tuned")

    # Compare default vs tuned models
    default_vs_tuned_model_comparison(base_results, tuned_results)

if __name__ == "__main__":
    main()