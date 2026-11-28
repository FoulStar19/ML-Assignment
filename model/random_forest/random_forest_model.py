import os
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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
def train_random_forest_base():
    X_train, y_train, X_test, y_test, meta_test = load_model_data()

    # Train a base Random Forest model
    rf_model = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Evaluate on the held-out (most recent) test dates
    train_pred_rf = rf_model.predict(X_train)
    test_pred_rf = rf_model.predict(X_test)

    results = {
        "random_forest": {
            "train_metrics": evaluate_model(y_train, train_pred_rf),
            "test_metrics": evaluate_model(y_test, test_pred_rf)
        }
    }
    return results, {"random_forest": rf_model}

# ─────────────────────────────────────────────
# 2. TUNED MODEL (with hyperparameter tuning)
# ─────────────────────────────────────────────
def train_random_forest_tuned():
    X_train, y_train, X_test, y_test, meta_test = load_model_data()

    # Define hyperparameter grid for Random Forest
    param_grid_rf = {
        "n_estimators": [100, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 3, 5],
    }

    # Tune Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    rf_search = GridSearchCV(
        rf_model,
        param_grid_rf,
        cv=5,
        scoring='neg_mean_squared_error',
    )
    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_

    # Evaluate on the held-out (most recent) test dates
    train_pred_rf = best_rf.predict(X_train)
    test_pred_rf = best_rf.predict(X_test)

    results = {
        "random_forest": {
            "best_params": rf_search.best_params_,
            "train_metrics": evaluate_model(y_train, train_pred_rf),
            "test_metrics": evaluate_model(y_test, test_pred_rf)
        }
    }
    return results, {"random_forest": best_rf}

def main():
    # Train base and tuned models
    base_results, base_models = train_random_forest_base()
    tuned_results, tuned_models = train_random_forest_tuned()

    # Compare default vs tuned models
    compare_model_results(base_results, tuned_results)

    # Save results and models
    save_results(base_results, "random_forest_base", OUTPUT_DIR)
    save_model(base_models["random_forest"], "random_forest_base", OUTPUT_DIR)
    save_results(tuned_results, "random_forest_tuned", OUTPUT_DIR)
    save_model(tuned_models["random_forest"], "random_forest_tuned", OUTPUT_DIR)

    # Plots for the tuned model
    X_train, y_train, X_test, y_test, _ = load_model_data()
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
    best_rf = tuned_models["random_forest"]

    # Model file used by the dashboard
    save_model(best_rf, "random_forest", MODEL_DIR)

    generate_model_plots(
        best_rf, "random_forest_tuned", X_train, y_train, X_test, y_test, OUTPUT_DIR,
        feature_names=feature_names,
        importances=best_rf.feature_importances_,
        validation_param=("max_depth", [3, 5, 10, 15, 20]),
    )

if __name__ == "__main__":
    main()