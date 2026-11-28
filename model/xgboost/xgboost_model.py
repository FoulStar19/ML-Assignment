import copy
import os
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_modeling import load_model_data, SEED
from model.evaluation import (
    evaluate_model, compare_model_results, save_results, save_model, generate_model_plots
)

# Models, results and plots are saved in an "outputs" folder inside this model's folder
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

def _to_float32(X):
    # Plain float32 arrays instead of DataFrames (avoids XGBoost's pandas/columnar path)
    return np.asarray(X, dtype=np.float32)

# ─────────────────────────────────────────────
# 1. BASE MODEL (no hyperparameter tuning)
# ─────────────────────────────────────────────
def train_xgboost_base():
    X_train, y_train, X_test, y_test, meta_test = load_model_data()
    X_train, X_test = _to_float32(X_train), _to_float32(X_test)

    # Train a base XGBoost model
    xgb_model = XGBRegressor(random_state=SEED)
    xgb_model.fit(X_train, y_train)

    # Evaluate on the held-out (most recent) test dates
    train_pred_xgb = xgb_model.predict(X_train)
    test_pred_xgb = xgb_model.predict(X_test)

    results = {
        "xgboost": {
            "train_metrics": evaluate_model(y_train, train_pred_xgb),
            "test_metrics": evaluate_model(y_test, test_pred_xgb)
        }
    }
    return results, {"xgboost": xgb_model}

# ─────────────────────────────────────────────
# 2. TUNED MODEL (with hyperparameter tuning)
# ─────────────────────────────────────────────
def train_xgboost_tuned():
    X_train, y_train, X_test, y_test, meta_test = load_model_data()
    X_train, X_test = _to_float32(X_train), _to_float32(X_test)

    # Define hyperparameter grid for XGBoost
    param_grid_xgb = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    }

    # Tune XGBoost Regressor (XGBoost threads inside, grid search in the main process)
    xgb_model = XGBRegressor(random_state=SEED, n_jobs=-1)
    xgb_search = GridSearchCV(
        xgb_model,
        param_grid_xgb,
        cv=5,
        scoring='neg_mean_squared_error',
    )
    xgb_search.fit(X_train, y_train)

    best_xgb = xgb_search.best_estimator_

    # Evaluate on the held-out (most recent) test dates
    train_pred_xgb = best_xgb.predict(X_train)
    test_pred_xgb = best_xgb.predict(X_test)

    results = {
        "xgboost": {
            "best_params": xgb_search.best_params_,
            "train_metrics": evaluate_model(y_train, train_pred_xgb),
            "test_metrics": evaluate_model(y_test, test_pred_xgb)
        }
    }
    return results, {"xgboost": best_xgb}

def main():
    # Train base and tuned models
    base_results, base_models = train_xgboost_base()
    tuned_results, tuned_models = train_xgboost_tuned()

    # Compare default vs tuned models
    compare_model_results(base_results, tuned_results)

    # Save results and models
    save_results(base_results, "xgboost_base", OUTPUT_DIR)
    save_model(base_models["xgboost"], "xgboost_base", OUTPUT_DIR)
    save_results(tuned_results, "xgboost_tuned", OUTPUT_DIR)
    save_model(tuned_models["xgboost"], "xgboost_tuned", OUTPUT_DIR)

    # Plots for the tuned model
    X_train, y_train, X_test, y_test, _ = load_model_data()
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
    X_train, X_test = _to_float32(X_train), _to_float32(X_test)
    best_xgb = tuned_models["xgboost"]

    # Model file used by the dashboard. It was trained on plain arrays, so attach the column
    # names (on a copy, so the model used for the plots below is untouched).
    dashboard_xgb = copy.deepcopy(best_xgb)
    dashboard_xgb.get_booster().feature_names = feature_names
    save_model(dashboard_xgb, "model", OUTPUT_DIR)

    # Refit the best configuration with an eval set to record loss per boosting round.
    # The "validation" curve here is the held-out test split, used for the plot only.
    loss_model = XGBRegressor(random_state=SEED, n_jobs=-1, **tuned_results["xgboost"]["best_params"])
    loss_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    evals = loss_model.evals_result()

    generate_model_plots(
        best_xgb, "xgboost_tuned", X_train, y_train, X_test, y_test, OUTPUT_DIR,
        feature_names=feature_names,
        importances=best_xgb.feature_importances_,
        loss_curves=(evals["validation_0"]["rmse"], evals["validation_1"]["rmse"]),
        validation_param=("max_depth", [2, 3, 5, 7, 9]),
    )

if __name__ == "__main__":
    main()